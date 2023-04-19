import numpy as np
import yaml
import os, glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import time
import cv2
import sys
sys.path.append("../")
from structures import collect
from semantic_kitti_dataset import SemanticKITTIDataset, MergeTest, Merge
from configs import config

# Color map for lidar intensities
def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a
def rgb_to_hex(rgb):
    rgb = totuple(rgb)
    return '0x%02x%02x%02x' % rgb
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
viridis = cm.get_cmap('plasma', 128)
cmap = []
for rgb in (np.uint8((viridis.colors)[:,:-1]*255.0)):
    hex_str = rgb_to_hex(rgb)
    hex_int = int(hex_str, 16)
    new_int = hex_int + 0x200
    cmap.append(new_int)

classes_colors = [[245, 150, 100],
[245, 230, 100],
[150, 60, 30],
[180, 30, 80],
[255, 0, 0],
[30, 30, 255],
[200, 40, 255],
[90, 30, 150],
[255, 0, 255],
[255, 150, 255],
[75, 0, 75],
[75, 0, 175],
[0, 200, 255],
[50, 120, 255],
[0, 175, 0],
[0, 60, 135],
[80, 240, 150],
[150, 240, 255],
[0, 0, 255],
[255, 255, 50]]
classes_cmap = []
for rgb in (np.uint8(classes_colors)):
    hex_str = rgb_to_hex(rgb)
    hex_int = int(hex_str, 16)
    new_int = hex_int + 0x200
    classes_cmap.append(new_int)

# load filenames of predications
skip=2
sorted(glob.glob('*.png'))
sequence = "08"
split = "valid"
predictions_path = "output/{}/sequences/{}/predictions/".format(split, sequence)
paths = []
for infile in sorted(glob.glob( os.path.join(predictions_path, '*.label') )):
    paths.append(infile)
print(len(paths))

classes_colors2 = np.uint8(classes_colors)
color_0 = np.array([[0,0,0]])
classes_colors2 = np.concatenate((color_0,classes_colors))

# Create images
plt.ioff()

for i, path in enumerate(tqdm(paths)):
    if i % skip == 0:
        continue
    
    prediction = np.fromfile(path, dtype=np.uint16) 
    prediction = np.int32(prediction.reshape(config.COMPLETION.FULL_SCALE))
    # remap
    config_file = os.path.join('configs/semantic-kitti.yaml')
    kitti_config = yaml.safe_load(open(config_file, 'r'))
    remapdict = kitti_config["learning_map"]

    # make lookup table for mapping
    maxkey = max(remapdict.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())
    prediction = remap_lut[prediction]
    
    locs_x, locs_y, locs_z = np.nonzero(prediction)
    locs = np.zeros((locs_x.shape[0],3))
    locs[:,0] = locs_x
    locs[:,1] = locs_y
    locs[:,2] = locs_z
    colors = prediction[locs_x,locs_y,locs_z]
    colors = classes_colors2[colors]
    colors = np.float32(colors)/255.0
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    # ax = plt.axes(projection='3d')
    ax.set_xlim([0, 256])
    ax.set_ylim([0, 256])
    ax.set_zlim([0, 32])
    ax.set_box_aspect((np.ptp(locs_x), np.ptp(locs_y), np.ptp(locs_z)))  # aspect ratio is 1:1:1 in data space
    ax.scatter(locs_x, locs_y, locs_z, s=10.0,  c=colors, edgecolors='k',linewidths=0.1)
    ax.view_init(elev=20., azim=180.)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.grid(False)
    plt.axis('off')
    # plt.margins(0,0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.gca().zaxis.set_major_locator(plt.NullLocator())
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    #         hspace = 0, wspace = 0)
    # plt.subplots_adjust(0,0,1,1,0,0)
    # for ax in fig.axes:
    #     ax.axis('off')
    #     ax.margins(0,0,0)
    #     ax.xaxis.set_major_locator(plt.NullLocator())
    #     ax.yaxis.set_major_locator(plt.NullLocator())
    # plt.tight_layout(pad=0)
    plt.savefig('output/imgs/{}/screenshot_{:05d}.png'.format(sequence,i), bbox_inches='tight',dpi=100, pad_inches=0)
    plt.close()
    

# Create video output
sorted(glob.glob('*.png'))
predictions_path = "output/imgs/{}/".format(sequence)
paths = []

for infile in sorted(glob.glob( os.path.join(predictions_path, '*.png') )):
    paths.append(infile)
print(len(paths))
print(paths[0])
frame = cv2.imread(paths[0])
height, width, layers = frame.shape
# print(frame.shape)
width = 586
height = 620
video_name = 'output/gifs/video{}.mp4'.format(sequence)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=10.0, frameSize=(width,height), isColor=True)

for i, image_name in enumerate(paths):
#     if i>10:
#         break
    img = cv2.imread(image_name)
    img = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)
    video.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()


## Create input images

config.GENERAL.OVERFIT = False
config.TRAIN.AUGMENT = False
config.TRAIN.NOISE_LEVEL = 0.0
config.TRAIN.NUM_WORKERS = 1
train_dataset = SemanticKITTIDataset("test",do_overfit=config.GENERAL.OVERFIT, num_samples_overfit=config.GENERAL.NUM_SAMPLES_OVERFIT, augment=config.TRAIN.AUGMENT)
train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        collate_fn=MergeTest,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )



viridis = cm.get_cmap('plasma', 128)
viridis_colors = np.array(viridis.colors)

pbar = tqdm(train_data_loader)
for i, batch in enumerate(pbar):
# for i in range(1):
    # batch = next(iter(train_data_loader))
    if i % skip == 0:
        continue
    filenames, complet_inputs, _, _ = batch
    coords = collect(complet_inputs, "complet_coords").squeeze()
    complet_features = collect(complet_inputs, "complet_features")[0]
    
    coords_np = coords[:,1:].detach().cpu().numpy()
    coords_np = coords_np.astype(int)
    voxels = np.zeros((256,256,32))
    # voxels[coords_np[:,0],coords_np[:,1],coords_np[:,2]]=np.int32(complet_features.detach().cpu().numpy()[0]*10.0)
    voxels[coords_np[:,0],coords_np[:,1],coords_np[:,2]]=(np.uint8(complet_features.detach().cpu().numpy()*126.0)+1.0)
    
    locs_x, locs_y, locs_z = np.nonzero(voxels)
    locs = np.zeros((locs_x.shape[0],3))
    locs[:,0] = locs_x
    locs[:,1] = locs_y
    locs[:,2] = locs_z
    colors = voxels[locs_x,locs_y,locs_z]
    colors = viridis_colors[np.uint8(colors)]
    # colors = np.float32(colors)/255.0
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    # ax = plt.axes(projection='3d')
    ax.set_xlim([0, 256])
    ax.set_ylim([0, 256])
    ax.set_zlim([0, 32])
    ax.set_box_aspect((np.ptp(locs_x), np.ptp(locs_y), np.ptp(locs_z)))  # aspect ratio is 1:1:1 in data space
    ax.scatter(locs_x, locs_y, locs_z, c=colors, edgecolors='k',linewidths=0.1, s=3.0)
    ax.view_init(elev=20., azim=180.)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.grid(False)
    plt.axis('off')
    plt.savefig('output/imgs_input/{}/screenshot_{:05d}.png'.format(sequence,i), bbox_inches='tight')
    plt.close()

# Create video input
sorted(glob.glob('*.png'))
predictions_path = "output/imgs_input/{}/".format(sequence)
paths = []

for infile in sorted(glob.glob( os.path.join(predictions_path, '*.png') )):
    paths.append(infile)
print(len(paths))
print(paths[0])
frame = cv2.imread(paths[0])
height, width, layers = frame.shape
# print(frame.shape)
width = 732
height = 775
video_name = 'output/gifs/video_input{}.mp4'.format(sequence)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=10.0, frameSize=(width,height), isColor=True)

for i, image_name in enumerate(paths):
#     if i>10:
#         break
    img = cv2.imread(image_name)
    img = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)
    video.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()