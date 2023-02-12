import numpy as np
from configs import config
import yaml
import os, glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

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

sorted(glob.glob('*.png'))
predictions_path = "output/valid/sequences/08/predictions/"
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
    
    fig = plt.figure(figsize=(25, 15))
    ax = fig.add_subplot(projection='3d')
    # ax = plt.axes(projection='3d')
    ax.set_xlim([0, 256])
    ax.set_ylim([0, 256])
    ax.set_zlim([0, 32])
    ax.set_box_aspect((np.ptp(locs_x), np.ptp(locs_y), np.ptp(locs_z)))  # aspect ratio is 1:1:1 in data space
    ax.scatter(locs_x, locs_y, locs_z, c=colors, edgecolors='k',linewidths=0.25)
    ax.view_init(elev=25., azim=150.)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.grid(False)
    plt.axis('off')
    plt.savefig('output/imgs/screenshot_%03d.png'%i, bbox_inches='tight')


# Create video
sorted(glob.glob('*.png'))
predictions_path = "output/imgs/"
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
video_name = 'output/gifs/video194.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, fourcc, 5.0, (width,height))

for i, image_name in enumerate(paths):
#     if i>10:
#         break
    img = cv2.imread(image_name)
    img = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)
    video.write(img)

cv2.destroyAllWindows()
video.release()