import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

classes_colors = [
    # [0, 0, 254],     # "unlabeled"
    [245, 150, 100], # "car" 10
    [245, 230, 100], # "bicycle" 11
    [150, 60, 30],   # "motorcycle" 15
    [180, 30, 80],   # "truck" 18
    [254, 0, 0],     # "other-vehicle" 20
    [30, 30, 254],   # "person" 30
    [200, 40, 254],  # "bicyclist" 31 
    [90, 30, 150],   # "motorcyclist" 32
    [255, 0, 255], # "road" 40
    [254, 150, 254], # "parking" 44
    [75, 0, 75],     # "sidewalk" 48
    [75, 0, 175],    # "other-ground" 49 
    [0, 200, 254],   # "building" 50
    [50, 120, 254],  # "fence" 51
    [0, 175, 0],     # "vegetation" 70
    [0, 60, 135],    # "trunk" 71
    [80, 240, 150],  # "terrain" 72
    [150, 240, 254], # "pole" 80
    [0, 0, 254]      # "traffic-sign" 81
]

classes_colors_poss = [
    # [0, 0, 0],    # 0: "unlabeled"
    [30, 30, 255],     # 1: "people"
    [200, 40, 255],    # 2: "rider"
    [245, 150, 100],   # 3: "car"
    [0,60,135],        # 4: "trunk"
    [0, 175, 0],       # 5: "plants"
    [0, 0, 255],       # 6: "traffic sign"
    [150, 240, 255],   # 7: "pole"
    [0, 200, 255],     # 8: "building"
    [50, 120, 255],    # 9: "fence"
    [245, 230, 100],   # 10: "bike"
    [128, 128, 128]    # 11: "ground"
]


def convert_rgb_to_hex(bgr_list):
    hex_colors = []
    for bgr in bgr_list:
        rgb = bgr[::-1]  # Reverse the color to RGB
        hex_color = ''.join([format(val, '02x') for val in rgb])
        hex_colors.append(int(hex_color, 16))
    return hex_colors


shapes = {"256": [256, 256, 32, 3], "128": [128, 128, 16, 2], "64": [64, 64, 8, 1]}


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a
def rgb_to_hex(rgb):
    rgb = totuple(rgb)
    return '0x%02x%02x%02x' % rgb

def hex_classes_cmap_deprecated():
    classes_cmap = []
    for rgb in (np.uint8(classes_colors)):
        hex_str = rgb_to_hex(rgb)
        hex_int = int(hex_str, 16)
        new_int = hex_int
        classes_cmap.append(new_int)

    classes_cmap = np.array(classes_cmap)
    return classes_cmap

def hex_classes_cmap():
    return convert_rgb_to_hex(classes_colors)

def hex_classes_cmap_poss():
    return convert_rgb_to_hex(classes_colors_poss)

def hex_lidar_intensities_cmap():
    viridis = cm.get_cmap('plasma', 128)
    cmap = []
    for rgb in (np.uint8((viridis.colors)[:,:-1]*255.0)):
        hex_str = rgb_to_hex(rgb)
        hex_int = int(hex_str, 16)
        new_int = hex_int + 0x200
        cmap.append(new_int)
    cmap = np.array(cmap)
    return cmap

def classes_cmap():
    
    classes_cmap = []
    for rgb in (np.uint8(classes_colors)):
        hex_str = rgb_to_hex(rgb)
        hex_int = int(hex_str, 16)
        new_int = hex_int + 0x200
        classes_cmap.append(new_int)
    return classes_cmap
    
def plot_3d_voxels(voxels, level="256", input=False):
    voxels = np.uint8(voxels)
    mult = shapes[level][3]
    if input:
        viridis = cm.get_cmap('plasma', 128)
        voxels[voxels>127] = 127
        classes_colors2 = np.uint8(np.array(viridis.colors)*255.0)
        level = "256"
    else:
        classes_colors2 = np.uint8(classes_colors)
        color_0 = np.array([[0,0,0]])
        voxels[voxels==255] = 0
        classes_colors2 = np.concatenate((color_0,classes_colors))
    # semantic_prediction = classes_colors[prediction]
    locs_x, locs_y, locs_z = np.nonzero(voxels)
    locs = np.zeros((locs_x.shape[0],3))
    locs[:,0] = locs_x
    locs[:,1] = locs_y
    locs[:,2] = locs_z
    colors = voxels[locs_x,locs_y,locs_z]
    colors = classes_colors2[colors]
    colors = np.float32(colors)/255.0
    fig = plt.figure(figsize=(6*mult, 4*mult))
    ax = fig.add_subplot(projection='3d')
    # ax = plt.axes(projection='3d')
    ax.set_xlim([0, shapes[level][0]])
    ax.set_ylim([0, shapes[level][1]])
    ax.set_zlim([0, shapes[level][2]])
    ax.set_box_aspect((np.ptp(locs_x), np.ptp(locs_y), np.ptp(locs_z)))  # aspect ratio is 1:1:1 in data space
    ax.scatter(locs_x, locs_y, locs_z, c=colors, edgecolors='k',linewidths=0.25)
    ax.view_init(elev=25., azim=150.)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.grid(False)
    # plt.axis('off')
    plt.show()

def plot_2d_input(features2d):
    for i in range(7):
        mean_feature = features2d[i,:,:].cpu().numpy()
        invalid = (mean_feature==255)
        mean_feature = mean_feature - np.amin(mean_feature)
        mean_feature = mean_feature/np.amax(mean_feature)
        mean_feature[invalid]=0
        plt.imshow(mean_feature)
        plt.show()

def plot_bev(bev_tensor):
    # Plot prediction
    cmap=np.float32(classes_colors)/255.0
    cmap = np.insert(cmap,0,[0.99,0.99,0.99],axis=0)
    bev = bev_tensor.cpu().detach().numpy()
    bev[bev==255]=0
    output = cmap[np.uint8(bev.flatten())]
    R,C = bev.shape
    output = output.reshape((R, C, -1))
    plt.imshow(output)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.show()

def plot_bev_input(bev_tensor):
    # Plot prediction
    viridis = cm.get_cmap('plasma', 128)
    cmap = np.uint8(np.array(viridis.colors)*255.0)
    
    bev = bev_tensor.cpu().detach().numpy()
    bev[bev>=128]=0
    output = cmap[np.uint8(bev.flatten())]
    R,C = bev.shape
    output = output.reshape((R, C, -1))
    plt.imshow(output)
    plt.gca().invert_yaxis()
    plt.show()

def labels_to_cmap2d(tensor):
    cmap=np.float32(classes_colors)/255.0
    cmap = np.insert(cmap,0,[0.99,0.99,0.99],axis=0)
    tensor = tensor.cpu().detach().numpy()
    tensor[tensor==255]=0
    output = cmap[np.uint8(tensor.flatten())]
    B,R,C = tensor.shape
    output = output.reshape((B,R, C, -1))
    output = np.transpose(output,(0,3,1,2))
    return output

def input_to_cmap2d(tensor):
    viridis = cm.get_cmap('plasma', 128)
    cmap = np.array(viridis.colors)[:,:3]
    tensor = tensor.cpu().detach().numpy()
    tensor[tensor>=128]=0
    output = cmap[np.uint8(tensor.flatten())]
    B,R,C = tensor.shape
    output = output.reshape((B,R, C, -1))
    output = np.transpose(output,(0,3,1,2))
    return output

def plot_3d_pointcloud(coords_tensor, labels_tensor, input = False):
    labels = labels_tensor.int().cpu().detach().numpy()
    coords = coords_tensor.int().cpu().detach().numpy()
    if input:
        viridis = cm.get_cmap('plasma', 128)
        classes_colors2 = np.uint8(np.array(viridis.colors)*255.0)
    else:
        classes_colors2 = np.uint8(classes_colors)
        # color_0 = np.array([[0,0,0]])
        # classes_colors2 = np.concatenate((color_0,classes_colors))

        
        coords = coords[labels!=-100]
        labels = labels[labels!=-100]

        print(np.unique(labels))

    colors = classes_colors2[labels]
    colors = np.float32(colors)/255.0

    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((np.ptp(coords[:,0]), np.ptp(coords[:,1]), np.ptp(coords[:,2])))  # aspect ratio is 1:1:1 in data space
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=0.5, c=colors, edgecolors='k',linewidths=0.1)
    ax.view_init(elev=25., azim=150.)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()