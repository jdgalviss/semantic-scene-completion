import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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

shapes = {"256": [256, 256, 32, 3], "128": [128, 128, 16, 2], "64": [64, 64, 8, 1]}


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a
def rgb_to_hex(rgb):
    rgb = totuple(rgb)
    return '0x%02x%02x%02x' % rgb

def lidar_intensities_cmap():
    viridis = cm.get_cmap('plasma', 128)
    cmap = []
    for rgb in (np.uint8((viridis.colors)[:,:-1]*255.0)):
        hex_str = rgb_to_hex(rgb)
        hex_int = int(hex_str, 16)
        new_int = hex_int + 0x200
        cmap.append(new_int)
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


