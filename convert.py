import os
import numpy as np

base_dir = "C:/Users/Mike/Documents/nrgbd/Guided-Depth-Map-Super-resolution-A-Survey/code/data/Depth/NYU/npy/nyu/train/"
gt_dir = base_dir + "gt/"
rgb_dir = base_dir + "rgb/"
output_dir = "./nyu_data"

gt_files = sorted(os.listdir(gt_dir))
rgb_files = sorted(os.listdir(rgb_dir))


depth_data = []
image_data = []
minmax_data = []

for gt_f, rgb_f in zip(gt_files, rgb_files):
    depth = np.load(os.path.join(gt_dir, gt_f))
    rgb = np.load(os.path.join(rgb_dir, rgb_f))
    
    depth_data.append(depth)
    image_data.append(rgb)
    
    # Store per-sample min and max
    minmax_data.append([depth.min(), depth.max()])


depth_array = np.array(depth_data)
image_array = np.array(image_data)
minmax_array = np.array(minmax_data).T

os.makedirs(output_dir, exist_ok=True)
# np.save(os.path.join(output_dir, "test_depth.npy"), depth_array)
# np.save(os.path.join(output_dir, "test_images_v2.npy"), image_array)
np.save(os.path.join(output_dir, "test_minmax.npy"), minmax_array)
