import os
import time
import copy
import argparse
from io import StringIO

import numpy as np
from numpy import linalg as LA

import open3d as o3d # intensity is read/written through the o3d.t (tensor) API; pypcd does not work with numpy 2

script_dir = os.path.dirname(os.path.abspath(__file__))
jet_table = np.load(os.path.join(script_dir, 'jet_table.npy'))
bone_table = np.load(os.path.join(script_dir, 'bone_table.npy'))

color_table = jet_table
color_table_len = color_table.shape[0]


##########################
# User only consider this block
##########################

parser = argparse.ArgumentParser(description="Stack the SC-PGO keyframe scans into one map, show it and save it as a PCD with intensity.")
parser.add_argument("data_dir", nargs="?", default="save_data", help="SC-PGO save_directory, holding Scans/ and *_poses.txt (default: ./save_data)")
parser.add_argument("--near", type=float, default=0.7,
                    help="drop points closer than this to the lidar, in m; 0 keeps them all (default: 0.7). "
                         "The original 2 m was meant for the car of the SC-PGO datasets and drops most of an indoor scan.")
parser.add_argument("--color", choices=["auto", "intensity", "height"], default="auto",
                    help="viewer coloring; auto colors by height when every intensity is 0, as with the Gazebo lidars (default: auto)")
parser.add_argument("--no-floor", action="store_true",
                    help="hide the floor in the viewer: the points up to --floor-margin above the estimated floor height. The saved map keeps them.")
parser.add_argument("--floor-margin", type=float, default=0.15, help="height above the estimated floor that --no-floor hides, in m (default: 0.15)")
args = parser.parse_args()

data_dir = os.path.join(os.path.expanduser(args.data_dir), "")
scan_idx_range_to_stack = [0, 200] # if you want a whole map, use [0, len(scan_files)]
node_skip = 1

num_points_in_a_scan = 150000 # for reservation (save faster) // e.g., use 150000 for 128 ray lidars, 100000 for 64 ray lidars, 30000 for 16 ray lidars, if error occured, use the larger value.

is_live_vis = False # recommend to use false
is_o3d_vis = True
intensity_color_max = 200

is_near_removal = args.near > 0
thres_near_removal = args.near # meter (to remove platform-myself structure ghost points)

##########################


def estimate_floor_z(z):
    # The floor is the most populated 5 cm height band in the lower half of the map.
    # Looking only below the median keeps a ceiling from winning.
    z_low = z[z <= np.median(z)]
    num_bins = max(1, int(np.ceil((z_low.max() - z_low.min()) / 0.05)))
    counts, edges = np.histogram(z_low, bins=num_bins)
    peak = np.argmax(counts)
    return 0.5 * (edges[peak] + edges[peak + 1])


#
scan_dir = data_dir + "Scans"
scan_files = os.listdir(scan_dir)
scan_files.sort()

poses = []
pose_file = data_dir + "optimized_poses.txt"
if not os.path.exists(pose_file): # runs recorded before SC-PGO wrote optimized_poses.txt
    pose_file = data_dir + "odom_poses.txt"
    print("optimized_poses.txt not found, falling back to (unoptimized)", pose_file)
f = open(pose_file, 'r')
while True:
    line = f.readline()
    if not line: break
    pose_SE3 = np.asarray([float(i) for i in line.split()])
    pose_SE3 = np.vstack( (np.reshape(pose_SE3, (3, 4)), np.asarray([0,0,0,1])) )
    poses.append(pose_SE3)
f.close()

# SC-PGO rewrites the pose files at 1 Hz, so the newest keyframes may not have a pose yet,
# and a pose file cut off by a shutdown mid-write can be much shorter than Scans/
num_nodes = min(len(scan_files), len(poses))
if len(poses) < len(scan_files):
    print("WARNING:", len(scan_files), "scans but only", len(poses), "poses in", pose_file, "- stacking the first", num_nodes, "scans only")


#
assert (scan_idx_range_to_stack[1] > scan_idx_range_to_stack[0])
print("Merging scans from", scan_idx_range_to_stack[0], "to", scan_idx_range_to_stack[1])


nodes_count = 0

# The scans from 000000.pcd should be prepared if it is not used (because below code indexing is designed in a naive way)

# manually reserve memory for fast write
num_all_points_expected = int(num_points_in_a_scan * np.round((scan_idx_range_to_stack[1] - scan_idx_range_to_stack[0])/node_skip))

np_xyz_all = np.empty([num_all_points_expected, 3])
np_intensity_all = np.empty([num_all_points_expected, 1])
curr_count = 0
scan_ends = [] # end of each stacked scan in np_xyz_all, for the live viewer

for node_idx in range(num_nodes):
    if(node_idx < scan_idx_range_to_stack[0] or node_idx >= scan_idx_range_to_stack[1]):
        continue

    nodes_count = nodes_count + 1
    if( nodes_count % node_skip != 0):
        if(node_idx != scan_idx_range_to_stack[0]): # to ensure the vis init
            continue

    print("read keyframe scan idx", node_idx)

    scan_pose = poses[node_idx]

    scan_path = os.path.join(scan_dir, scan_files[node_idx])
    scan_pcd = o3d.io.read_point_cloud(scan_path)
    scan_xyz_local = copy.deepcopy(np.asarray(scan_pcd.points))

    scan_intensity = o3d.t.io.read_point_cloud(scan_path).point.intensity.numpy()[:, 0]

    scan_pcd_global = scan_pcd.transform(scan_pose) # global coord, note that this is not deepcopy
    scan_xyz = np.asarray(scan_pcd_global.points)

    scan_intensity = np.expand_dims(scan_intensity, axis=1)
    scan_ranges = LA.norm(scan_xyz_local, axis=1)

    if(is_near_removal):
        eff_idxes = np.where (scan_ranges > thres_near_removal)
        scan_xyz = scan_xyz[eff_idxes[0], :]
        scan_intensity = scan_intensity[eff_idxes[0], :]

    # save
    np_xyz_all[curr_count:curr_count + scan_xyz.shape[0], :] = scan_xyz
    np_intensity_all[curr_count:curr_count + scan_xyz.shape[0], :] = scan_intensity

    curr_count = curr_count + scan_xyz.shape[0]
    scan_ends.append(curr_count)
    print(curr_count)

np_xyz_all = np_xyz_all[0:curr_count, :]
np_intensity_all = np_intensity_all[0:curr_count, :]


# The viewer colors and floor cut need the whole map (height range, floor height), so they are applied after stacking
if(is_o3d_vis or is_live_vis):
    vis_mask = np.ones(curr_count, dtype=bool)
    if(args.no_floor):
        floor_z = estimate_floor_z(np_xyz_all[:, 2])
        vis_mask = np_xyz_all[:, 2] > floor_z + args.floor_margin
        print("floor at z = %.2f m, hiding the points below z = %.2f m" % (floor_z, floor_z + args.floor_margin))

    is_height_color = args.color == "height" or (args.color == "auto" and not np.any(np_intensity_all))
    if(is_height_color):
        z_color_min, z_color_max = np.percentile(np_xyz_all[vis_mask, 2], [1, 99])
        print("coloring by height, z from %.2f to %.2f m" % (z_color_min, z_color_max))
        color_ratio = (np_xyz_all[:, 2] - z_color_min) / max(z_color_max - z_color_min, 1e-6)
    else:
        color_ratio = np_intensity_all[:, 0] / intensity_color_max
    colors_idx = np.round( (color_table_len-1) * np.minimum( 1, np.maximum(0, color_ratio) ) )
    colors_all = color_table[colors_idx.astype(int)]

    pcd_combined_for_vis = o3d.geometry.PointCloud()

if(is_live_vis):
    vis = o3d.visualization.Visualizer()
    vis.create_window('Map', visible = True)

    scan_start = 0
    for scan_end in scan_ends:
        scan_mask = vis_mask[scan_start:scan_end]
        pcd_combined_for_vis.points.extend(o3d.utility.Vector3dVector(np_xyz_all[scan_start:scan_end][scan_mask]))
        pcd_combined_for_vis.colors.extend(o3d.utility.Vector3dVector(colors_all[scan_start:scan_end][scan_mask]))

        if(scan_start == 0): # to ensure the vis init
            vis.add_geometry(pcd_combined_for_vis)

        vis.update_geometry(pcd_combined_for_vis)
        vis.poll_events()
        vis.update_renderer()
        scan_start = scan_end

#
if(is_o3d_vis):
    print("draw the merged map.")
    pcd_combined_for_vis.points = o3d.utility.Vector3dVector(np_xyz_all[vis_mask])
    pcd_combined_for_vis.colors = o3d.utility.Vector3dVector(colors_all[vis_mask])
    o3d.visualization.draw_geometries([pcd_combined_for_vis])


# save ply having intensity
xyzi = o3d.t.geometry.PointCloud()
xyzi.point.positions = o3d.core.Tensor(np_xyz_all.astype(np.float32))
xyzi.point.intensity = o3d.core.Tensor(np_intensity_all.astype(np.float32))

map_name = data_dir + "map_" + str(scan_idx_range_to_stack[0]) + "_to_" + str(scan_idx_range_to_stack[1]) + "_with_intensity.pcd"
o3d.t.io.write_point_cloud(map_name, xyzi, compressed=True) # binary_compressed
print("intensity map is save (path:", map_name, ")")

# save rgb colored points
# map_name = data_dir + "map_" + str(scan_idx_range_to_stack[0]) + "_to_" + str(scan_idx_range_to_stack[1]) + ".pcd"
# o3d.io.write_point_cloud(map_name, pcd_combined_for_vis)
# print("the map is save (path:", map_name, ")")


