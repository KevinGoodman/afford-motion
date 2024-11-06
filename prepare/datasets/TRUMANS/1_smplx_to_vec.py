import os
import numpy as np
from tqdm import tqdm
DATA_DIR = './data/TRUMANS'
JOINT_NUM = 22


""" Load necessary base data """
# Load necessary data
human_joints = np.load(os.path.join(DATA_DIR, 'human_joints.npy'))
seg_names = np.load(os.path.join(DATA_DIR, 'seg_name.npy'))
scene_flags = np.load(os.path.join(DATA_DIR, 'scene_flag.npy'))
scene_list = np.load(os.path.join(DATA_DIR, 'scene_list.npy'))

# Process seg_names to get unique segments and their indices
unique_segs, seg_indices = np.unique(seg_names, return_index=True)
sort_order = np.argsort(seg_indices)
unique_segs = unique_segs[sort_order]
seg_indices = np.sort(seg_indices)

valid_segs = []
valid_seg_indices = []
for seg, idx in tqdm(zip(unique_segs, seg_indices), desc='Checking valid segments'):
    if os.path.exists(os.path.join(DATA_DIR, 'Actions', f"{seg}.txt")):
        valid_segs.append(seg)
        valid_seg_indices.append(idx)

valid_segs = np.array(valid_segs)
valid_seg_indices = np.array(valid_seg_indices)

"""Prepare text-motion pairs"""
data_pairs = []
for idx, seg_name in tqdm(enumerate(valid_segs), desc='Preparing text-motion pairs'):
    text_file = os.path.join(DATA_DIR, 'Actions', f"{seg_name}.txt")
    with open(text_file, 'r') as f:
        lines = f.readlines()
    
    start_idx = valid_seg_indices[idx]
    end_idx = valid_seg_indices[idx + 1] if idx < len(valid_segs) - 1 else len(seg_names)
    
    motion_index = 0
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            start, end = int(parts[0]), int(parts[1])
            if start < end:
                scene_id = scene_flags[start_idx + start]
                scene_name = scene_list[scene_id]
                with open(os.path.join(DATA_DIR, 'Scene_text', f'{scene_name}.txt'), 'r') as f:
                    scene_text = f.readline()
                data_pairs.append((seg_name, parts[2], start_idx, start, end, motion_index, scene_text))
                motion_index += 1


motion_pos_dir = os.path.join(DATA_DIR, 'motions_pos')
os.makedirs(motion_pos_dir, exist_ok=True)

for i, dp in tqdm(enumerate(data_pairs), desc='Saving data pairs', total=len(data_pairs)):
    seg_name, action, start_idx, start, end, motion_index, scene_text = dp
    
    motion_data = human_joints[start_idx + start: start_idx + end][:, :JOINT_NUM]  # [length, 22, 3]
    name = f'{seg_name}_{motion_index}'
    np.save(os.path.join(motion_pos_dir, f"{name}.npy"), motion_data)

    # if i == 5:
    #     break

    
