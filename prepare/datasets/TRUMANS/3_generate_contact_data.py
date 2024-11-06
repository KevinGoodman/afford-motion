import os, sys
sys.path.append(os.path.abspath('.'))
import cv2
import csv
import json
import random
import trimesh
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
from trimesh import transform_points
from smplkit.constants import SKELETON_CHAIN
from utils.visualize import skeleton_to_mesh
from sklearn.neighbors import NearestNeighbors
kinematic_chain = SKELETON_CHAIN.SMPLH['kinematic_chain'] # remove the hands, jaw, and eyes


def load_trumans(min_horizon: int, max_horizon: int, **kwargs: Dict) -> Tuple:
    """ Load necessary base data """
    data_dir = 'dataset/trumans'

    # Load necessary data
    seg_names = np.load(os.path.join(data_dir, 'seg_name.npy'))
    scene_flags = np.load(os.path.join(data_dir, 'scene_flag.npy'))
    scene_list = np.load(os.path.join(data_dir, 'scene_list.npy')).tolist()
    
    # Process seg_names to get unique segments and their indices
    unique_segs, seg_indices = np.unique(seg_names, return_index=True)
    sort_order = np.argsort(seg_indices)
    unique_segs = unique_segs[sort_order]
    seg_indices = np.sort(seg_indices)

    valid_segs = []
    valid_seg_indices = []
    for seg, idx in zip(unique_segs, seg_indices):
        if os.path.exists(os.path.join(data_dir, 'Actions', f"{seg}.txt")):
            valid_segs.append(seg)
            valid_seg_indices.append(idx)

    valid_segs = np.array(valid_segs)
    valid_seg_indices = np.array(valid_seg_indices)
    
    motions = []
    scene_data = {}

    for idx, seg_name in enumerate(valid_segs):
        text_file = os.path.join(data_dir, 'Actions', f"{seg_name}.txt")
        with open(text_file, 'r') as f:
            lines = f.readlines()
        
        start_idx = valid_seg_indices[idx]
        
        motion_index = 0
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                start, end, motion_text = int(parts[0]), int(parts[1]), parts[2]
                
                length = end - start + 1
                if length < min_horizon or length > max_horizon:
                    continue

                if start < end:
                    scene_id = scene_flags[start_idx + start]
                    scene_name = scene_list[scene_id]
                    with open(os.path.join(data_dir, 'Scene_text', f'{scene_name}.txt'), 'r') as f:
                        scene_text = f.readline()

                    pose_seq = np.load(os.path.join(data_dir, 'motions_pos', f"{seg_name}_{motion_index}.npy"))
                    motions.append((pose_seq, scene_id, seg_name, motion_index, motion_text))

                    scene_data[scene_id] = {
                        'scene_name': scene_name,
                        'scene_text': scene_text,
                        'pcd': np.load(os.path.join(data_dir, 'Scene_points_none', f'{scene_name}.npy')),
                        'mesh_path': os.path.join(data_dir, 'Scene_mesh', f'{scene_name}.obj'),
                    }
                    motion_index += 1

    return motions, scene_data

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Reference: https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default l2
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        return min_y_to_x
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        return min_x_to_y
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]

        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        return min_y_to_x, min_x_to_y
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

def visualize_contact_map(xyz, rgb, contact, joint_id, threshold=0.8, gray=True) -> None:
    """ Visualize contact map on scene point cloud
    
    Args:
        xyz: scene point cloud xyz
        rgb: scene point cloud rgb
        contact: contact map
        joint_id: joint id
        threshold: threshold for contact map
    """
    contact = np.exp(-0.5 * contact ** 2 / 0.5 ** 2)
    color = ((rgb.copy() + 1.0) * 127.5).astype(np.uint8)
    if gray:
        color = color.reshape(-1, 1, 3)
        color = cv2.cvtColor(np.uint8(color), cv2.COLOR_RGB2GRAY).repeat(3, axis=-1)
    
    contact = contact[:, joint_id:joint_id+1]
    overlay_mask = (contact > threshold).reshape(-1)
    contact = contact[overlay_mask]
    if len(contact) != 0:
        contact_map = (contact - contact.min()) / (contact.max() - contact.min())
        contact_map = np.uint8(255 * contact_map)
        heatmap = cv2.applyColorMap(contact_map, cv2.COLORMAP_PARULA)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR).reshape(-1, 3)
        color[overlay_mask, :] = heatmap
    
    S = trimesh.Scene()
    S.add_geometry(trimesh.creation.axis())
    S.add_geometry(trimesh.PointCloud(vertices=xyz, colors=color))
    S.show()

    assert False

def visualize_partial_scene(partial_scene, pose_seq, scene_trans, scene_path):
    """ Visualize partial scene and skeleton
    
    Args:
        partial_scene: partial scene point cloud
        pose_seq: pose sequence
        scene_trans: transformation matrix for scene mesh
        scene_path: scene mesh path
    """

    S = trimesh.Scene()
    S.add_geometry(trimesh.creation.axis())
    xyz = partial_scene[:, 0:3]
    color = ((partial_scene[:, 3:6] + 1) * 127.5).astype(np.uint8)
    S.add_geometry(trimesh.PointCloud(vertices=xyz, vertex_colors=color))
    skeleton = pose_seq[:, :22 * 3].reshape(-1, 22, 3)
    mesh = skeleton_to_mesh(skeleton, kinematic_chain)
    S.add_geometry(mesh)
    S.show()

    S = trimesh.Scene()
    S.add_geometry(trimesh.creation.axis())
    if scene_path is not None:
        scene_mesh = trimesh.load(scene_path, process=False)
        scene_mesh.apply_translation(scene_trans)
        S.add_geometry(scene_mesh)
    S.add_geometry(mesh)
    S.show()

def process(motions, scene_data, save_dir, num_points: int=8192, region_size: float=4.0, **kwargs) -> None:
    """ Process motion-condition pairs
    
    Args:
        motions: motion data
        scene_data: scene data
        num_points: number of points for each scene point cloud chunk
    """
    JOINTS = 22
    REGION_SIZE = region_size
    TRAJ_PAD = REGION_SIZE * kwargs.get('traj_pad_ratio', 0.5)

    train_set = json.load(open('dataset/trumans/train_set.json'))
    test_set = json.load(open('dataset/trumans/test_set.json'))

    anno_list = []
    train_idx, test_idx = 0, 0

    for i in tqdm(range(len(motions))):
        pose_seq, scene_id, seg_name, motion_index, motion_text = motions[i]
        
        if f'{seg_name}_{motion_index}' in train_set:
            split = 'train'
        elif f'{seg_name}_{motion_index}' in test_set:
            split = 'test'
        else:
            continue    # skip val
    
        ## pose sequence
        pose_seq = pose_seq.copy().astype(np.float32)   # [L, 22, 3]
        pelvis_seq = pose_seq[:, 0, :3]    # [L, 3]
        
        ## scene
        traj_max = pelvis_seq.max(axis=0)[0:2]
        traj_min = pelvis_seq.min(axis=0)[0:2]
        traj_size = traj_max - traj_min
        traj_size = traj_size + TRAJ_PAD * np.exp(-traj_size)

        pad = (REGION_SIZE - traj_size) / 2
        pad = np.maximum(pad, [0, 0])

        center = (traj_max + traj_min) / 2
        center_region_max = center + pad
        center_region_min = center - pad
        sample_xy = np.random.uniform(low=center_region_min, high=center_region_max)
        sample_region_max = sample_xy + REGION_SIZE / 2
        sample_region_min = sample_xy - REGION_SIZE / 2

        scene_pcd = scene_data[scene_id]['pcd'].copy()
        point_in_region = (scene_pcd[:, 0] >= sample_region_min[0]) & (scene_pcd[:, 0] <= sample_region_max[0]) & \
                            (scene_pcd[:, 1] >= sample_region_min[1]) & (scene_pcd[:, 1] <= sample_region_max[1])
        
        indices = np.arange(len(scene_pcd))[point_in_region]
        assert len(indices) > 0, "No points in the region!"
        if len(indices) < num_points:
            if len(indices) < num_points // 4:
                print(f"Warning: only {len(indices)} points in the region! Less than {num_points // 4} points!")
            while len(indices) < num_points:
                indices = np.concatenate([indices, indices])    

        indices = np.random.choice(indices, num_points, replace=False)

        ## save the partial scene without transformation
        points = scene_data[scene_id]['pcd'].copy()
        points = points[indices]

        ## transform the partial scene and motion to center
        xyz = points[:, 0:3]
        xy_center = (xyz[:, 0:2].max(axis=0) + xyz[:, 0:2].min(axis=0)) * 0.5
        z_height = np.percentile(xyz[:, 2], 2) # 2% height
        trans_vec = np.array([-xy_center[0], -xy_center[1], -z_height])
        points[:, 0:3] += trans_vec

        pose_seq += trans_vec
        scene_trans = trans_vec

        ## use the partial scene for computing distance map
        partial_scene = points.copy()
        ## visualize partial scene
        ## for debug
        # print(partial_scene.shape, points.shape, points.dtype, indices.shape, indices.dtype)
        # visualize_partial_scene(partial_scene, pose_seq, scene_trans, scene_data[scene_id]['mesh_path'] if scene_id is not None else None)

        ## dist map
        dist = []
        for j in range(JOINTS):
            joint = pose_seq[:, j, :]
            scene_xyz = partial_scene[:, 0:3]
            c_d = chamfer_distance(joint, scene_xyz, metric='l2', direction='y_to_x')
            dist.append(c_d)
        dist = np.concatenate(dist, axis=-1).astype(np.float32)
        
        ## visualize contact map
        ## for debug
        # visualize_contact_map(partial_scene[:, 0:3], partial_scene[:, 3:6], dist, 10)
        # visualize_contact_map(partial_scene[:, 0:3], partial_scene[:, 3:6], dist, 11)
        # visualize_contact_map(partial_scene[:, 0:3], partial_scene[:, 3:6], dist, 20)
        # visualize_contact_map(partial_scene[:, 0:3], partial_scene[:, 3:6], dist, 21)
        
        ## re-index and save
        idx = f'{train_idx:0>5d}' if split == "train" else f'{test_idx:0>5d}'
        save_motion_path = os.path.join(save_dir, 'motions', split, f'{idx}.npy')
        os.makedirs(os.path.dirname(save_motion_path), exist_ok=True)
        with open(save_motion_path, 'wb') as fp:
            np.save(fp, pose_seq)

        save_scene_path = os.path.join(save_dir, 'contacts', split, f'{idx}.npz')
        os.makedirs(os.path.dirname(save_scene_path), exist_ok=True) 

        with open(save_scene_path, 'wb') as fp:
            np.savez(fp, points=points, mask=indices, dist=dist)

        ## save annotation
        anno_list.append([
            scene_id,
            f"{scene_trans[0]:.8f}",
            f"{scene_trans[1]:.8f}",
            f"{scene_trans[2]:.8f}",
            motion_text,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            scene_data[scene_id],
        ])

        if split == "train":    
            train_idx += 1
        else:
            test_idx += 1
        
    with open(os.path.join(save_dir, 'anno.csv'), 'w') as fp:
        csvwriter = csv.writer(fp)
        csvwriter.writerow(['scene_id', 'scene_trans_x', 'scene_trans_y', 'scene_trans_z', 'utterance', 'others'])
        csvwriter.writerows(anno_list)

def visualize(motions, scene_data):
    for i in range(10):
        index = random.randint(0, len(motions))
        pose_seq, texts, (scene_id, scene_trans), other_info = motions[index]
        scene_pcd = scene_data[scene_id]['pcd']
        scene_mesh_path = scene_data[scene_id]['mesh_path']
        scene_mesh = trimesh.load(scene_mesh_path, process=False)
        scene_mesh.apply_transform(scene_trans)
        assert len(scene_mesh.vertices) == len(scene_pcd) 

        ## visualize
        S = trimesh.Scene()
        S.add_geometry(trimesh.creation.axis())
        xyz = transform_points(scene_pcd[:, :3], scene_trans)
        color = ((scene_pcd[:, 3:6] + 1) * 127.5).astype(np.uint8)
        S.add_geometry(trimesh.PointCloud(vertices=xyz, vertex_colors=color))

        skeleton = pose_seq[:, :22 * 3].reshape(-1, 22, 3)
        mesh = skeleton_to_mesh(skeleton, kinematic_chain)
        S.add_geometry(mesh)
        S.show()

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--min_horizon', type=int, default=24)
    parser.add_argument('--max_horizon', type=int, default=120)
    parser.add_argument('--num_points', type=int, default=8192)
    parser.add_argument('--segment_horizon', type=int, default=120)
    parser.add_argument('--segment_stride', type=int, default=4)
    parser.add_argument('--random_segment', action='store_true', default=False)
    parser.add_argument('--random_segment_window', type=int, default=60)
    parser.add_argument('--region_size', type=float, default=4.0)
    parser.add_argument('--traj_pad_ratio', type=float, default=0.5)
    args = parser.parse_args()

    save_dir = f'./dataset/trumans/contact_motion/'
    motions, scene_data = eval('load_trumans')(**vars(args))
    # visualize(motions, scene_data)
    process(motions, scene_data, save_dir, **vars(args))
