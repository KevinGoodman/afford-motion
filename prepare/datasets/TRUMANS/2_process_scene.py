import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import trimesh
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_farthest_points, sample_points_from_meshes


def voxel_to_points(voxel_grid, voxel_size=1.0):
    points = np.argwhere(voxel_grid == 1) * voxel_size
    return points

def points_to_ply(points, ply_file):
    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # Save the point cloud as a .ply file
    o3d.io.write_point_cloud(ply_file, point_cloud)
    print(f"Point cloud saved to {ply_file}")

def trimesh_to_pytorch3d(trimesh_mesh):
    # Extract vertices and faces from the trimesh object
    vertices = torch.tensor(trimesh_mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(trimesh_mesh.faces, dtype=torch.int64)
    
    # Create a Meshes object from the vertices and faces
    pytorch3d_mesh = Meshes(verts=[vertices], faces=[faces])
    return pytorch3d_mesh

def main(args):
    # get points from meshes, better visual effect than from voxels
    scene_points_dir = os.path.join(args.data_dir, f'Scene_points_{args.sample}')
    os.makedirs(scene_points_dir, exist_ok=True)

    # scene_points_debug_vis_dir = os.path.join(args.data_dir, f'Scene_points_{args.sample}_debug_vis')
    # os.makedirs(scene_points_debug_vis_dir, exist_ok=True)

    scene_dir = os.path.join(args.data_dir, 'Scene_mesh')
    scene_lists = sorted(os.listdir(scene_dir))
    for i, scene in tqdm(enumerate(scene_lists), total=len(scene_lists)):
        scene_path = os.path.join(scene_dir, scene)
        mesh = trimesh.load(scene_path, process=False)
        verts = mesh.vertices # (N, 3)
        color = mesh.visual.vertex_colors[:, 0:3] / 127.5 - 1.0 # (N, 3), actually no color info in the mesh file, just constant value here

        if args.sample == 'random':
            # random downsample args.point_num points
            down_sampled_verts = verts[np.random.choice(len(verts), args.point_num, replace=False)] # (8192, 3)
        elif args.sample == 'mesh_fps':
            # downsample using mesh FPS
            meshes = trimesh_to_pytorch3d(mesh)
            down_sampled_verts = sample_points_from_meshes(meshes=meshes, num_samples=args.point_num)[0].cpu().numpy() # (8192, 3)
        elif args.sample == 'vert_fps':
            # use vertices farthest point sampling (FPS) to sample 8192 points, better visual effect than random sampling, and mesh fps sampling
            verts = verts[None, :, :]   # [1, N, 3]
            down_sampled_verts = sample_farthest_points(points=torch.from_numpy(verts).cuda(), K=args.point_num)[0].squeeze(0).cpu().numpy() # (8192, 3)
        elif args.sample == 'none':
            down_sampled_verts = verts
        else:
            raise ValueError(f"Invalid sampling method: {args.sample}")
        
        out_filename = os.path.join(scene_points_dir, scene.replace('.obj', '.npy'))
        
        # points_to_ply(down_sampled_verts, os.path.join(scene_points_debug_vis_dir, scene.replace('.obj', '.ply')))
        
        # pass
        np.save(out_filename, np.concatenate([down_sampled_verts, color[:verts.shape[0]]], axis=1).astype(np.float32))

    #     if i == 5:
    #         break

    # get points from voxels
    # """ Load necessary base data """
    # # Load necessary data
    # scene_points_dir = os.path.join(args.data_dir, 'Scene_points')
    # os.makedirs(scene_points_dir, exist_ok=True)

    # voxels_dir = os.path.join(args.data_dir, 'voxels')
    # seg_names = np.load(os.path.join(args.data_dir, 'seg_name.npy'))

    # # Process seg_names to get unique segments and their indices
    # unique_segs, seg_indices = np.unique(seg_names, return_index=True)
    # sort_order = np.argsort(seg_indices)
    # unique_segs = unique_segs[sort_order]
    # seg_indices = np.sort(seg_indices)

    # valid_segs = []
    # for seg, idx in tqdm(zip(unique_segs, seg_indices), desc='Checking valid segments'):
    #     if os.path.exists(os.path.join(args.data_dir, 'Actions', f"{seg}.txt")):
    #         valid_segs.append(seg)

    # valid_segs = np.array(valid_segs)

    # scene_voxels = {}
    # scene_points = {}

    # for seg_name in tqdm(valid_segs, total=len(valid_segs), desc='Loading scene voxels'):
    #     # 加载场景点云
    #     scene_path = os.path.join(voxels_dir, f"{seg_name}.npy")
    #     scene_voxels[seg_name] = np.load(scene_path, mmap_mode='r')

    #     points = voxel_to_points(scene_voxels[seg_name])
    #     if args.sample == 'none':
    #         partial_scene = points
    #     elif args.sample == 'random':
    #         # random downsample args.point_num points
    #         partial_scene = points[np.random.choice(len(points), args.point_num, replace=False)]
    #     elif 'fps' in args.sample:
    #         # downsample using FPS
    #         partial_scene = sample_farthest_points(points=torch.from_numpy(points[None, :, :]).cuda(), K=args.point_num)[0].squeeze(0).cpu().numpy()

    #     scene_points[seg_name] = partial_scene
    #     # points_to_ply(partial_scene, os.path.join(scene_points_dir, f"{seg_name}.ply"))
        
    #     np.save(os.path.join(scene_points_dir, f"{seg_name}.npy"), points)

    #     pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/TRUMANS')
    parser.add_argument('--point_num', type=int, default=8192)
    parser.add_argument('--sample', type=str, default='none', choices=['random', 'mesh_fps', 'vert_fps', 'none'])
    # parser.add_argument('--sample', type=str, default='random', choices=['random', 'mesh_fps', 'vert_fps', 'none'])
    # parser.add_argument('--sample', type=str, default='vert_fps', choices=['random', 'mesh_fps', 'vert_fps'])
    args = parser.parse_args()

    main(args)
