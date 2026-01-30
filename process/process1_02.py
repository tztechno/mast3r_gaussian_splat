#process1_02.py

import os
import struct
import numpy as np
import torch
from PIL import Image
from pathlib import Path

def rotmat2qvec(R):
    """Convert rotation matrix to COLMAP quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

def export_mast3r_to_colmap(scene, image_paths, output_dir, max_points=1_000_000):
    """Extracts data and saves COLMAP binaries in one workflow."""
    print("\n" + "="*50 + "\nSTARTING COLMAP EXPORT\n" + "="*50)
    
    # --- 1. Point Cloud & Color Extraction ---
    pts_all = scene.get_pts3d()
    if isinstance(pts_all, list):
        pts_all = torch.stack([p if isinstance(p, torch.Tensor) else torch.tensor(p) for p in pts_all])
    
    if len(pts_all.shape) == 4: # Batched [B, H, W, 3]
        B, H, W, _ = pts_all.shape
        pts3d = pts_all.reshape(-1, 3).detach().cpu().numpy()
        colors = []
        for path in image_paths:
            img = Image.open(path).resize((W, H))
            colors.append(np.array(img))
        colors = np.stack(colors).reshape(-1, 3) / 255.0
    else:
        pts3d = pts_all.detach().cpu().numpy() if isinstance(pts_all, torch.Tensor) else pts_all
        colors = np.ones((len(pts3d), 3)) * 0.5

    # Filter invalid and Downsample
    mask = ~(np.isnan(pts3d).any(axis=1) | np.isinf(pts3d).any(axis=1))
    pts3d, colors = pts3d[mask], colors[mask]
    
    if len(pts3d) > max_points:
        print(f"Downsampling from {len(pts3d):,} to {max_points:,} points...")
        idx = np.random.choice(len(pts3d), max_points, replace=False)
        pts3d, colors = pts3d[idx], colors[idx]

    # --- 2. Camera & Pose Extraction ---
    poses_c2w = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    pps = scene.get_principal_points().detach().cpu().numpy()
    mast3r_size = 224.0
    
    cameras_data, w2c_poses = [], []
    for i, path in enumerate(image_paths):
        with Image.open(path) as img:
            orig_W, orig_H = img.size
        
        scale = orig_W / mast3r_size
        fx = focals[i, 0] * scale
        fy = (focals[i, 1] if focals.shape[1] > 1 else focals[i, 0]) * scale
        cx, cy = pps[i, 0] * scale, pps[i, 1] * scale
        
        cameras_data.append({'id': i+1, 'W': orig_W, 'H': orig_H, 'params': [fx, fy, cx, cy]})
        w2c_poses.append(np.linalg.inv(poses_c2w[i]))

    # --- 3. Binary Writing ---
    save_path = Path(output_dir) / 'sparse' / '0'
    save_path.mkdir(parents=True, exist_ok=True)

    # cameras.bin
    with open(save_path / 'cameras.bin', 'wb') as f:
        f.write(struct.pack('Q', len(cameras_data)))
        for cam in cameras_data:
            f.write(struct.pack('iiQQdddd', cam['id'], 1, cam['W'], cam['H'], *cam['params']))

    # images.bin
    with open(save_path / 'images.bin', 'wb') as f:
        f.write(struct.pack('Q', len(image_paths)))
        for i, (path, pose) in enumerate(zip(image_paths, w2c_poses)):
            qvec = rotmat2qvec(pose[:3, :3])
            f.write(struct.pack('i', i+1))
            for val in qvec: f.write(struct.pack('d', val))
            for val in pose[:3, 3]: f.write(struct.pack('d', val))
            f.write(struct.pack('i', i+1))
            f.write(os.path.basename(path).encode('utf-8') + b'\x00')
            f.write(struct.pack('Q', 0))

    # points3D.bin
    with open(save_path / 'points3D.bin', 'wb') as f:
        f.write(struct.pack('Q', len(pts3d)))
        for i, (pt, col) in enumerate(zip(pts3d, colors)):
            f.write(struct.pack('QdddBBB dQ', i, *pt, * (col*255).astype(np.uint8), 0.0, 0))
            if (i + 1) % 500_000 == 0: print(f"  Wrote {i+1} / {len(pts3d)} points...")

    print(f"\n✓ Exported to {save_path}")
    print(f"  Cameras: {len(cameras_data)} | Points: {len(pts3d):,}")
    return save_path

# Execute:
# export_mast3r_to_colmap(scene, image_paths, './output')
