#process2_02.py

import os
import struct
import numpy as np
import torch
from PIL import Image
from pathlib import Path

def rotmat_to_qvec(R):
    """Convert rotation matrix to COLMAP-style quaternion (w, x, y, z)."""
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
    qvec = np.array([w, x, y, z], dtype=np.float64)
    return qvec / np.linalg.norm(qvec)

def extract_process2_data(scene, image_paths, conf_threshold=1.5):
    """Extracts camera params, filtered 3D points, and matching colors."""
    print("\n=== Stage 1: Extracting Camera Parameters & 3D Points ===")
    mast3r_size = 224.0
    cameras_dict, all_pts3d, all_conf, all_colors = {}, [], [], []

    # Get data from scene object
    poses = scene.get_im_poses() if hasattr(scene, 'get_im_poses') else getattr(scene, 'im_poses', None)
    focals = scene.get_focals() if hasattr(scene, 'get_focals') else getattr(scene, 'im_focals', None)
    pps = scene.get_principal_points() if hasattr(scene, 'get_principal_points') else getattr(scene, 'im_pp', None)

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        with Image.open(img_path) as img:
            W, H = img.size
            # Prepare colors (resized to match MASt3R internal grid)
            img_resized = img.resize((224, 224), Image.BILINEAR)
            all_colors.append(np.array(img_resized).reshape(-1, 3))
        
        scale = W / mast3r_size

        # Camera Intrinsics (Handling fx, fy)
        if focals is not None and idx < len(focals):
            f_val = focals[idx].detach().cpu().numpy()
            fx, fy = (f_val[0] * scale, f_val[1] * scale) if f_val.size > 1 else (f_val.item() * scale, f_val.item() * scale)
        else:
            fx = fy = 1000.0

        pp = (pps[idx].detach().cpu().numpy() if pps is not None else np.array([112, 112])) * scale
        
        # Camera Extrinsics (C2W to W2C)
        if poses is not None and idx < len(poses):
            c2w = poses[idx].detach().cpu().numpy() if isinstance(poses[idx], torch.Tensor) else poses[idx]
            w2c = np.linalg.inv(c2w)
        else:
            w2c = np.eye(4)

        cameras_dict[img_name] = {
            'focal': (fx, fy), 'pp': pp, 'rotation': w2c[:3, :3], 
            'translation': w2c[:3, 3], 'width': W, 'height': H
        }

        # Collect Points and Confidence
        pts_img = scene.im_pts3d[idx].detach().cpu().numpy().reshape(-1, 3)
        conf_img = scene.im_conf[idx].detach().cpu().numpy().reshape(-1)
        all_pts3d.append(pts_img)
        all_conf.append(conf_img)

    # Flatten and Filter
    pts3d_raw = np.vstack(all_pts3d)
    conf_raw = np.concatenate(all_conf)
    colors_raw = np.vstack(all_colors)

    valid_mask = conf_raw > conf_threshold
    pts3d = pts3d_raw[valid_mask]
    confidence = conf_raw[valid_mask]
    colors = colors_raw[valid_mask]

    print(f"✓ Extracted {len(cameras_dict)} cameras")
    print(f"✓ Points after filtering (>{conf_threshold}): {len(pts3d):,}")
    return cameras_dict, pts3d, confidence, colors

def write_colmap_binaries(cameras_dict, pts3d, confidence, colors, output_dir):
    """Writes cameras.bin, images.bin, and points3D.bin (with colors)."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # 1. cameras.bin
    with open(path / 'cameras.bin', 'wb') as f:
        f.write(struct.pack('Q', len(cameras_dict)))
        for i, (name, c) in enumerate(cameras_dict.items(), 1):
            f.write(struct.pack('IiQQdddd', i, 1, c['width'], c['height'], c['focal'][0], c['focal'][1], c['pp'][0], c['pp'][1]))

    # 2. images.bin
    with open(path / 'images.bin', 'wb') as f:
        f.write(struct.pack('Q', len(cameras_dict)))
        for i, (name, c) in enumerate(cameras_dict.items(), 1):
            q, t = rotmat_to_qvec(c['rotation']), c['translation']
            f.write(struct.pack('I7dI', i, *q, *t, i))
            f.write(name.encode('utf-8') + b'\x00')
            f.write(struct.pack('Q', 0))

    # 3. points3D.bin (The Color Update)
    with open(path / 'points3D.bin', 'wb') as f:
        f.write(struct.pack('Q', len(pts3d)))
        for i, (pt, conf, rgb) in enumerate(zip(pts3d, confidence, colors), 1):
            f.write(struct.pack('QdddBBB', i, *pt, *np.clip(rgb, 0, 255).astype(int)))
            f.write(struct.pack('dQ', 1.0 / max(conf, 0.001), 0))

    print(f"✓ Binary files exported to: {output_dir}")

def create_full_colmap_export(scene, image_paths, output_dir, conf_threshold=1.5):
    """One-call function to run the entire export pipeline."""
    print("="*60)
    print("STARTING COLMAP EXPORT WITH COLORS")
    print("="*60)
    
    cameras, pts3d, conf, colors = extract_process2_data(scene, image_paths, conf_threshold)
    write_colmap_binaries(cameras, pts3d, conf, colors, output_dir)
    
    print("\n✓ Process Complete!")
    return cameras, pts3d, conf, colors

# Usage:
# cameras, pts, conf, colors = create_full_colmap_export(scene, image_paths, 'output/sparse/0')
