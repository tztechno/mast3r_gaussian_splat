import struct
import numpy as np
from pathlib import Path
from PIL import Image
import os
import torch

def rotmat_to_qvec(R):
    """Convert Rotation Matrix to Quaternion (w, x, y, z)"""
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

def extract_all_data(scene, image_paths, conf_threshold=1.5):
    """
    Extract 3D points, colors, confidence, and camera parameters from the scene.
    """
    print("\n=== Extracting Data from Scene ===")
    
    # Basic data from scene
    pts3d_list = scene.get_pts3d()
    im_poses = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    pp = scene.get_principal_points().detach().cpu().numpy()
    im_conf = scene.im_conf

    all_pts, all_cols, all_conf = [], [], []
    cameras_dict = {}

    for i, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        img_raw = Image.open(img_path).convert('RGB')
        W_orig, H_orig = img_raw.size
        
        # Get 3D points and confidence
        pts = pts3d_list[i].detach().cpu().numpy()
        conf = im_conf[i].detach().cpu().numpy()
        H_pts, W_pts = pts.shape[:2]

        # Sample colors at the same resolution as 3D points
        img_res = img_raw.resize((W_pts, H_pts), Image.BILINEAR)
        cols = np.array(img_res)

        # Filter by confidence threshold
        mask = conf > conf_threshold
        all_pts.append(pts[mask])
        all_cols.append(cols[mask])
        all_conf.append(conf[mask])

        # Camera parameter calculation with scale correction
        scale = W_orig / W_pts
        fx = float(focals[i, 0] if focals.ndim > 1 else focals[i]) * scale
        fy = float(focals[i, 1] if (focals.ndim > 1 and focals.shape[1] > 1) else (focals[i, 0] if focals.ndim > 1 else focals[i])) * scale
        cx, cy = pp[i, 0] * scale, pp[i, 1] * scale

        cameras_dict[img_name] = {
            'id': i + 1,
            'w': W_orig, 'h': H_orig,
            'params': (fx, fy, cx, cy),
            'pose_w2c': np.linalg.inv(im_poses[i]) # World-to-Camera
        }
        print(f"  Image {i+1}: {img_name} ({len(pts[mask]):,} points extracted)")

    return (np.concatenate(all_pts), np.concatenate(all_cols), 
            np.concatenate(all_conf), cameras_dict)

def save_colmap_binary(pts3d, colors, conf, cameras_dict, output_dir):
    """
    Save data in COLMAP binary format (.bin).
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # 1. cameras.bin (PINHOLE model)
    with open(path / 'cameras.bin', 'wb') as f:
        f.write(struct.pack('Q', len(cameras_dict)))
        for name, cam in cameras_dict.items():
            f.write(struct.pack('IiQQ', cam['id'], 1, cam['w'], cam['h']))
            f.write(struct.pack('dddd', *cam['params']))

    # 2. images.bin
    with open(path / 'images.bin', 'wb') as f:
        f.write(struct.pack('Q', len(cameras_dict)))
        for name, cam in cameras_dict.items():
            q = rotmat_to_qvec(cam['pose_w2c'][:3, :3])
            t = cam['pose_w2c'][:3, 3]
            f.write(struct.pack('I', cam['id']))
            f.write(struct.pack('dddd', *q))
            f.write(struct.pack('ddd', *t))
            f.write(struct.pack('I', cam['id']))
            f.write(name.encode('utf-8') + b'\x00')
            f.write(struct.pack('Q', 0))

    # 3. points3D.bin (Colored)
    with open(path / 'points3D.bin', 'wb') as f:
        f.write(struct.pack('Q', len(pts3d)))
        for i, (pt, col, cf) in enumerate(zip(pts3d, colors, conf)):
            f.write(struct.pack('Q', i + 1))
            f.write(struct.pack('ddd', *pt))
            f.write(struct.pack('BBB', *col)) # RGB
            f.write(struct.pack('d', 1.0 / max(cf, 0.01))) # Error estimate
            f.write(struct.pack('Q', 0))

    print(f"\n✓ COLMAP binary files exported to: {output_dir}")

def write_colored_ply(pts3d, colors, output_path):
    """Export a colored PLY file for visualization."""
    with open(output_path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts3d)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n")
        for pt, col in zip(pts3d, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(col[0])} {int(col[1])} {int(col[2])}\n")
    print(f"✓ PLY file saved: {output_path}")

def create_colmap_bins(scene, image_paths, output_dir, conf_threshold=1.5):
    """
    Main entry point to generate COLMAP reconstruction data.
    """
    print("="*60)
    print("DUST3R TO COLMAP RECONSTRUCTION")
    print("="*60)

    # 1. Extraction
    pts3d, colors, conf, cameras_dict = extract_all_data(scene, image_paths, conf_threshold)

    # 2. COLMAP Binary Output
    save_colmap_binary(pts3d, colors, conf, cameras_dict, output_dir)

    # 3. PLY Output (Optional verification)
    write_colored_ply(pts3d, colors, Path(output_dir) / "point_cloud.ply")

    print("\n✓ PROCESS COMPLETE")
    return cameras_dict, pts3d, conf, colors


