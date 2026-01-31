# process2_08_optimized.py
# COLMAP binary generation (cameras.bin, images.bin, points3D.bin) - Optimized version

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
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    qvec = np.array([w, x, y, z], dtype=np.float64)
    return qvec / np.linalg.norm(qvec)


def write_cameras_binary(cameras_dict, image_size, output_file):
    """Output cameras.bin (PINHOLE model: fx, fy, cx, cy)"""
    width, height = image_size
    num_cameras = len(cameras_dict)
    PINHOLE = 1

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', num_cameras))

        for camera_id, (img_id, cam_params) in enumerate(cameras_dict.items(), start=1):
            focal = cam_params['focal']
            fx, fy = focal if isinstance(focal, (tuple, list)) else (focal, focal)

            if 'pp' in cam_params:
                cx, cy = float(cam_params['pp'][0]), float(cam_params['pp'][1])
            else:
                cx, cy = width / 2.0, height / 2.0

            f.write(struct.pack('I', camera_id))
            f.write(struct.pack('i', PINHOLE))
            f.write(struct.pack('Q', width))
            f.write(struct.pack('Q', height))
            f.write(struct.pack('dddd', fx, fy, cx, cy))

    print(f"✓ cameras.bin: {num_cameras} cameras")


def write_images_binary(cameras_dict, output_file):
    """Output images.bin"""
    num_images = len(cameras_dict)

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', num_images))

        for image_id, (img_id, cam_params) in enumerate(cameras_dict.items(), start=1):
            quat = rotmat_to_qvec(cam_params['rotation'])
            t = cam_params['translation']

            f.write(struct.pack('I', image_id))
            f.write(struct.pack('dddd', *quat))
            f.write(struct.pack('ddd', *t))
            f.write(struct.pack('I', image_id))
            f.write(img_id.encode('utf-8') + b'\x00')
            f.write(struct.pack('Q', 0))

    print(f"✓ images.bin: {num_images} images")


def write_points3D_binary(pts3d, confidence, colors, output_file):
    """Output points3D.bin (with colors)"""
    num_points = len(pts3d)

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', num_points))

        for point_id, (pt, color) in enumerate(zip(pts3d, colors), start=1):
            error = 1.0 / max(confidence[point_id-1], 0.001) if confidence is not None else 1.0
            r, g, b = [int(np.clip(c, 0, 255)) for c in color]

            f.write(struct.pack('Q', point_id))
            f.write(struct.pack('ddd', *pt))
            f.write(struct.pack('BBB', r, g, b))
            f.write(struct.pack('d', error))
            f.write(struct.pack('Q', 0))

    print(f"✓ points3D.bin: {num_points} points")


def extract_camera_params(scene, image_paths, conf_threshold=1.5):
    """Extract camera parameters and 3D points"""
    print("\n=== Extracting Camera Parameters ===")

    cameras_dict = {}
    all_pts3d = []
    all_confidence = []

    poses = getattr(scene, 'im_poses', None) or (scene.get_im_poses() if hasattr(scene, 'get_im_poses') else None)
    focals = getattr(scene, 'im_focals', None) or (scene.get_focals() if hasattr(scene, 'get_focals') else None)
    pps = getattr(scene, 'im_pp', None) or (scene.get_principal_points() if hasattr(scene, 'get_principal_points') else None)

    mast3r_size = 224.0
    n_images = min(len(poses) if poses is not None else len(image_paths), len(image_paths))

    for idx in range(n_images):
        img_name = os.path.basename(image_paths[idx])

        try:
            img = Image.open(image_paths[idx])
            W, H = img.size
            img.close()
            scale = W / mast3r_size

            # Pose
            if poses is not None and idx < len(poses):
                pose_c2w = poses[idx]
                if isinstance(pose_c2w, torch.Tensor):
                    pose_c2w = pose_c2w.detach().cpu().numpy()
                pose = np.linalg.inv(pose_c2w) if pose_c2w.shape == (4, 4) else np.eye(4)
            else:
                pose = np.eye(4)

            # Focal Length
            if focals is not None and idx < len(focals):
                focal_mast3r = focals[idx]
                if isinstance(focal_mast3r, torch.Tensor):
                    focal_mast3r = focal_mast3r.detach().cpu()
                
                if focals.shape[1] == 1:
                    fx = fy = float(focal_mast3r[0] if focal_mast3r.numel() > 1 else focal_mast3r) * scale
                else:
                    fx, fy = float(focal_mast3r[0]) * scale, float(focal_mast3r[1]) * scale
            else:
                fx = fy = 1000.0

            # Principal Point
            if pps is not None and idx < len(pps):
                pp_mast3r = pps[idx]
                if isinstance(pp_mast3r, torch.Tensor):
                    pp_mast3r = pp_mast3r.detach().cpu().numpy()
                pp = pp_mast3r * scale
            else:
                pp = np.array([W / 2.0, H / 2.0])

            cameras_dict[img_name] = {
                'focal': (fx, fy),
                'pp': pp,
                'pose': pose,
                'rotation': pose[:3, :3],
                'translation': pose[:3, 3],
                'width': W,
                'height': H
            }

            # 3D Points
            pts3d_img = scene.im_pts3d[idx] if hasattr(scene, 'im_pts3d') and idx < len(scene.im_pts3d) else None
            conf_img = scene.im_conf[idx] if hasattr(scene, 'im_conf') and idx < len(scene.im_conf) else None

            if pts3d_img is not None:
                if isinstance(pts3d_img, torch.Tensor):
                    pts3d_img = pts3d_img.detach().cpu().numpy()

                pts3d_flat = pts3d_img.reshape(-1, 3)
                all_pts3d.append(pts3d_flat)

                if conf_img is not None:
                    if isinstance(conf_img, torch.Tensor):
                        conf_img = conf_img.detach().cpu().numpy()
                    conf_flat = conf_img.reshape(-1) if conf_img.ndim > 1 else conf_img
                    all_confidence.append(conf_flat if len(conf_flat) == len(pts3d_flat) else np.ones(len(pts3d_flat)))
                else:
                    all_confidence.append(np.ones(len(pts3d_flat)))

        except Exception as e:
            print(f"⚠️ Error processing {img_name}: {e}")
            continue

    # Merge
    pts3d_raw = np.vstack(all_pts3d) if all_pts3d else np.zeros((0, 3))
    conf_raw = np.concatenate(all_confidence) if all_confidence else np.zeros(0)

    print(f"✓ {len(cameras_dict)} cameras, {len(pts3d_raw):,} raw points")

    # Filtering
    if len(conf_raw) > 0:
        valid_mask = conf_raw > conf_threshold
        pts3d = pts3d_raw[valid_mask]
        confidence = conf_raw[valid_mask]
        print(f"✓ {len(pts3d):,} points after filtering (>{conf_threshold})")
    else:
        pts3d, confidence = pts3d_raw, conf_raw

    return cameras_dict, pts3d, confidence, pts3d_raw, conf_raw


def extract_colors_from_images(image_paths, pts3d, conf_raw, conf_threshold=1.5):
    """Extract colors from images"""
    print("\n=== Extracting Colors ===")

    mast3r_size = 224
    all_colors = []

    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        img_resized = img.resize((mast3r_size, mast3r_size), Image.BILINEAR)
        colors_flat = np.array(img_resized).reshape(-1, 3)
        img.close()

        pts_per_image = len(conf_raw) // len(image_paths)
        all_colors.append(colors_flat[:pts_per_image])

    colors_all = np.vstack(all_colors)
    valid_mask = conf_raw > conf_threshold
    colors_filtered = colors_all[valid_mask]

    if len(colors_filtered) > len(pts3d):
        colors_filtered = colors_filtered[:len(pts3d)]

    print(f"✓ {len(colors_filtered):,} colors extracted")
    return colors_filtered


def export_colmap_binary(cameras_dict, pts3d, confidence, colors, image_size, output_dir):
    """Output three types of COLMAP binary files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    write_cameras_binary(cameras_dict, image_size, output_path / 'cameras.bin')
    write_images_binary(cameras_dict, output_path / 'images.bin')
    write_points3D_binary(pts3d, confidence, colors, output_path / 'points3D.bin')

    print(f"\n{'='*60}")
    print(f"✓ Output: {output_dir}/")
    print(f"{'='*60}")


def create_colmap_bins(scene, image_paths, output_dir, conf_threshold=1.5):
    """
    Main Function: Create COLMAP binary files
    
    Usage:
        create_colmap_bins(scene, image_paths, '/output/sparse/0', conf_threshold=1.5)
    """
    print("="*60)
    print("COLMAP BINARY FILES CREATION")
    print("="*60)

    cameras_dict, pts3d, confidence, pts3d_raw, conf_raw = extract_camera_params(
        scene, image_paths, conf_threshold
    )

    colors = extract_colors_from_images(
        image_paths, pts3d, conf_raw, conf_threshold
    )

    img = Image.open(image_paths[0])
    image_size = img.size
    img.close()

    export_colmap_binary(
        cameras_dict, pts3d, confidence, colors, image_size, output_dir
    )

    print("✓ COMPLETE!")
    return cameras_dict, pts3d, confidence, colors


# Optional: PLY Output
def write_colored_ply(pts3d, colors, output_path):
    """Output PLY file (with colors)"""
    with open(output_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts3d)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for pt, color in zip(pts3d, colors):
            r, g, b = [int(np.clip(c, 0, 255)) for c in color]
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {r} {g} {b}\n")

    print(f"✓ PLY: {len(pts3d)} points")
