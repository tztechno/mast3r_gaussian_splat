# =====================================================================
# CELL 12: COLMAP Export Functions (PINHOLE版) (REVISED 2026/01/26)
# Point3D.binがカメラ座標系になっているのが問題、世界座標系に直すべき
# The problem is that Point3D.bin is in camera coordinate system. It should be changed to world coordinate system.
# =====================================================================

import struct
import numpy as np
from pathlib import Path
import torch
import os
from PIL import Image

def rotmat_to_qvec(R):
    """回転行列をクォータニオンに変換"""
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
    qvec = qvec / np.linalg.norm(qvec)

    return qvec


def write_cameras_binary(cameras_dict, image_size, output_file):
    """
    cameras.binを出力（PINHOLEモデル使用）
    """
    width, height = image_size
    num_cameras = len(cameras_dict)

    # COLMAP camera models
    PINHOLE = 1  # 🔧 SIMPLE_PINHOLE (0) から PINHOLE (1) に変更

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', num_cameras))

        for camera_id, (img_id, cam_params) in enumerate(cameras_dict.items(), start=1):
            focal = cam_params['focal']

            # PINHOLEの場合: fx, fy, cx, cy
            #fx = fy = focal  # 等方性カメラを仮定
            
            #new settiing 2026/01/26
            if isinstance(focal, (tuple, list)):
                fx, fy = focal
            else:
                fx = fy = focal
            

            # Principal pointを取得（存在しない場合は中心）
            if 'pp' in cam_params:
                pp = cam_params['pp']
                cx = float(pp[0])
                cy = float(pp[1])
            else:
                cx = width / 2.0
                cy = height / 2.0

            # camera_id
            f.write(struct.pack('I', camera_id))
            # model_id (PINHOLE = 1)
            f.write(struct.pack('i', PINHOLE))
            # width
            f.write(struct.pack('Q', width))
            # height
            f.write(struct.pack('Q', height))
            # params: fx, fy, cx, cy (4パラメータ)
            f.write(struct.pack('d', fx))
            f.write(struct.pack('d', fy))
            f.write(struct.pack('d', cx))
            f.write(struct.pack('d', cy))

    print(f"COLMAP cameras.bin saved to {output_file}")


def write_images_binary(cameras_dict, output_file):
    """images.binを出力"""
    num_images = len(cameras_dict)

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', num_images))

        for image_id, (img_id, cam_params) in enumerate(cameras_dict.items(), start=1):
            R = cam_params['rotation']
            quat = rotmat_to_qvec(R)
            t = cam_params['translation']
            camera_id = image_id

            f.write(struct.pack('I', image_id))
            for q in quat:
                f.write(struct.pack('d', q))
            for ti in t:
                f.write(struct.pack('d', ti))
            f.write(struct.pack('I', camera_id))

            name_bytes = img_id.encode('utf-8') + b'\x00'
            f.write(name_bytes)
            f.write(struct.pack('Q', 0))

    print(f"COLMAP images.bin saved to {output_file}")


def write_points3D_binary(pts3d, confidence, output_file):
    """points3D.binを出力"""
    num_points = len(pts3d)

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', num_points))

        for point_id, pt in enumerate(pts3d, start=1):
            x, y, z = pt

            f.write(struct.pack('Q', point_id))
            f.write(struct.pack('d', x))
            f.write(struct.pack('d', y))
            f.write(struct.pack('d', z))

            # RGB (グレー)
            f.write(struct.pack('B', 128))
            f.write(struct.pack('B', 128))
            f.write(struct.pack('B', 128))

            # error
            if confidence is not None and point_id <= len(confidence):
                error = 1.0 / max(confidence[point_id-1], 0.001)
            else:
                error = 1.0
            f.write(struct.pack('d', error))

            # track_length
            f.write(struct.pack('Q', 0))

    print(f"COLMAP points3D.bin saved to {output_file}")


def export_colmap_binary(cameras_dict, pts3d, confidence, image_size, output_dir):
    """COLMAPバイナリファイルを出力"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    write_cameras_binary(
        cameras_dict,
        image_size,
        output_path / 'cameras.bin'
    )

    write_images_binary(
        cameras_dict,
        output_path / 'images.bin'
    )

    write_points3D_binary(
        pts3d,
        confidence,
        output_path / 'points3D.bin'
    )

    print(f"\nCOLMAP binary files exported to {output_dir}/")
    print(f"  - cameras.bin: {len(cameras_dict)} cameras (PINHOLE model)")
    print(f"  - images.bin: {len(cameras_dict)} images")
    print(f"  - points3D.bin: {len(pts3d)} points")


# =====================================================================
# CELL 11: Camera Parameter Extraction (REVISED 2026/01/26)
# =====================================================================
# =====================================================================
# CELL 11: Camera Parameter Extraction (FIXED VERSION)
# =====================================================================
# 修正内容: カメラ座標系の点をワールド座標系に変換
# Process2用の修正版 extract_camera_params_process2_fixed
# =====================================================================

def extract_camera_params_process2(scene, image_paths, conf_threshold=1.5):
    """
    Extracts camera parameters and 3D points from the scene (FIXED VERSION).
    
    修正点:
    - カメラ座標系の点をワールド座標系に変換
    - 点の重複を解消 (各ビューの平均を取る)
    """
    print("\n=== [PROCESS2 FIXED] Extracting Camera Parameters ===")

    cameras_dict = {}
    all_pts3d_world = []  # 🔧 NEW: ワールド座標系の点を格納
    all_confidence = []

    try:
        # Attempt to get camera poses
        if hasattr(scene, 'get_im_poses'):
            poses_c2w = scene.get_im_poses()  # Camera-to-World poses
        elif hasattr(scene, 'im_poses'):
            poses_c2w = scene.im_poses
        else:
            poses_c2w = None

        # Attempt to get focal lengths
        if hasattr(scene, 'get_focals'):
            focals = scene.get_focals()
        elif hasattr(scene, 'im_focals'):
            focals = scene.im_focals
        else:
            focals = None

        # Attempt to get principal points
        if hasattr(scene, 'get_principal_points'):
            pps = scene.get_principal_points()
        elif hasattr(scene, 'im_pp'):
            pps = scene.im_pp
        else:
            pps = None
    except Exception as e:
        print(f"⚠️ Error getting camera parameters: {e}")
        poses_c2w = None
        focals = None
        pps = None

    # Convert poses to numpy
    if poses_c2w is not None:
        if isinstance(poses_c2w, torch.Tensor):
            poses_c2w_np = poses_c2w.detach().cpu().numpy()
        else:
            poses_c2w_np = np.array(poses_c2w)
        print(f"Retrieved camera-to-world poses: shape {poses_c2w_np.shape}")
    else:
        poses_c2w_np = None

    # [Important] MASt3R internal processing size
    mast3r_size = 224.0

    n_images = min(len(poses_c2w_np) if poses_c2w_np is not None else len(image_paths), len(image_paths))

    # 🔧 NEW: First pass - collect all points in world coordinates
    print("\n🔄 Converting camera coordinates to world coordinates...")
    
    for idx in range(n_images):
        img_name = os.path.basename(image_paths[idx])

        try:
            # Get original image dimensions
            img = Image.open(image_paths[idx])
            W, H = img.size
            img.close()

            # Calculate scaling ratio
            scale = W / mast3r_size

            # Get Pose (Convert camera-to-world to world-to-camera for COLMAP)
            if poses_c2w_np is not None and idx < len(poses_c2w_np):
                pose_c2w = poses_c2w_np[idx]
                if not isinstance(pose_c2w, np.ndarray) or pose_c2w.shape != (4, 4):
                    pose_c2w = np.eye(4)

                # Invert to get world-to-camera pose (for COLMAP)
                pose_w2c = np.linalg.inv(pose_c2w)
            else:
                pose_c2w = np.eye(4)
                pose_w2c = np.eye(4)

            # Get and scale focal length
            if focals is not None and idx < len(focals):
                focal_mast3r = focals[idx]
                if isinstance(focal_mast3r, torch.Tensor):
                    focal_mast3r = focal_mast3r.detach().cpu()

                # Check if isotropic (fx = fy) or anisotropic (fx ≠ fy)
                if focals.shape[1] == 1:
                    # Isotropic camera (fx = fy)
                    focal_val = float(focal_mast3r) if focal_mast3r.numel() == 1 else float(focal_mast3r[0])
                    fx = fy = focal_val * scale
                else:
                    # Anisotropic camera (fx ≠ fy)
                    fx = float(focal_mast3r[0]) * scale
                    fy = float(focal_mast3r[1]) * scale
            else:
                # Default fallback
                fx = fy = 1000.0

            # Get and scale principal point
            if pps is not None and idx < len(pps):
                pp_mast3r = pps[idx]
                if isinstance(pp_mast3r, torch.Tensor):
                    pp_mast3r = pp_mast3r.detach().cpu().numpy()

                # Apply scaling
                pp = pp_mast3r * scale
            else:
                pp = np.array([W / 2.0, H / 2.0])

            # Store camera parameters
            cameras_dict[img_name] = {
                'focal': (fx, fy),
                'pp': pp,
                'pose': pose_w2c,  # W2C for COLMAP
                'rotation': pose_w2c[:3, :3],
                'translation': pose_w2c[:3, 3],
                'width': W,
                'height': H
            }

            # Debugging info (First image only)
            if idx == 0:
                print(f"\nExample camera 0:")
                print(f"  Original size: {W}x{H}")
                print(f"  MASt3R size: {mast3r_size}")
                print(f"  Scale factor: {scale:.3f}")
                print(f"  focals.shape: {focals.shape}")
                if focals.shape[1] == 1:
                    print(f"  MASt3R focal: {focal_val:.2f}")
                    print(f"  Scaled focal: fx = fy = {fx:.2f}")
                else:
                    print(f"  MASt3R focals: fx={float(focal_mast3r[0]):.2f}, fy={float(focal_mast3r[1]):.2f}")
                    print(f"  Scaled focals: fx={fx:.2f}, fy={fy:.2f}")
                print(f"  MASt3R pp: [{pp_mast3r[0]:.2f}, {pp_mast3r[1]:.2f}]")
                print(f"  Scaled pp: [{pp[0]:.2f}, {pp[1]:.2f}]")

            # 🔧 NEW: Extract 3D points and convert to world coordinates
            if hasattr(scene, 'im_pts3d') and idx < len(scene.im_pts3d):
                pts3d_cam = scene.im_pts3d[idx]  # Camera coordinates
            elif hasattr(scene, 'get_pts3d'):
                pts3d_all = scene.get_pts3d()
                pts3d_cam = pts3d_all[idx] if idx < len(pts3d_all) else None
            else:
                pts3d_cam = None

            # Extract confidence scores
            if hasattr(scene, 'im_conf') and idx < len(scene.im_conf):
                conf_img = scene.im_conf[idx]
            elif hasattr(scene, 'get_conf'):
                conf_all = scene.get_conf()
                conf_img = conf_all[idx] if idx < len(conf_all) else None
            else:
                conf_img = None

            # 🔧 NEW: Convert camera coordinates to world coordinates
            if pts3d_cam is not None:
                if isinstance(pts3d_cam, torch.Tensor):
                    pts3d_cam = pts3d_cam.detach().cpu().numpy()

                pts3d_cam_flat = pts3d_cam.reshape(-1, 3) if pts3d_cam.ndim == 3 else pts3d_cam
                
                # Camera coords → World coords
                pts_homo = np.hstack([pts3d_cam_flat, np.ones((len(pts3d_cam_flat), 1))])
                pts3d_world = (pose_c2w @ pts_homo.T).T[:, :3]  # ✓ World coordinates
                
                all_pts3d_world.append(pts3d_world)
                
                if idx == 0:
                    print(f"\n  View {idx} coordinate transformation:")
                    print(f"    Camera coords (first point): {pts3d_cam_flat[0]}")
                    print(f"    World coords (first point):  {pts3d_world[0]}")

                # Process confidence
                if conf_img is not None:
                    if isinstance(conf_img, (list, torch.Tensor)):
                        conf_img = np.array(conf_img) if isinstance(conf_img, list) else conf_img.detach().cpu().numpy()

                    conf_flat = conf_img.reshape(-1) if conf_img.ndim > 1 else conf_img
                    
                    if len(conf_flat) != len(pts3d_world):
                        conf_flat = np.ones(len(pts3d_world))
                    
                    all_confidence.append(conf_flat)
                else:
                    all_confidence.append(np.ones(len(pts3d_world)))

        except Exception as e:
            print(f"⚠️ Error processing image {idx} ({img_name}): {e}")
            # Fallback to default values
            img = Image.open(image_paths[idx])
            W, H = img.size
            img.close()

            cameras_dict[img_name] = {
                'focal': (1000.0 * (W / mast3r_size), 1000.0 * (W / mast3r_size)),
                'pp': np.array([W / 2.0, H / 2.0]),
                'pose': np.eye(4),
                'rotation': np.eye(3),
                'translation': np.zeros(3),
                'width': W,
                'height': H
            }
            continue

    # 🔧 NEW: Merge points from all views (average)
    if all_pts3d_world:
        pts3d_world_array = np.array(all_pts3d_world)  # (N_views, N_points, 3)
        print(f"\n✓ All views world coords shape: {pts3d_world_array.shape}")
        
        # Average across all views to merge duplicates
        pts3d = np.mean(pts3d_world_array, axis=0)  # (N_points, 3)
        print(f"✓ Averaged across {len(all_pts3d_world)} views: {pts3d.shape}")
        
        # Average confidence as well
        if all_confidence:
            confidence_array = np.array(all_confidence)  # (N_views, N_points)
            confidence = np.mean(confidence_array, axis=0)  # (N_points,)
        else:
            confidence = np.ones(len(pts3d))
    else:
        pts3d = np.zeros((0, 3))
        confidence = np.zeros(0)

    print(f"\n✓ Final point cloud in WORLD coordinates:")
    print(f"  Shape: {pts3d.shape}")
    print(f"  Mean: {pts3d.mean(axis=0)}")
    print(f"  Std:  {pts3d.std(axis=0)}")
    print(f"  Min:  {pts3d.min(axis=0)}")
    print(f"  Max:  {pts3d.max(axis=0)}")

    print(f"\n✓ Extracted parameters for {len(cameras_dict)} cameras")
    print(f"✓ Total 3D points (before filtering): {len(pts3d)}")

    # Filter points by confidence
    if len(confidence) > 0:
        valid_mask = confidence > conf_threshold
        pts3d = pts3d[valid_mask]
        confidence = confidence[valid_mask]
        print(f"✓ Points after confidence filtering (>{conf_threshold}): {len(pts3d)}")
    
    print(f"✓ Points are now in WORLD coordinate system")
    print(f"✓ Duplicate points merged: {len(all_pts3d_world)} views → {len(pts3d)} unique points")

    return cameras_dict, pts3d, confidence


# =====================================================================
# Complete Color Extraction for Process2 (newly defined 2026/01/26)
# =====================================================================

import numpy as np
from PIL import Image
import struct
from pathlib import Path

# =====================================================================
# STEP 1: Color Extraction Function
# =====================================================================

def extract_colors_from_images(scene, image_paths, pts3d, confidence, conf_threshold=1.5):
    """
    Extract colors from images that match the filtered pts3d.
    
    This matches Traditional method's color extraction.
    
    Args:
        scene: MASt3R scene object
        image_paths: List of image file paths
        pts3d: (N, 3) filtered 3D points (after confidence filtering)
        confidence: (N,) filtered confidence scores
        conf_threshold: Confidence threshold used for filtering
    
    Returns:
        colors: (N, 3) RGB colors [0-255] matching pts3d
    """
    print("\n=== Extracting Colors from Images ===")
    
    # Get all 3D points BEFORE filtering (to match with colors)
    all_pts3d = []
    for idx in range(len(image_paths)):
        if hasattr(scene, 'im_pts3d') and idx < len(scene.im_pts3d):
            pts3d_img = scene.im_pts3d[idx]
        elif hasattr(scene, 'get_pts3d'):
            pts3d_all = scene.get_pts3d()
            pts3d_img = pts3d_all[idx] if idx < len(pts3d_all) else None
        else:
            pts3d_img = None
        
        if pts3d_img is not None:
            if isinstance(pts3d_img, torch.Tensor):
                pts3d_img = pts3d_img.detach().cpu().numpy()
            pts3d_flat = pts3d_img.reshape(-1, 3) if pts3d_img.ndim == 3 else pts3d_img
            all_pts3d.append(pts3d_flat)
    
    # Get dimensions from first image
    first_img = Image.open(image_paths[0])
    W_orig, H_orig = first_img.size
    first_img.close()
    
    # MASt3R uses 224x224 internally
    mast3r_size = 224
    
    # Extract colors from all images
    print(f"Extracting colors from {len(image_paths)} images...")
    all_colors = []
    
    for idx, img_path in enumerate(image_paths):
        # Open and resize image to MASt3R size (224x224)
        img = Image.open(img_path)
        img_resized = img.resize((mast3r_size, mast3r_size), Image.BILINEAR)
        img_array = np.array(img_resized)  # Shape: (224, 224, 3)
        img.close()
        
        # Reshape to (224*224, 3) to match point order
        colors_flat = img_array.reshape(-1, 3)
        all_colors.append(colors_flat)
        
        if idx == 0:
            print(f"  Example image 0:")
            print(f"    Original size: {W_orig}x{H_orig}")
            print(f"    Resized to: {mast3r_size}x{mast3r_size}")
            print(f"    Colors shape: {colors_flat.shape}")
    
    # Stack all colors
    colors_all = np.vstack(all_colors)  # Shape: (N_total, 3)
    print(f"✓ Total colors extracted: {len(colors_all):,}")
    
    # Get confidence for all points (before filtering)
    all_conf = []
    for idx in range(len(image_paths)):
        if hasattr(scene, 'im_conf') and idx < len(scene.im_conf):
            conf_img = scene.im_conf[idx]
        elif hasattr(scene, 'get_conf'):
            conf_all = scene.get_conf()
            conf_img = conf_all[idx] if idx < len(conf_all) else None
        else:
            conf_img = None
        
        if conf_img is not None:
            if isinstance(conf_img, torch.Tensor):
                conf_img = conf_img.detach().cpu().numpy()
            conf_flat = conf_img.reshape(-1) if conf_img.ndim > 1 else conf_img
        else:
            conf_flat = np.ones(len(all_pts3d[idx]))
        
        all_conf.append(conf_flat)
    
    conf_all = np.concatenate(all_conf)
    
    # Apply THE SAME filtering as pts3d
    valid_mask = conf_all > conf_threshold
    colors_filtered = colors_all[valid_mask]
    
    print(f"✓ Colors after confidence filtering (>{conf_threshold}): {len(colors_filtered):,}")
    
    # Verify shapes match
    if len(colors_filtered) != len(pts3d):
        print(f"⚠️ WARNING: Color count ({len(colors_filtered)}) != Point count ({len(pts3d)})")
        print(f"  Adjusting to match...")
        min_len = min(len(colors_filtered), len(pts3d))
        colors_filtered = colors_filtered[:min_len]
    else:
        print(f"✓ Colors match points: {len(colors_filtered):,} colors for {len(pts3d):,} points")
    
    # Verify colors are diverse
    unique_colors = len(np.unique(colors_filtered, axis=0))
    print(f"✓ Unique colors: {unique_colors:,}")
    
    if unique_colors < 100:
        print(f"⚠️ WARNING: Very few unique colors!")
    else:
        print(f"✓ Good color diversity")
    
    return colors_filtered


# =====================================================================
# STEP 2: Write points3D.bin with Colors
# =====================================================================

def write_points3D_binary_with_colors(pts3d, confidence, colors, output_file):
    """
    Export points3D.bin with actual colors.
    
    Args:
        pts3d: (N, 3) array of 3D points
        confidence: (N,) array of confidence scores
        colors: (N, 3) array of RGB colors [0-255]
        output_file: Path to output file
    """
    num_points = len(pts3d)

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', num_points))

        for point_id, (pt, color) in enumerate(zip(pts3d, colors), start=1):
            x, y, z = pt

            f.write(struct.pack('Q', point_id))
            f.write(struct.pack('d', x))
            f.write(struct.pack('d', y))
            f.write(struct.pack('d', z))

            # RGB Color (ACTUAL colors now!)
            r = int(np.clip(color[0], 0, 255))
            g = int(np.clip(color[1], 0, 255))
            b = int(np.clip(color[2], 0, 255))
            
            f.write(struct.pack('B', r))
            f.write(struct.pack('B', g))
            f.write(struct.pack('B', b))

            # Error estimation
            if confidence is not None and point_id <= len(confidence):
                error = 1.0 / max(confidence[point_id-1], 0.001)
            else:
                error = 1.0
            f.write(struct.pack('d', error))

            # track_length (Set to 0)
            f.write(struct.pack('Q', 0))

    print(f"COLMAP points3D.bin saved to {output_file}")
    print(f"  ✓ With actual RGB colors from images!")


# =====================================================================
# STEP 3: Export with Colors
# =====================================================================

def export_colmap_binary_with_colors(cameras_dict, pts3d, confidence, colors, 
                                     image_size, output_dir):
    """
    Export COLMAP binary files with actual colors.
    
    Args:
        cameras_dict: Dictionary of camera parameters
        pts3d: (N, 3) filtered 3D points
        confidence: (N,) filtered confidence scores
        colors: (N, 3) RGB colors [0-255]
        image_size: (width, height) tuple
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write cameras.bin (same as before)
    write_cameras_binary(
        cameras_dict,
        image_size,
        output_path / 'cameras.bin'
    )

    # Write images.bin (same as before)
    write_images_binary(
        cameras_dict,
        output_path / 'images.bin'
    )

    # Write points3D.bin WITH COLORS (NEW!)
    write_points3D_binary_with_colors(
        pts3d,
        confidence,
        colors,  # ← Actual colors!
        output_path / 'points3D.bin'
    )

    print(f"\n✓ COLMAP binary files exported to {output_dir}/")
    print(f"  - cameras.bin: {len(cameras_dict)} cameras (PINHOLE model)")
    print(f"  - images.bin: {len(cameras_dict)} images")
    print(f"  - points3D.bin: {len(pts3d)} points WITH COLORS")


# =====================================================================
# STEP 4: Complete Workflow
# =====================================================================

def create_process2_with_colors(scene, image_paths, output_dir, conf_threshold=1.5):
    """
    Complete workflow: Process2 with color extraction.
    
    Usage:
        create_process2_with_colors(
            scene, 
            image_paths, 
            '/kaggle/working/output/sparse_process2_with_colors/0',
            conf_threshold=1.5
        )
    """
    print("="*80)
    print("CREATING PROCESS2 COLMAP WITH COLORS")
    print("="*80)
    
    # Step 1: Extract camera parameters and points
    cameras_dict, pts3d, confidence = extract_camera_params_process2(
        scene, image_paths, conf_threshold=conf_threshold
    )
    
    print(f"\n✓ Extracted:")
    print(f"  - {len(cameras_dict)} cameras")
    print(f"  - {len(pts3d):,} 3D points")
    
    # Step 2: Extract colors (NEW!)
    colors = extract_colors_from_images(
        scene, image_paths, pts3d, confidence, conf_threshold
    )
    
    # Step 3: Get image size
    img = Image.open(image_paths[0])
    image_size = img.size
    img.close()
    
    # Step 4: Export with colors
    export_colmap_binary_with_colors(
        cameras_dict, pts3d, confidence, colors,
        image_size, output_dir
    )
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print("\nOutput directory:", output_dir)
    print("\nNext steps:")
    print("1. Train 3DGS with this reconstruction")
    print("2. Compare quality with gray Process2 and Traditional")
    print("3. Check if colors improve geometry convergence")
    
    return cameras_dict, pts3d, confidence, colors
