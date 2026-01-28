# =====================================================================
# CELL 20: Traditional Method Functions (FIXED VERSION)
# =====================================================================
# 修正内容: points3D.binをワールド座標系で出力するように変更
# カメラ座標系 → ワールド座標系への変換を追加
# =====================================================================
import struct
import numpy as np
from pathlib import Path
import torch
import os
from PIL import Image

# ===== Traditional Method: extract_colmap_data (FIXED) =====
def extract_colmap_data_traditional_fixed(scene, image_paths, max_points=1000000):
    """
    Traditional Method (FIXED): Extract COLMAP-compatible data from a MASt3R scene.
    
    修正点:
    - カメラ座標系の点をワールド座標系に変換
    - 点の重複を解消 (各ビューの平均を取る)
    """
    print("\n=== [TRADITIONAL FIXED] Extracting COLMAP-compatible data ===")

    # Extract point cloud (カメラ座標系)
    pts_all = scene.get_pts3d()
    print(f"pts_all type: {type(pts_all)}")

    if isinstance(pts_all, list):
        print(f"pts_all is a list with {len(pts_all)} elements")
        if len(pts_all) > 0:
            print(f"First element type: {type(pts_all[0])}")
            if hasattr(pts_all[0], 'shape'):
                print(f"First element shape: {pts_all[0].shape}")

        pts_all = torch.stack([p if isinstance(p, torch.Tensor) else torch.tensor(p)
                              for p in pts_all])
        print(f"pts_all shape after conversion: {pts_all.shape}")

    # Get camera-to-world poses
    poses_c2w = scene.get_im_poses().detach().cpu().numpy()
    print(f"Retrieved camera-to-world poses: shape {poses_c2w.shape}")

    # 🔧 NEW: Convert camera coordinates to world coordinates
    print("\n🔄 Converting camera coordinates to world coordinates...")
    
    if len(pts_all.shape) == 4:
        # Batched point cloud: (B, H, W, 3)
        print(f"Found batched point cloud: {pts_all.shape}")
        B, H, W, _ = pts_all.shape
        
        # Convert each view to world coordinates
        pts3d_world_list = []
        for i in range(B):
            pts_cam = pts_all[i].reshape(-1, 3).detach().cpu().numpy()  # (H*W, 3)
            
            # Camera coords → World coords
            pts_homo = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])  # (N, 4)
            pts_world = (poses_c2w[i] @ pts_homo.T).T[:, :3]  # (N, 3)
            
            pts3d_world_list.append(pts_world)
            
            if i == 0:
                print(f"  View {i} example:")
                print(f"    Camera coords (first point): {pts_cam[0]}")
                print(f"    World coords (first point):  {pts_world[0]}")
        
        # Average across all views to merge duplicates
        pts3d_world_array = np.array(pts3d_world_list)  # (B, H*W, 3)
        print(f"\n  All views world coords shape: {pts3d_world_array.shape}")
        
        pts3d = np.mean(pts3d_world_array, axis=0)  # (H*W, 3)
        print(f"  ✓ Averaged across {B} views: {pts3d.shape}")
        
        # Extract colors
        colors = []
        for img_path in image_paths:
            img = Image.open(img_path).resize((W, H))
            colors.append(np.array(img))
        colors = np.stack(colors).reshape(-1, 3) / 255.0
        
        # Average colors as well
        colors = colors.reshape(B, -1, 3).mean(axis=0)  # (H*W, 3)
        
    elif len(pts_all.shape) == 3:
        # Multi-view point cloud: (N_views, N_points, 3)
        print(f"Found multi-view point cloud: {pts_all.shape}")
        N_views, N_points, _ = pts_all.shape
        
        # Convert each view to world coordinates
        pts3d_world_list = []
        for i in range(N_views):
            pts_cam = pts_all[i].detach().cpu().numpy()  # (N_points, 3)
            
            # Camera coords → World coords
            pts_homo = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])  # (N, 4)
            pts_world = (poses_c2w[i] @ pts_homo.T).T[:, :3]  # (N, 3)
            
            pts3d_world_list.append(pts_world)
            
            if i == 0:
                print(f"  View {i} example:")
                print(f"    Camera coords (first point): {pts_cam[0]}")
                print(f"    World coords (first point):  {pts_world[0]}")
        
        # Average across all views
        pts3d_world_array = np.array(pts3d_world_list)  # (N_views, N_points, 3)
        print(f"\n  All views world coords shape: {pts3d_world_array.shape}")
        
        pts3d = np.mean(pts3d_world_array, axis=0)  # (N_points, 3)
        print(f"  ✓ Averaged across {N_views} views: {pts3d.shape}")
        
        # Colors (gray)
        colors = np.ones((len(pts3d), 3)) * 0.5
        
    else:
        # Fallback: assume already in world coordinates (unlikely)
        print(f"⚠ Unexpected shape: {pts_all.shape}, using as-is")
        pts3d = pts_all.detach().cpu().numpy() if isinstance(pts_all, torch.Tensor) else pts_all
        colors = np.ones((len(pts3d), 3)) * 0.5

    print(f"\n✓ Final point cloud in WORLD coordinates:")
    print(f"  Shape: {pts3d.shape}")
    print(f"  Mean: {pts3d.mean(axis=0)}")
    print(f"  Std:  {pts3d.std(axis=0)}")
    print(f"  Min:  {pts3d.min(axis=0)}")
    print(f"  Max:  {pts3d.max(axis=0)}")

    # Downsample points if needed
    if len(pts3d) > max_points:
        print(f"\n⚠ Downsampling from {len(pts3d)} to {max_points} points...")
        valid_mask = ~(np.isnan(pts3d).any(axis=1) | np.isinf(pts3d).any(axis=1))
        pts3d_valid = pts3d[valid_mask]
        colors_valid = colors[valid_mask]
        
        num_excluded = len(pts3d_valid) - max_points
        
        indices = np.random.choice(len(pts3d_valid), size=max_points, replace=False)
        pts3d = pts3d_valid[indices]
        colors = colors_valid[indices]
        print(f"✓ Downsampled to {len(pts3d)} points")
        print(f"⚠ Excluded {num_excluded} points due to max_points limit")

    # Extract camera parameters
    print("\nExtracting camera parameters...")

    # Convert C2W to W2C for COLMAP
    poses = []
    for i, pose_c2w in enumerate(poses_c2w):
        pose_w2c = np.linalg.inv(pose_c2w)
        poses.append(pose_w2c)
    poses = np.array(poses)
    print("Converted to world-to-camera poses for COLMAP")

    focals = scene.get_focals().detach().cpu().numpy()
    pp = scene.get_principal_points().detach().cpu().numpy()
    print(f"Focals shape: {focals.shape}")
    print(f"Principal points shape: {pp.shape}")

    mast3r_size = 224.0

    cameras = []
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        W, H = img.size
        scale = W / mast3r_size

        if focals.shape[1] == 1:
            focal_mast3r = float(focals[i, 0])
            fx = fy = focal_mast3r * scale
        else:
            fx = float(focals[i, 0]) * scale
            fy = float(focals[i, 1]) * scale

        cx = float(pp[i, 0]) * scale
        cy = float(pp[i, 1]) * scale

        camera = {
            'camera_id': i + 1,
            'model': 'PINHOLE',
            'width': W,
            'height': H,
            'params': [fx, fy, cx, cy]
        }
        cameras.append(camera)

        if i == 0:
            print(f"\nExample camera 0:")
            print(f"  Image size: {W}x{H}")
            print(f"  MASt3R focal: {focal_mast3r:.2f}, pp: ({pp[i,0]:.2f}, {pp[i,1]:.2f})")
            print(f"  Scaled fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            print(f"  Pose (first row): {poses[i][0]}")

    print(f"\n✓ Extracted {len(cameras)} cameras and {len(poses)} poses")
    print(f"✓ Points are now in WORLD coordinate system")
    print(f"✓ Duplicate points merged: original views × points → {len(pts3d)} unique points")

    return pts3d, colors, cameras, poses


# ===== Traditional Method: rotmat2qvec (unchanged) =====
def rotmat2qvec_traditional(R):
    """Traditional Method: Convert rotation matrix to quaternion."""
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


# ===== Traditional Method: Save Functions (unchanged) =====
def write_cameras_binary_traditional(cameras, output_file):
    """Traditional Method: Write cameras.bin."""
    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', len(cameras)))

        for i, cam in enumerate(cameras):
            camera_id = cam.get('camera_id', i + 1)
            model_id = 1  # PINHOLE
            width = cam['width']
            height = cam['height']
            params = cam['params']

            f.write(struct.pack('i', camera_id))
            f.write(struct.pack('i', model_id))
            f.write(struct.pack('Q', width))
            f.write(struct.pack('Q', height))

            for param in params[:4]:
                f.write(struct.pack('d', param))


def write_images_binary_traditional(image_paths, cameras, poses, output_file):
    """Traditional Method: Write images.bin."""
    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', len(image_paths)))

        for i, (img_path, pose) in enumerate(zip(image_paths, poses)):
            image_id = i + 1
            camera_id = cameras[i].get('camera_id', i + 1)
            image_name = os.path.basename(img_path)

            R = pose[:3, :3]
            t = pose[:3, 3]
            qvec = rotmat2qvec_traditional(R)
            tvec = t

            f.write(struct.pack('i', image_id))
            for q in qvec:
                f.write(struct.pack('d', float(q)))
            for tv in tvec:
                f.write(struct.pack('d', float(tv)))
            f.write(struct.pack('i', camera_id))
            f.write(image_name.encode('utf-8') + b'\x00')
            f.write(struct.pack('Q', 0))


def write_points3d_binary_traditional(pts3d, colors, output_file):
    """Traditional Method: Write points3D.bin."""
    valid_indices = []
    invalid_count = 0
    
    for i, pt in enumerate(pts3d):
        if not (np.isnan(pt).any() or np.isinf(pt).any()):
            valid_indices.append(i)
        else:
            invalid_count += 1

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', len(valid_indices)))

        for idx, point_id in enumerate(valid_indices):
            pt = pts3d[point_id]
            color = colors[point_id]

            f.write(struct.pack('Q', point_id))
            for coord in pt:
                f.write(struct.pack('d', float(coord)))

            col_int = (color * 255).astype(np.uint8)
            for c in col_int:
                f.write(struct.pack('B', int(c)))

            f.write(struct.pack('d', 0.0))
            f.write(struct.pack('Q', 0))

    if invalid_count > 0:
        print(f"  ⚠ Excluded {invalid_count} invalid points (NaN/Inf)")

    return len(valid_indices)


def save_colmap_reconstruction_traditional_fixed(pts3d, colors, cameras, poses, image_paths, output_dir):
    """Traditional Method (FIXED): Save COLMAP reconstruction."""
    print("\n=== [TRADITIONAL FIXED] Saving COLMAP reconstruction ===")

    sparse_dir = Path(output_dir) / 'sparse_traditional_fixed' / '0'
    sparse_dir.mkdir(parents=True, exist_ok=True)

    write_cameras_binary_traditional(cameras, sparse_dir / 'cameras.bin')
    print(f"  ✓ Wrote {len(cameras)} cameras")

    write_images_binary_traditional(image_paths, cameras, poses, sparse_dir / 'images.bin')
    print(f"  ✓ Wrote {len(image_paths)} images")

    num_points = write_points3d_binary_traditional(pts3d, colors, sparse_dir / 'points3D.bin')
    print(f"  ✓ Wrote {num_points} 3D points (WORLD coordinates)")

    print(f"\n✓ Traditional COLMAP reconstruction (FIXED) saved to {sparse_dir}")
    print(f"✓ Coordinate system: WORLD (consistent with images.bin W2C poses)")

    return sparse_dir


# =====================================================================
# 使用例
# =====================================================================

# 従来版の関数名を保持しつつ、修正版を使用する場合:
# extract_colmap_data_traditional = extract_colmap_data_traditional_fixed
# save_colmap_reconstruction_traditional = save_colmap_reconstruction_traditional_fixed

# または、明示的に修正版を呼び出す:
"""
# MASt3Rシーンから抽出 (修正版)
pts3d, colors, cameras, poses = extract_colmap_data_traditional_fixed(
    scene, image_paths, max_points=1000000
)

# COLMAP形式で保存 (修正版)
sparse_dir = save_colmap_reconstruction_traditional_fixed(
    pts3d, colors, cameras, poses, image_paths, 
    output_dir='/kaggle/working/output'
)
"""
