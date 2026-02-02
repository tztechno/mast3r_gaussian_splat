#process3_13.py

import numpy as np
import cv2
from pathlib import Path
import struct
from scipy.spatial.transform import Rotation
import torch
from PIL import Image


def write_next_bytes(fid, data, format_str):
    """Helper function to write bytes to file"""
    if isinstance(data, (list, tuple, np.ndarray)):
        fid.write(struct.pack("<" + format_str, *data))
    else:
        fid.write(struct.pack("<" + format_str, data))


def matrix_to_quaternion_translation(matrix: np.ndarray):
    """Robust conversion of 4x4 transformation matrix to quaternion and translation."""
    R = matrix[:3, :3]
    t = matrix[:3, 3]

    # Use scipy for robust quaternion conversion
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    
    # COLMAP format is [w, x, y, z]
    qvec = np.array([quat[3], quat[0], quat[1], quat[2]])

    return qvec, t


def write_cameras_binary(cameras, path_to_model_file):
    """Write COLMAP cameras.bin file"""
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for camera_id, cam in cameras.items():
            model_id = 1  # PINHOLE
            write_next_bytes(fid, camera_id, "I")
            write_next_bytes(fid, model_id, "I")
            write_next_bytes(fid, cam['width'], "Q")
            write_next_bytes(fid, cam['height'], "Q")
            for p in cam['params']:
                write_next_bytes(fid, float(p), "d")


def write_images_binary(images, path_to_model_file):
    """Write COLMAP images.bin file"""
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for image_id, img in images.items():
            write_next_bytes(fid, image_id, "I")
            write_next_bytes(fid, img['qvec'], "dddd")
            write_next_bytes(fid, img['tvec'], "ddd")
            write_next_bytes(fid, img['camera_id'], "I")
            
            # Write image name
            for char in img['name']:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            
            # Write 2D points
            write_next_bytes(fid, len(img['xys']), "Q")
            for xy, point3D_id in zip(img['xys'], img['point3D_ids']):
                write_next_bytes(fid, xy, "dd")
                write_next_bytes(fid, point3D_id, "Q")


def write_points3d_binary(points3D, path_to_model_file):
    """
    Write COLMAP points3D.bin file
    
    Args:
        points3D: list or dict of 3D point data
        path_to_model_file: path to points3D.bin
    """
    with open(path_to_model_file, "wb") as fid:
        # Write number of points
        if isinstance(points3D, dict):
            write_next_bytes(fid, len(points3D), "Q")
            points_iter = points3D.values()
        else:
            write_next_bytes(fid, len(points3D), "Q")
            points_iter = points3D
        
        # Write each point
        for point_id, point in enumerate(points_iter):
            # Handle both dict with 'id' key and list with index
            if isinstance(point, dict) and 'id' in point:
                pid = point['id']
            else:
                pid = point_id
            
            write_next_bytes(fid, pid, "Q")
            write_next_bytes(fid, point['xyz'], "ddd")
            write_next_bytes(fid, point['rgb'], "BBB")
            write_next_bytes(fid, point['error'], "d")
            
            # Write track
            track_length = len(point['image_ids'])
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_idx in zip(point['image_ids'], point['point2D_idxs']):
                write_next_bytes(fid, int(image_id), "I")
                write_next_bytes(fid, int(point2D_idx), "I")


def save_image_data(scene, images_dir, depth_dir, normal_dir, mask_dir, min_conf_thr, verbose, processed_image_paths=None):
    """Save RGB images, depth maps, normal maps, and masks"""
    if verbose:
        print("\nSaving image data...")
    
    # Ensure directories exist
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the number of views
    if hasattr(scene, 'imgs'):
        num_views = len(scene.imgs)
        imgs = scene.imgs
    elif hasattr(scene, 'views'):
        num_views = len(scene.views)
        imgs = scene.views
    else:
        if verbose:
            print("  Warning: Cannot access views")
        return
    
    # Use processed images if provided
    if processed_image_paths is not None and len(processed_image_paths) > 0:
        if verbose:
            print(f"  Using {len(processed_image_paths)} processed images")
        
        import shutil
        for idx, src_path in enumerate(processed_image_paths):
            if idx >= num_views:
                break
            
            try:
                # Copy processed images
                dst_path = images_dir / f'image_{idx:04d}.jpg'
                shutil.copy2(src_path, dst_path)
                
                if verbose and idx < 3:
                    print(f"  Copied image {idx}: {Path(src_path).name}")
            except Exception as e:
                if verbose:
                    print(f"  Error copying image {idx}: {e}")
    else:
        # If no processed images, extract images from the scene
        if verbose:
            print("  No processed images provided, extracting from scene...")
        
        for idx in range(num_views):
            try:
                # Save RGB images
                img_path = images_dir / f'image_{idx:04d}.jpg'
                
                # Retrieve image data
                if hasattr(imgs[idx], 'img'):
                    img = imgs[idx].img
                elif hasattr(imgs[idx], 'image'):
                    img = imgs[idx].image
                else:
                    img = imgs[idx]
                
                # Convert tensor to numpy array
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu().numpy()
                
                # Convert image to correct format
                if isinstance(img, np.ndarray):
                    # Convert (C, H, W) -> (H, W, C)
                    if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                        img = np.transpose(img, (1, 2, 0))
                    
                    # Normalize values to [0, 255] range
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    
                    # Convert grayscale to RGB
                    if img.ndim == 2:
                        img = np.stack([img, img, img], axis=-1)
                    elif img.shape[-1] == 1:
                        img = np.repeat(img, 3, axis=-1)
                    
                    # Save the image
                    Image.fromarray(img).save(img_path)
                    
                    if verbose and idx < 3:
                        print(f"  Saved image {idx}: {img_path}")
            except Exception as e:
                if verbose:
                    print(f"  Error saving image {idx}: {e}")
    
    # Save depth maps
    try:
        if hasattr(scene, 'get_depthmaps'):
            depthmaps = scene.get_depthmaps()
            if depthmaps is not None:
                for idx in range(min(num_views, len(depthmaps))):
                    depth = depthmaps[idx]
                    if isinstance(depth, torch.Tensor):
                        depth = depth.detach().cpu().numpy()
                    
                    if isinstance(depth, np.ndarray):
                        depth_path = depth_dir / f'depth_{idx:04d}.npy'
                        np.save(depth_path, depth)
                        
                        if verbose and idx < 3:
                            print(f"  Saved depth {idx}: {depth_path}")
    except Exception as e:
        if verbose:
            print(f"  Note: Could not save depth maps: {e}")
    
    # Save masks
    try:
        if hasattr(scene, 'get_masks'):
            masks = scene.get_masks()
            if masks is not None:
                for idx in range(min(num_views, len(masks))):
                    mask = masks[idx]
                    if isinstance(mask, torch.Tensor):
                        mask = mask.detach().cpu().numpy()
                    
                    if isinstance(mask, np.ndarray):
                        mask_path = mask_dir / f'mask_{idx:04d}.png'
                        mask_img = (mask * 255).astype(np.uint8)
                        Image.fromarray(mask_img).save(mask_path)
                        
                        if verbose and idx < 3:
                            print(f"  Saved mask {idx}: {mask_path}")
    except Exception as e:
        if verbose:
            print(f"  Note: Could not save masks: {e}")
    
    if verbose:
        print(f"  Completed saving {num_views} images")




def extract_scene_data(scene, views, output_dir):
    """
    Extract camera parameters, 3D points, and image data from MASt3R scene
    and save in COLMAP format
    """
    import numpy as np
    import os
    import shutil
    from pathlib import Path
    
    print("\n" + "="*70)
    print("Converting MASt3R scene to COLMAP format")
    print("="*70)
    print(f"Output directory: {output_dir}")
    
    # Create output directories
    sparse_dir = os.path.join(output_dir, 'sparse', '0')
    image_dir = os.path.join(output_dir, 'images')
    depth_dir = os.path.join(output_dir, 'depth')
    mask_dir = os.path.join(output_dir, 'mask')
    
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    print("\nExtracting scene data...")
    
    # Get number of views
    num_views = len(views)
    print(f"Number of views: {num_views}")
    
    # Initialize COLMAP data structures
    cameras_data = {}
    images_data = {}
    points3D_data = {}
    
    # Extract 3D points with colors
    print("\nExtracting 3D points with colors...")
    pts3d = scene.get_pts3d(clip_thred=None)
    pts3d_np = pts3d.cpu().numpy()
    
    # Get confidence masks
    conf_list = scene.get_masks()
    
    # Extract colors from images
    colors_list = []
    valid_pts_list = []
    
    for idx, view in enumerate(views):
        img = view['img'].cpu().numpy()
        conf = conf_list[idx].cpu().numpy()
        pts = pts3d_np[idx]
        
        # Filter by confidence
        valid_mask = conf > 0.001
        valid_pts = pts[valid_mask]
        
        # Get colors (assuming img is in [H, W, 3] format with values 0-1)
        if img.max() <= 1.0:
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = img.astype(np.uint8)
        
        colors = img_uint8[valid_mask]
        
        valid_pts_list.append(valid_pts)
        colors_list.append(colors)
    
    # Combine all points and colors
    all_pts3d = np.concatenate(valid_pts_list, axis=0)
    all_colors = np.concatenate(colors_list, axis=0)
    
    # Remove invalid points
    valid_3d = np.isfinite(all_pts3d).all(axis=1)
    all_pts3d = all_pts3d[valid_3d]
    all_colors = all_colors[valid_3d]
    
    print(f"  Extracted {len(all_pts3d)} 3D points with colors")
    print(f"  Sample colors: {all_colors[:3].tolist()}")
    
    # Create points3D data
    for point_id, (xyz, rgb) in enumerate(zip(all_pts3d, all_colors), start=1):
        points3D_data[point_id] = Point3D(
            id=point_id,
            xyz=xyz.astype(np.float64),
            rgb=rgb.astype(np.uint8),
            error=0.0,
            image_ids=np.array([], dtype=np.int32),
            point2D_idxs=np.array([], dtype=np.int32)
        )
    
    print(f"\nTotal: {len(cameras_data)} cameras, {len(images_data)} images, {len(points3D_data)} points")
    
    # Get actual image filenames from processed_images directory
    print("\nSaving image data...")
    processed_images_dir = os.path.join(os.path.dirname(output_dir), 'processed_images')
    
    if not os.path.exists(processed_images_dir):
        raise FileNotFoundError(f"Processed images directory not found: {processed_images_dir}")
    
    actual_image_files = sorted([f for f in os.listdir(processed_images_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(actual_image_files) != len(views):
        raise ValueError(f"Number of images mismatch: {len(actual_image_files)} files vs {len(views)} views")
    
    print(f"  Using {len(actual_image_files)} processed images")
    
    # Process each view with actual filenames
    for idx, (view, actual_filename) in enumerate(zip(views, actual_image_files)):
        image_id = idx + 1
        
        # ★修正：実際のファイル名を使用
        image_name = actual_filename  # 例: "image_004_bottom.jpeg"
        
        # Get camera parameters
        focals = view['camera_intrinsics'][0, [0, 1, 1]].cpu().numpy()
        principal_point = view['camera_intrinsics'][0, [0, 1], [2, 2]].cpu().numpy()
        camera_id = idx + 1
        
        # Get image dimensions
        img_shape = view['img'].shape
        height, width = img_shape[0], img_shape[1]
        
        # Camera model: PINHOLE
        camera_model_id = 1  # PINHOLE in COLMAP
        params = np.array([
            focals[0],           # fx
            focals[1],           # fy
            principal_point[0],  # cx
            principal_point[1]   # cy
        ], dtype=np.float64)
        
        # Store camera data
        cameras_data[camera_id] = Camera(
            id=camera_id,
            model='PINHOLE',
            width=width,
            height=height,
            params=params
        )
        
        # Get pose (camera-to-world transformation)
        cam_to_world = view['camera_pose'].cpu().numpy()[0]
        
        # Convert to world-to-camera (COLMAP convention)
        world_to_cam = np.linalg.inv(cam_to_world)
        
        # Extract rotation and translation
        R = world_to_cam[:3, :3]
        t = world_to_cam[:3, 3]
        
        # Convert rotation matrix to quaternion (w, x, y, z)
        qvec = rotmat2qvec(R)
        tvec = t.astype(np.float64)
        
        # Store image data
        images_data[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,  # ★実際のファイル名を使用
            xys=np.zeros((0, 2)),
            point3D_ids=np.full(0, -1, dtype=np.int64)
        )
        
        # Copy image file with actual filename
        src_path = os.path.join(processed_images_dir, actual_filename)
        dst_path = os.path.join(image_dir, actual_filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  Copied image {idx}: {actual_filename}")
        else:
            print(f"  ⚠️  Warning: Image not found: {src_path}")
        
        # Save depth map
        depth = view['pts3d'][..., 2].cpu().numpy()  # Z-coordinate as depth
        depth_path = os.path.join(depth_dir, f'depth_{idx:04d}.npy')
        np.save(depth_path, depth)
        print(f"  Saved depth {idx}: {depth_path}")
        
        # Save confidence mask
        conf = conf_list[idx].cpu().numpy()
        mask_path = os.path.join(mask_dir, f'mask_{idx:04d}.png')
        mask_img = (conf * 255).astype(np.uint8)
        from PIL import Image as PILImage
        PILImage.fromarray(mask_img).save(mask_path)
        print(f"  Saved mask {idx}: {mask_path}")
    
    print(f"  Completed saving {len(views)} images")
    
    # Write COLMAP binary files
    print("\nWriting COLMAP binary files...")
    
    cameras_path = os.path.join(sparse_dir, 'cameras.bin')
    images_path = os.path.join(sparse_dir, 'images.bin')
    points3D_path = os.path.join(sparse_dir, 'points3D.bin')
    
    write_cameras_binary(cameras_data, cameras_path)
    print(f"  ✓ cameras.bin ({len(cameras_data)} cameras)")
    
    write_images_binary(images_data, images_path)
    print(f"  ✓ images.bin ({len(images_data)} images)")
    
    write_points3D_binary(points3D_data, points3D_path)
    print(f"  ✓ points3D.bin ({len(points3D_data)} points)")
    
    print("\n" + "="*70)
    print("✓ COLMAP conversion complete!")
    print("="*70)
    
    return output_dir



#-------------------------------------------------------------------------


def convert_mast3r_to_colmap(scene, output_dir, min_conf_thr=1.5, 
                             clean_depth=True, mask_images=True, 
                             verbose=True, processed_image_paths=None):
    """Convert MASt3R scene to COLMAP format"""
    import numpy as np
    import torch
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Converting MASt3R scene to COLMAP format...")
        print(f"Output directory: {output_dir}")
    
    # Get 3D points - handle both list and tensor cases
    pts3d = scene.get_pts3d()
    
    # Convert to numpy array
    if isinstance(pts3d, list):
        # List of tensors case
        pts3d_list = []
        for pts in pts3d:
            if isinstance(pts, torch.Tensor):
                pts3d_list.append(pts.cpu().numpy())
            else:
                pts3d_list.append(np.array(pts))
        pts3d_np = np.concatenate([p.reshape(-1, 3) for p in pts3d_list], axis=0)
    elif isinstance(pts3d, torch.Tensor):
        # Single tensor case
        pts3d_np = pts3d.cpu().numpy()
        if pts3d_np.ndim > 2:
            pts3d_np = pts3d_np.reshape(-1, 3)
    else:
        # Numpy array case
        pts3d_np = np.array(pts3d)
        if pts3d_np.ndim > 2:
            pts3d_np = pts3d_np.reshape(-1, 3)
    
    if verbose:
        print(f"Total 3D points shape: {pts3d_np.shape}")
    
    # Get confidence masks
    conf_masks = scene.get_masks()
    if isinstance(conf_masks, list):
        conf_list = []
        for conf in conf_masks:
            if isinstance(conf, torch.Tensor):
                conf_list.append(conf.cpu().numpy())
            else:
                conf_list.append(np.array(conf))
        confidences = np.concatenate([c.reshape(-1) for c in conf_list], axis=0)
    elif isinstance(conf_masks, torch.Tensor):
        confidences = conf_masks.cpu().numpy().reshape(-1)
    else:
        confidences = np.array(conf_masks).reshape(-1)
    
    # Filter points by confidence
    valid_mask = confidences > min_conf_thr
    pts3d_filtered = pts3d_np[valid_mask]
    
    if verbose:
        print(f"Points after confidence filtering: {pts3d_filtered.shape[0]}")
    
    # Remove invalid points (inf, nan)
    valid_pts = np.isfinite(pts3d_filtered).all(axis=1)
    pts3d_filtered = pts3d_filtered[valid_pts]
    
    if verbose:
        print(f"Points after removing invalid values: {pts3d_filtered.shape[0]}")
    
    # Get images
    imgs = scene.imgs
    if not isinstance(imgs, list):
        imgs = [imgs]
    
    num_images = len(imgs)
    if verbose:
        print(f"Number of images: {num_images}")
    
    # Create cameras.bin
    cameras = {}
    for i in range(num_images):
        img = imgs[i]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        h, w = img.shape[:2] if img.ndim == 3 else img.shape[1:3]
        
        # Simple PINHOLE camera model
        focal_length = max(w, h) * 1.2
        cameras[i+1] = {
            'id': i+1,
            'model': 'PINHOLE',
            'width': w,
            'height': h,
            'params': np.array([focal_length, focal_length, w/2, h/2])
        }
    
    write_cameras_binary(cameras, output_dir / 'cameras.bin')
    if verbose:
        print(f"Written {len(cameras)} cameras")
    
    # Create images.bin
    images_data = {}
    for i in range(num_images):
        qvec = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
        tvec = np.array([0.0, 0.0, 0.0])
        
        if processed_image_paths and i < len(processed_image_paths):
            image_name = os.path.basename(processed_image_paths[i])
        else:
            image_name = f"image_{i:04d}.jpg"
        
        images_data[i+1] = {
            'id': i+1,
            'qvec': qvec,
            'tvec': tvec,
            'camera_id': i+1,
            'name': image_name,
            'xys': np.zeros((0, 2)),  # No 2D points for now
            'point3D_ids': np.zeros(0, dtype=np.int64)
        }
    
    write_images_binary(images_data, output_dir / 'images.bin')
    if verbose:
        print(f"Written {len(images_data)} images")
    
    # Create points3D.bin
    points3d_data = {}
    num_valid_points = min(len(pts3d_filtered), 100000)  # Limit for memory
    
    for i in range(num_valid_points):
        points3d_data[i+1] = {
            'id': i+1,
            'xyz': pts3d_filtered[i],
            'rgb': np.array([128, 128, 128], dtype=np.uint8),
            'error': 1.0,
            'image_ids': np.array([], dtype=np.int32),
            'point2D_idxs': np.array([], dtype=np.int32)
        }
    
    write_points3D_binary(points3d_data, output_dir / 'points3D.bin')
    if verbose:
        print(f"Written {len(points3d_data)} 3D points")
    
    return output_dir
