#process2_05.py

import struct
import numpy as np
from pathlib import Path

def rotmat_to_qvec(R):
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
    Export cameras.bin using the PINHOLE camera model.
    """
    width, height = image_size
    num_cameras = len(cameras_dict)

    # COLMAP camera models
    PINHOLE = 1  # 🔧 Changed from SIMPLE_PINHOLE (0) to PINHOLE (1)

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', num_cameras))

        for camera_id, (img_id, cam_params) in enumerate(cameras_dict.items(), start=1):
            focal = cam_params['focal']

            # For PINHOLE model: [fx, fy, cx, cy]
            # New configuration as of 2026/01/26
            if isinstance(focal, (tuple, list)):
                fx, fy = focal
            else:
                fx = fy = focal # Assume isotropic camera
            
            # Get Principal Point (defaults to image center if not provided)
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
            # params: fx, fy, cx, cy (4 parameters)
            f.write(struct.pack('d', fx))
            f.write(struct.pack('d', fy))
            f.write(struct.pack('d', cx))
            f.write(struct.pack('d', cy))

    print(f"COLMAP cameras.bin saved to {output_file}")


def write_images_binary(cameras_dict, output_file):
    """Export images.bin"""
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

            # Encode image name with null terminator
            name_bytes = img_id.encode('utf-8') + b'\x00'
            f.write(name_bytes)
            f.write(struct.pack('Q', 0)) # Points2D count (set to 0)

    print(f"COLMAP images.bin saved to {output_file}")


def write_points3D_binary(pts3d, confidence, output_file):
    """Export points3D.bin"""
    num_points = len(pts3d)

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', num_points))

        for point_id, pt in enumerate(pts3d, start=1):
            x, y, z = pt

            f.write(struct.pack('Q', point_id))
            f.write(struct.pack('d', x))
            f.write(struct.pack('d', y))
            f.write(struct.pack('d', z))

            # RGB Color (Default: Gray)
            f.write(struct.pack('B', 128))
            f.write(struct.pack('B', 128))
            f.write(struct.pack('B', 128))

            # Error calculation based on confidence
            if confidence is not None and point_id <= len(confidence):
                error = 1.0 / max(confidence[point_id-1], 0.001)
            else:
                error = 1.0
            f.write(struct.pack('d', error))

            # track_length
            f.write(struct.pack('Q', 0))

    print(f"COLMAP points3D.bin saved to {output_file}")


def export_colmap_binary(cameras_dict, pts3d, confidence, image_size, output_dir):
    """Main function to export all COLMAP binary files"""
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
def extract_camera_params_process2(scene, image_paths, conf_threshold=1.5):
    """
    Extracts camera parameters and 3D points from the scene (FIXED: proper fx, fy handling).
    """
    print("\n=== Extracting Camera Parameters ===")

    cameras_dict = {}
    all_pts3d = []
    all_confidence = []

    try:
        # Attempt to get camera poses
        if hasattr(scene, 'get_im_poses'):
            poses = scene.get_im_poses()
        elif hasattr(scene, 'im_poses'):
            poses = scene.im_poses
        else:
            poses = None

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
        poses = None
        focals = None
        pps = None

    # [Important] MASt3R internal processing size
    mast3r_size = 224.0

    n_images = min(len(poses) if poses is not None else len(image_paths), len(image_paths))

    for idx in range(n_images):
        img_name = os.path.basename(image_paths[idx])

        try:
            # Get original image dimensions
            img = Image.open(image_paths[idx])
            W, H = img.size
            img.close()

            # Calculate scaling ratio
            scale = W / mast3r_size

            # Get Pose (Convert camera-to-world to world-to-camera)
            if poses is not None and idx < len(poses):
                pose_c2w = poses[idx]
                if isinstance(pose_c2w, torch.Tensor):
                    pose_c2w = pose_c2w.detach().cpu().numpy()
                if not isinstance(pose_c2w, np.ndarray) or pose_c2w.shape != (4, 4):
                    pose_c2w = np.eye(4)

                # Invert to get world-to-camera pose
                pose = np.linalg.inv(pose_c2w)
            else:
                pose = np.eye(4)

            # 🔧 FIX: Get and scale focal length (handle both isotropic and anisotropic)
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

                # 🔧 Apply scaling
                pp = pp_mast3r * scale
            else:
                pp = np.array([W / 2.0, H / 2.0])

            # 🔧 FIX: Store camera parameters with focal as tuple (fx, fy)
            cameras_dict[img_name] = {
                'focal': (fx, fy),  # ← FIXED: Store as tuple
                'pp': pp,
                'pose': pose,
                'rotation': pose[:3, :3],
                'translation': pose[:3, 3],
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

            # Extract 3D points
            if hasattr(scene, 'im_pts3d') and idx < len(scene.im_pts3d):
                pts3d_img = scene.im_pts3d[idx]
            elif hasattr(scene, 'get_pts3d'):
                pts3d_all = scene.get_pts3d()
                pts3d_img = pts3d_all[idx] if idx < len(pts3d_all) else None
            else:
                pts3d_img = None

            # Extract confidence scores
            if hasattr(scene, 'im_conf') and idx < len(scene.im_conf):
                conf_img = scene.im_conf[idx]
            elif hasattr(scene, 'get_conf'):
                conf_all = scene.get_conf()
                conf_img = conf_all[idx] if idx < len(conf_all) else None
            else:
                conf_img = None

            # Process 3D points and confidence
            if pts3d_img is not None:
                if isinstance(pts3d_img, torch.Tensor):
                    pts3d_img = pts3d_img.detach().cpu().numpy()

                pts3d_flat = pts3d_img.reshape(-1, 3) if pts3d_img.ndim == 3 else pts3d_img
                all_pts3d.append(pts3d_flat)

                if conf_img is not None:
                    if isinstance(conf_img, (list, torch.Tensor)):
                        conf_img = np.array(conf_img) if isinstance(conf_img, list) else conf_img.detach().cpu().numpy()

                    conf_flat = conf_img.reshape(-1) if conf_img.ndim > 1 else conf_img
                    
                    if len(conf_flat) != len(pts3d_flat):
                        conf_flat = np.ones(len(pts3d_flat))
                    
                    all_confidence.append(conf_flat)
                else:
                    all_confidence.append(np.ones(len(pts3d_flat)))

        except Exception as e:
            print(f"⚠️ Error processing image {idx} ({img_name}): {e}")
            # Fallback to default values with scaling applied
            img = Image.open(image_paths[idx])
            W, H = img.size
            img.close()

            cameras_dict[img_name] = {
                'focal': (1000.0 * (W / mast3r_size), 1000.0 * (W / mast3r_size)),  # ← FIXED: Tuple
                'pp': np.array([W / 2.0, H / 2.0]),
                'pose': np.eye(4),
                'rotation': np.eye(3),
                'translation': np.zeros(3),
                'width': W,
                'height': H
            }
            continue

    # Consolidate all 3D points
    if all_pts3d:
        pts3d = np.vstack(all_pts3d)
        confidence = np.concatenate(all_confidence)
    else:
        pts3d = np.zeros((0, 3))
        confidence = np.zeros(0)

    print(f"✓ Extracted parameters for {len(cameras_dict)} cameras")
    print(f"✓ Total 3D points: {len(pts3d)}")

    # Filter points by confidence
    if len(confidence) > 0:
        valid_mask = confidence > conf_threshold
        pts3d = pts3d[valid_mask]
        confidence = confidence[valid_mask]
        print(f"✓ Points after confidence filtering (>{conf_threshold}): {len(pts3d)}")

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


#------------------ additional scripts for ply -----------------------


import struct


def write_colmap_sparse(cameras_dict, pts3d, confidence, image_paths, output_dir):
    """
    Write COLMAP sparse reconstruction format
    """
    print(f"=== Writing COLMAP sparse reconstruction to {output_dir} ===")
    
    # Create a mapping from image names to integer IDs
    image_name_to_id = {}
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        image_name_to_id[img_name] = idx
    
    # Write cameras.txt
    cameras_file = os.path.join(output_dir, "cameras.txt")
    with open(cameras_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(cameras_dict)))
        
        for cam_name, cam_params in cameras_dict.items():
            # Get integer ID from image name
            cam_id = image_name_to_id.get(cam_name, len(image_name_to_id))
            
            # Get camera parameters
            if 'fx' in cam_params:
                fx, fy = cam_params['fx'], cam_params['fy']
                cx, cy = cam_params['cx'], cam_params['cy']
                width, height = cam_params['width'], cam_params['height']
            elif 'focal' in cam_params:
                focal = cam_params['focal']
                if isinstance(focal, (list, tuple, np.ndarray, torch.Tensor)):
                    if len(focal) == 1:
                        fx = fy = float(focal[0]) if isinstance(focal, (np.ndarray, torch.Tensor)) else focal[0]
                    else:
                        fx, fy = float(focal[0]), float(focal[1])
                else:
                    fx = fy = float(focal)
                
                if 'pp' in cam_params:
                    pp = cam_params['pp']
                    cx, cy = float(pp[0]), float(pp[1])
                else:
                    width = cam_params.get('width', 1024)
                    height = cam_params.get('height', 1024)
                    cx, cy = width / 2, height / 2
                
                width = cam_params.get('width', 1024)
                height = cam_params.get('height', 1024)
            else:
                print(f"Warning: Unknown camera parameter structure for camera {cam_name}")
                print(f"Available keys: {cam_params.keys()}")
                continue
            
            # COLMAP PINHOLE model: fx, fy, cx, cy
            f.write(f"{cam_id} PINHOLE {int(width)} {int(height)} "
                   f"{fx} {fy} {cx} {cy}\n")
    
    print(f"✓ Wrote {len(cameras_dict)} cameras to cameras.txt")
    
    # Write images.txt
    images_file = os.path.join(output_dir, "images.txt")
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}\n".format(len(cameras_dict)))
        
        for cam_name, cam_params in cameras_dict.items():
            # Get integer ID from image name
            cam_id = image_name_to_id.get(cam_name, len(image_name_to_id))
            
            # Get rotation as quaternion (w, x, y, z)
            if 'R' in cam_params:
                R = cam_params['R']
                quat = rotation_matrix_to_quaternion(R)
            else:
                # Identity rotation
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            
            # Get translation
            if 't' in cam_params:
                t = cam_params['t']
                if isinstance(t, torch.Tensor):
                    t = t.cpu().numpy()
                tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
            else:
                tx, ty, tz = 0.0, 0.0, 0.0
            
            # Write image line
            f.write(f"{cam_id} {quat[0]} {quat[1]} {quat[2]} {quat[3]} "
                   f"{tx} {ty} {tz} {cam_id} {cam_name}\n")
            
            # Write empty points2D line
            f.write("\n")
    
    print(f"✓ Wrote {len(cameras_dict)} images to images.txt")
    
    # Write points3D.txt
    points_file = os.path.join(output_dir, "points3D.txt")
    with open(points_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: {}\n".format(len(pts3d)))
        
        for i, (pt, conf) in enumerate(zip(pts3d, confidence)):
            # Use confidence as grayscale color (0-255)
            color_val = int(np.clip(conf * 50, 0, 255))
            
            # Write point
            f.write(f"{i} {pt[0]} {pt[1]} {pt[2]} {color_val} {color_val} {color_val} 0\n")
    
    print(f"✓ Wrote {len(pts3d)} points to points3D.txt")
    
    # Also write as PLY
    write_ply(pts3d, confidence, os.path.join(output_dir, "points3D.ply"))


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion (w, x, y, z)
    """
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()
    
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
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
    
    return np.array([w, x, y, z])


def write_ply(pts3d, confidence, output_path):
    """
    Write point cloud as PLY file
    """
    print(f"Writing PLY to {output_path}...")
    
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Data
        for pt, conf in zip(pts3d, confidence):
            color_val = int(np.clip(conf * 50, 0, 255))
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {color_val} {color_val} {color_val}\n")
    
    print(f"✓ Wrote PLY with {len(pts3d)} points")


import struct

def write_colmap_sparse_binary(cameras_dict, pts3d, confidence, image_paths, output_dir):
    """
    Write COLMAP sparse reconstruction in BINARY format (.bin files)
    """
    print(f"=== Writing COLMAP sparse reconstruction (BINARY) to {output_dir} ===")
    
    # Create a mapping from image names to integer IDs
    image_name_to_id = {}
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        image_name_to_id[img_name] = idx
    
    # Write cameras.bin
    cameras_file = os.path.join(output_dir, "cameras.bin")
    with open(cameras_file, 'wb') as f:
        # Write number of cameras
        f.write(struct.pack('Q', len(cameras_dict)))
        
        for cam_name, cam_params in cameras_dict.items():
            cam_id = image_name_to_id.get(cam_name, len(image_name_to_id))
            
            # Get camera parameters
            if 'focal' in cam_params:
                focal = cam_params['focal']
                if isinstance(focal, (list, tuple, np.ndarray, torch.Tensor)):
                    if len(focal) == 1:
                        fx = fy = float(focal[0]) if isinstance(focal, (np.ndarray, torch.Tensor)) else focal[0]
                    else:
                        fx, fy = float(focal[0]), float(focal[1])
                else:
                    fx = fy = float(focal)
                
                if 'pp' in cam_params:
                    pp = cam_params['pp']
                    cx, cy = float(pp[0]), float(pp[1])
                else:
                    width = cam_params.get('width', 1024)
                    height = cam_params.get('height', 1024)
                    cx, cy = width / 2, height / 2
                
                width = cam_params.get('width', 1024)
                height = cam_params.get('height', 1024)
            else:
                continue
            
            # Write camera data
            # camera_id (uint32), model_id (int32), width (uint64), height (uint64)
            f.write(struct.pack('I', cam_id))  # camera_id
            f.write(struct.pack('i', 1))  # model_id (1 = PINHOLE)
            f.write(struct.pack('Q', int(width)))  # width
            f.write(struct.pack('Q', int(height)))  # height
            
            # params: fx, fy, cx, cy (4 doubles)
            f.write(struct.pack('d', fx))
            f.write(struct.pack('d', fy))
            f.write(struct.pack('d', cx))
            f.write(struct.pack('d', cy))
    
    print(f"✓ Wrote {len(cameras_dict)} cameras to cameras.bin")
    
    # Write images.bin
    images_file = os.path.join(output_dir, "images.bin")
    with open(images_file, 'wb') as f:
        # Write number of images
        f.write(struct.pack('Q', len(cameras_dict)))
        
        for cam_name, cam_params in cameras_dict.items():
            cam_id = image_name_to_id.get(cam_name, len(image_name_to_id))
            
            # Get rotation as quaternion (w, x, y, z)
            if 'R' in cam_params:
                R = cam_params['R']
                quat = rotation_matrix_to_quaternion(R)
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            
            # Get translation
            if 't' in cam_params:
                t = cam_params['t']
                if isinstance(t, torch.Tensor):
                    t = t.cpu().numpy()
                tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
            else:
                tx, ty, tz = 0.0, 0.0, 0.0
            
            # Write image data
            f.write(struct.pack('I', cam_id))  # image_id
            f.write(struct.pack('d', quat[0]))  # qw
            f.write(struct.pack('d', quat[1]))  # qx
            f.write(struct.pack('d', quat[2]))  # qy
            f.write(struct.pack('d', quat[3]))  # qz
            f.write(struct.pack('d', tx))  # tx
            f.write(struct.pack('d', ty))  # ty
            f.write(struct.pack('d', tz))  # tz
            f.write(struct.pack('I', cam_id))  # camera_id
            
            # Write image name
            name_bytes = cam_name.encode('utf-8')
            f.write(name_bytes)
            f.write(b'\x00')  # null terminator
            
            # Write number of 2D points (0 for now)
            f.write(struct.pack('Q', 0))
    
    print(f"✓ Wrote {len(cameras_dict)} images to images.bin")
    
    # Write points3D.bin
    points_file = os.path.join(output_dir, "points3D.bin")
    with open(points_file, 'wb') as f:
        # Write number of points
        f.write(struct.pack('Q', len(pts3d)))
        
        for i, (pt, conf) in enumerate(zip(pts3d, confidence)):
            # Use confidence as grayscale color (0-255)
            color_val = int(np.clip(conf * 50, 0, 255))
            
            # Write point data
            f.write(struct.pack('Q', i))  # point3D_id
            f.write(struct.pack('d', float(pt[0])))  # X
            f.write(struct.pack('d', float(pt[1])))  # Y
            f.write(struct.pack('d', float(pt[2])))  # Z
            f.write(struct.pack('B', color_val))  # R
            f.write(struct.pack('B', color_val))  # G
            f.write(struct.pack('B', color_val))  # B
            f.write(struct.pack('d', 0.0))  # error
            
            # Write track length (0 for now)
            f.write(struct.pack('Q', 0))
    
    print(f"✓ Wrote {len(pts3d)} points to points3D.bin")
    
    # Also write as PLY with better colors
    write_colored_ply(pts3d, confidence, os.path.join(output_dir, "points3D.ply"))


def write_colored_ply(pts3d, confidence, output_path):
    """
    Write point cloud as PLY file with height-based coloring
    """
    print(f"Writing colored PLY to {output_path}...")
    
    # Normalize Z values for coloring
    z_values = pts3d[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    z_norm = (z_values - z_min) / (z_max - z_min + 1e-8)
    
    # Create colormap (blue -> cyan -> green -> yellow -> red)
    colors = np.zeros((len(pts3d), 3), dtype=np.uint8)
    colors[:, 0] = np.clip(255 * 2 * z_norm, 0, 255).astype(np.uint8)  # Red
    colors[:, 1] = np.clip(255 * 2 * (1 - np.abs(z_norm - 0.5)), 0, 255).astype(np.uint8)  # Green
    colors[:, 2] = np.clip(255 * 2 * (1 - z_norm), 0, 255).astype(np.uint8)  # Blue
    
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Data
        for pt, color in zip(pts3d, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {color[0]} {color[1]} {color[2]}\n")
    
    print(f"✓ Wrote colored PLY with {len(pts3d)} points")


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion (w, x, y, z)
    """
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()
    
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
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
    
    return np.array([w, x, y, z])
