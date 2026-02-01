# process2_14.py
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
    """Extract camera parameters and 3D points from DUSt3R scene"""
    
    cameras_dict = {}
    all_pts3d = []
    all_confidence = []
    all_pts3d_raw = []
    all_conf_raw = []
    
    print("=== Extracting 3D Points ===")
    
    # Get all data at once using the correct methods
    pts3d_list = scene.get_pts3d()  # List of 3D points for each image
    im_poses = scene.get_im_poses()  # Camera poses
    focals = scene.get_focals()  # Focal lengths
    principal_points = scene.get_principal_points()  # Principal points
    
    # Get confidence maps - use im_conf attribute directly
    confidence_maps = []
    for idx in range(len(image_paths)):
        conf = scene.im_conf[idx]
        
        # Handle ParameterList or Parameter
        if isinstance(conf, torch.nn.ParameterList):
            conf = conf[0]
        
        if torch.is_tensor(conf):
            conf = conf.detach().cpu().numpy()
        
        confidence_maps.append(conf)
    
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        
        # Get intrinsics
        focal = focals[idx]
        if torch.is_tensor(focal):
            focal = focal.detach().cpu().item()
        
        principal_point = principal_points[idx]
        if torch.is_tensor(principal_point):
            principal_point = principal_point.detach().cpu().numpy()
        
        # Get pose
        pose = im_poses[idx]
        if torch.is_tensor(pose):
            pose = pose.detach().cpu().numpy()
        
        cameras_dict[img_name] = {
            'focal': focal,
            'pp': principal_point,
            'pose': pose
        }
        
        # Get 3D points
        pts = pts3d_list[idx]
        if torch.is_tensor(pts):
            pts = pts.detach().cpu().numpy()
        
        # Get confidence
        conf = confidence_maps[idx]
        
        # Reshape
        H, W = pts.shape[:2]
        pts_flat = pts.reshape(-1, 3)
        conf_flat = conf.reshape(-1)
        
        # Store raw data
        all_pts3d_raw.append(pts_flat)
        all_conf_raw.append(conf_flat)
        
        # Filter by confidence
        mask = conf_flat > conf_threshold
        pts_filtered = pts_flat[mask]
        conf_filtered = conf_flat[mask]
        
        all_pts3d.append(pts_filtered)
        all_confidence.append(conf_filtered)
        
        print(f"  Image {idx+1}/{len(image_paths)}: {img_name}")
        print(f"    Points: {len(pts_filtered)}/{len(pts_flat)} (confidence > {conf_threshold})")
    
    # Concatenate all points
    pts3d = np.concatenate(all_pts3d, axis=0)
    confidence = np.concatenate(all_confidence, axis=0)
    pts3d_raw = np.concatenate(all_pts3d_raw, axis=0)
    conf_raw = np.concatenate(all_conf_raw, axis=0)
    
    print(f"\nTotal 3D points: {len(pts3d)}")
    print(f"Confidence range: [{confidence.min():.2f}, {confidence.max():.2f}]")
    
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


def write_binary(f, value, fmt):
    """Write binary value to file"""
    import struct
    f.write(struct.pack(fmt, value))


def write_string(f, s):
    """Write null-terminated string"""
    f.write(s.encode('utf-8') + b'\x00')


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion (w, x, y, z)"""
    import numpy as np
    
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
    
    return np.array([w, x, y, z])




import os
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R

def export_colmap_binary_simple(cameras_dict, pts3d, confidence, colors, image_paths, output_dir):
    """Exports sparse COLMAP data to binary files (Simple version)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get camera parameters from the first image
    first_key = list(cameras_dict.keys())[0]
    first_cam = cameras_dict[first_key]
    
    w = 1024  # image_size
    h = 1024
    focal = float(first_cam.get('focal', 1228.8))
    cx = w / 2.0
    cy = h / 2.0
    
    print("\n=== Exporting to COLMAP Binary Format (Simple) ===")
    
    # 1. cameras.bin - Only one camera defined
    cameras_file = os.path.join(output_dir, 'cameras.bin')
    with open(cameras_file, 'wb') as f:
        f.write(struct.pack('Q', 1))  # Number of cameras: 1
        f.write(struct.pack('I', 1))  # Camera ID: 1
        f.write(struct.pack('i', 1))  # Model ID: 1 (SIMPLE_PINHOLE)
        f.write(struct.pack('Q', w))
        f.write(struct.pack('Q', h))
        f.write(struct.pack('d', focal))
        f.write(struct.pack('d', cx))
        f.write(struct.pack('d', cy))
    
    print(f"✓ cameras.bin: 1 camera")
    
    # 2. images.bin
    images_file = os.path.join(output_dir, 'images.bin')
    with open(images_file, 'wb') as f:
        f.write(struct.pack('Q', len(image_paths)))
        
        for i, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            cam_info = cameras_dict.get(img_name, first_cam)
            pose = cam_info['pose']
            
            # Rotation matrix to quaternion
            rotation_mat = pose[:3, :3]
            # Assumes a helper function 'rotation_matrix_to_quaternion' exists
            quat = rotation_matrix_to_quaternion(rotation_mat)
            
            # Translation
            tvec = pose[:3, 3]
            
            image_id = i + 1
            f.write(struct.pack('I', image_id))
            
            # Quaternion (w, x, y, z)
            for q in quat:
                f.write(struct.pack('d', q))
            
            # Translation
            for val in tvec:
                f.write(struct.pack('d', val))
            
            # Camera ID - Use 1 for all images
            f.write(struct.pack('I', 1))
            
            # Image name (requires a helper or logic to write null-terminated string)
            write_string(f, img_name)
            
            # Number of 2D points (0 for sparse placeholder)
            f.write(struct.pack('Q', 0))
    
    print(f"✓ images.bin: {len(image_paths)} images")
    
    # 3. points3D.bin
    points_file = os.path.join(output_dir, 'points3D.bin')
    with open(points_file, 'wb') as f:
        f.write(struct.pack('Q', len(pts3d)))
        
        for point_id, (pt, conf, color) in enumerate(zip(pts3d, confidence, colors), 1):
            f.write(struct.pack('Q', point_id))
            
            # XYZ coordinates
            for coord in pt:
                f.write(struct.pack('d', coord))
            
            # RGB color
            for c in color:
                f.write(struct.pack('B', int(c)))
            
            # Error (reprojection error estimate)
            error = 1.0 / max(conf, 0.01)
            f.write(struct.pack('d', error))
            
            # Track (visibility list length 0)
            f.write(struct.pack('Q', 0))
    
    print(f"✓ points3D.bin: {len(pts3d)} points")
    print(f"\n✅ Export complete: {output_dir}")


def write_colmap_sparse(cameras_dict, pts3d, confidence, image_paths, output_dir):
    """Exports sparse COLMAP data to binary files (Original logic)"""
    os.makedirs(output_dir, exist_ok=True)
    
    if not cameras_dict:
        raise ValueError("cameras_dict is empty")
    
    first_key = list(cameras_dict.keys())[0]
    first_cam = cameras_dict[first_key]
    
    w = int(first_cam.get('width', 1920))
    h = int(first_cam.get('height', 1080))
    focal = float(first_cam.get('focal', max(w, h) * 1.2))
    cx = w / 2.0
    cy = h / 2.0
    
    # cameras.bin
    cameras_file = os.path.join(output_dir, 'cameras.bin')
    with open(cameras_file, 'wb') as f:
        f.write(struct.pack('Q', 1))  # Num cameras
        camera_id = 1
        model_id = 1  # PINHOLE
        f.write(struct.pack('i', camera_id))
        f.write(struct.pack('i', model_id))
        f.write(struct.pack('Q', w))
        f.write(struct.pack('Q', h))
        f.write(struct.pack('d', focal)) # fx
        f.write(struct.pack('d', focal)) # fy
        f.write(struct.pack('d', cx))
        f.write(struct.pack('d', cy))
    
    print(f"✓ Written cameras.bin")
    
    # images.bin
    images_file = os.path.join(output_dir, 'images.bin')
    with open(images_file, 'wb') as f:
        f.write(struct.pack('Q', len(image_paths)))
        
        for i, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            
            cam_info = cameras_dict.get(img_name)
            if cam_info is None:
                pose = np.eye(4)
            else:
                pose = cam_info['pose']
            
            # Convert Camera-to-World (c2w) to World-to-Camera (w2c)
            try:
                w2c = np.linalg.inv(pose)
            except np.linalg.LinAlgError:
                w2c = np.eye(4)
            
            rot_mat = w2c[:3, :3]
            tvec = w2c[:3, 3]
            
            # Scipy returns quaternions as (x, y, z, w)
            quat = R.from_matrix(rot_mat).as_quat()
            # COLMAP expects (qw, qx, qy, qz)
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            
            image_id = i + 1
            f.write(struct.pack('i', image_id))
            f.write(struct.pack('d', qw))
            f.write(struct.pack('d', qx))
            f.write(struct.pack('d', qy))
            f.write(struct.pack('d', qz))
            f.write(struct.pack('d', tvec[0]))
            f.write(struct.pack('d', tvec[1]))
            f.write(struct.pack('d', tvec[2]))
            f.write(struct.pack('i', 1))  # Camera ID
            
            # Null-terminated string for image name
            img_name_bytes = img_name.encode('utf-8') + b'\x00'
            f.write(img_name_bytes)
            f.write(struct.pack('Q', 0))  # No 2D points
    
    print(f"✓ Written images.bin ({len(image_paths)} images)")
    
    # points3D.bin
    points_file = os.path.join(output_dir, 'points3D.bin')
    with open(points_file, 'wb') as f:
        f.write(struct.pack('Q', len(pts3d)))
        
        for point_id, point in enumerate(pts3d, start=1):
            f.write(struct.pack('Q', point_id))
            f.write(struct.pack('d', point[0]))
            f.write(struct.pack('d', point[1]))
            f.write(struct.pack('d', point[2]))
            f.write(struct.pack('B', 255)) # R
            f.write(struct.pack('B', 255)) # G
            f.write(struct.pack('B', 255)) # B
            f.write(struct.pack('d', 0.0)) # Error
            f.write(struct.pack('Q', 0))   # Track
    
    print(f"✓ Written points3D.bin ({len(pts3d)} points)")
    print(f"\n✓ COLMAP sparse reconstruction saved")
    return output_dir

