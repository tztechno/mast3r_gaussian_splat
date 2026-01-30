# ============================================================================
# COLMAP Conversion (process3_06.py)
# ============================================================================

import numpy as np
import cv2
from pathlib import Path
import struct
from scipy.spatial.transform import Rotation

# ============================================================================
# COLMAP Conversion (process3_05.py) - FIXED VERSION
# ============================================================================

def convert_mast3r_to_colmap(
    scene,
    output_dir: str,
    min_conf_thr: float = 2.0,
    clean_depth: bool = False,
    mask_images: bool = True,
    verbose: bool = True
) -> str:
    """Converts a MASt3R scene to COLMAP format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sparse_dir = output_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    depth_dir = output_path / "stereo" / "depth_maps"
    depth_dir.mkdir(parents=True, exist_ok=True)

    normal_dir = output_path / "stereo" / "normal_maps"
    normal_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = None
    if mask_images:
        mask_dir = output_path / "stereo" / "confidence_maps"
        mask_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Converting MASt3R scene to COLMAP format...")
        print(f"Output directory: {output_dir}")

    cameras, images_data, points3D = extract_scene_data(scene, min_conf_thr, verbose)

    if verbose:
        print(f"Extracted {len(cameras)} cameras")
        print(f"Extracted {len(images_data)} images")
        print(f"Extracted {len(points3D)} 3D points")

    save_image_data(scene, images_dir, depth_dir, normal_dir, mask_dir, min_conf_thr, verbose)

    write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    write_images_binary(images_data, sparse_dir / "images.bin")
    write_points3d_binary(points3D, sparse_dir / "points3D.bin")

    if verbose:
        print(f"✓ COLMAP conversion completed")
        print(f"  Sparse model: {sparse_dir}")
        print(f"  Images: {images_dir}")
        print(f"  Depth maps: {depth_dir}")
        print(f"  Normal maps: {normal_dir}")

    return str(output_path)


def extract_scene_data(scene, min_conf_thr: float, verbose: bool):
    """Extracts cameras, images, and 3D points from a MASt3R scene."""
    cameras = {}
    images_data = {}

    num_images = len(scene.imgs)

    # Get scene data
    all_confidences = scene.get_conf()
    all_pts3d = scene.get_pts3d()
    
    # FIXED: Get camera poses from scene if available
    if hasattr(scene, 'im_poses') and scene.im_poses is not None:
        all_poses = scene.im_poses
    else:
        all_poses = [None] * num_images

    # FIXED: Get focals from scene if available
    if hasattr(scene, 'im_focals') and scene.im_focals is not None:
        all_focals = scene.im_focals
    else:
        all_focals = [None] * num_images

    for idx in range(num_images):
        img = scene.imgs[idx]
        h, w = img.shape[:2]

        camera_id = idx + 1  # FIXED: Each image gets its own camera

        # FIXED: Use actual focal length from scene
        if all_focals[idx] is not None:
            focal = float(all_focals[idx])
        else:
            focal = max(w, h) * 1.2
            
        cx = w / 2.0
        cy = h / 2.0

        cameras[camera_id] = {
            'id': camera_id,
            'model': 'PINHOLE',
            'width': w,
            'height': h,
            'params': np.array([focal, focal, cx, cy])
        }

        # FIXED: Use actual pose from scene
        if all_poses[idx] is not None:
            pose = all_poses[idx]
            if hasattr(pose, 'cpu'):
                pose = pose.detach().cpu().numpy()
            # MASt3R poses are world-to-camera, COLMAP needs camera-to-world
            pose_inv = np.linalg.inv(pose)
            qvec, tvec = matrix_to_quaternion_translation(pose_inv)
        else:
            # Fallback: estimate from 3D points
            pts3d = all_pts3d[idx]
            confidence = all_confidences[idx]
            pose = estimate_camera_pose(pts3d, confidence, min_conf_thr)
            qvec, tvec = matrix_to_quaternion_translation(pose)

        image_name = f"image_{idx:04d}.jpg"

        images_data[idx + 1] = {
            'id': idx + 1,
            'qvec': qvec,
            'tvec': tvec,
            'camera_id': camera_id,
            'name': image_name,
            'xys': np.array([]),  # Will be filled with 2D points
            'point3D_ids': np.array([])  # Will be filled with 3D point IDs
        }

    # Extract 3D points with proper correspondences
    points3D = extract_3d_points_with_correspondences(
        scene, images_data, min_conf_thr, verbose
    )

    return cameras, images_data, points3D


def estimate_camera_pose(pts3d: np.ndarray, confidence: np.ndarray, min_conf_thr: float) -> np.ndarray:
    """
    FIXED: Estimates camera pose from 3D points using PCA for orientation.
    """
    if hasattr(pts3d, 'cpu'):
        pts3d = pts3d.detach().cpu().numpy()
    if hasattr(confidence, 'cpu'):
        confidence = confidence.detach().cpu().numpy()

    h, w = pts3d.shape[:2]
    pts_flat = pts3d.reshape(-1, 3)
    conf_flat = confidence.reshape(-1)

    mask = conf_flat > min_conf_thr
    valid_pts = pts_flat[mask]

    if len(valid_pts) < 4:
        return np.eye(4)

    # Compute center
    center = np.median(valid_pts, axis=0)
    centered_pts = valid_pts - center

    # FIXED: Use PCA to estimate orientation
    try:
        cov = np.cov(centered_pts.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort eigenvectors by eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1
            
        R = eigenvectors
    except:
        R = np.eye(3)

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = -R @ center  # Camera-to-world transformation

    return pose


def matrix_to_quaternion_translation(matrix: np.ndarray):
    """
    FIXED: Robust conversion of 4x4 transformation matrix to quaternion and translation.
    """
    R = matrix[:3, :3]
    t = matrix[:3, 3]

    # FIXED: Use scipy for robust quaternion conversion
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    
    # COLMAP format is [w, x, y, z]
    qvec = np.array([quat[3], quat[0], quat[1], quat[2]])

    return qvec, t


def extract_3d_points_with_correspondences(scene, images_data, min_conf_thr: float, verbose: bool):
    """
    FIXED: Extracts 3D points with proper 2D-3D correspondences across images.
    """
    points3D = {}
    point_id = 1

    num_images = len(scene.imgs)
    all_confidences = scene.get_conf()
    all_pts3d = scene.get_pts3d()

    # Build a spatial hash for matching 3D points across views
    point_map = {}  # Maps quantized 3D coordinates to point IDs
    quantization = 0.01  # 1cm grid for matching

    def quantize_point(pt):
        """Quantize 3D point to grid cell."""
        return tuple((pt / quantization).astype(int))

    for idx in range(num_images):
        pts3d = all_pts3d[idx]
        confidence = all_confidences[idx]
        img = scene.imgs[idx]

        if hasattr(pts3d, 'cpu'):
            pts3d = pts3d.detach().cpu().numpy()
        if hasattr(confidence, 'cpu'):
            confidence = confidence.detach().cpu().numpy()
        if hasattr(img, 'cpu'):
            img = img.detach().cpu().numpy()

        h, w = pts3d.shape[:2]
        pts_flat = pts3d.reshape(-1, 3)
        conf_flat = confidence.reshape(-1)

        # Extract colors
        if len(img.shape) == 3:
            colors = img.reshape(-1, 3)
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = colors.astype(np.uint8)
        else:
            gray = img.reshape(-1)
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
            colors = np.stack([gray] * 3, axis=1)

        mask = conf_flat > min_conf_thr

        # FIXED: Limit points but maintain spatial distribution
        if mask.sum() > 10000:
            indices = np.where(mask)[0]
            # Sample uniformly across image
            step = len(indices) // 10000
            sampled_indices = indices[::step][:10000]
            mask = np.zeros_like(mask, dtype=bool)
            mask[sampled_indices] = True

        valid_pts = pts_flat[mask]
        valid_colors = colors[mask]
        valid_indices = np.where(mask)[0]

        # FIXED: Create 2D pixel coordinates
        y_coords, x_coords = np.unravel_index(valid_indices, (h, w))
        pixel_coords = np.stack([x_coords, y_coords], axis=1).astype(np.float64)

        # Lists to store correspondences for this image
        image_xys = []
        image_point3D_ids = []

        for i, (pt, color, xy) in enumerate(zip(valid_pts, valid_colors, pixel_coords)):
            q_pt = quantize_point(pt)
            
            if q_pt in point_map:
                # Point already exists - add correspondence
                pid = point_map[q_pt]
                points3D[pid]['image_ids'] = np.append(points3D[pid]['image_ids'], idx + 1)
                points3D[pid]['point2D_idxs'] = np.append(points3D[pid]['point2D_idxs'], len(image_xys))
            else:
                # New point
                point_map[q_pt] = point_id
                points3D[point_id] = {
                    'id': point_id,
                    'xyz': pt.astype(np.float64),
                    'rgb': color.astype(np.uint8),
                    'error': 0.0,
                    'image_ids': np.array([idx + 1], dtype=np.int32),
                    'point2D_idxs': np.array([len(image_xys)], dtype=np.int32)
                }
                pid = point_id
                point_id += 1
            
            image_xys.append(xy)
            image_point3D_ids.append(pid)

        # FIXED: Update image with 2D-3D correspondences
        images_data[idx + 1]['xys'] = np.array(image_xys, dtype=np.float64)
        images_data[idx + 1]['point3D_ids'] = np.array(image_point3D_ids, dtype=np.uint64)

    if verbose:
        print(f"Extracted {len(points3D)} unique 3D points with correspondences")
        multi_view = sum(1 for p in points3D.values() if len(p['image_ids']) > 1)
        print(f"  {multi_view} points visible in multiple views")

    return points3D


# Keep the rest of your functions (save_image_data, compute_normals_from_depth, etc.) as they are
# Just need to add this fix to save_depth_map:

def save_depth_map(depth: np.ndarray, path: Path):
    """
    FIXED: Saves depth map in COLMAP binary format with proper handling.
    """
    h, w = depth.shape

    # FIXED: Handle invalid depth values
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    
    with open(path, 'wb') as f:
        f.write(struct.pack('i', w))
        f.write(struct.pack('i', h))
        f.write(struct.pack('i', 1))  # Number of channels
        depth_flat = depth.astype(np.float32).flatten()
        f.write(depth_flat.tobytes())


# Keep all your write_*_binary functions as they look correct
