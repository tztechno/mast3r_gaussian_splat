# ============================================================================
# COLMAP Conversion (process3_07.py)
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



def extract_scene_data(scene, min_conf_thr, verbose):
    """Extract cameras, images, and 3D points from MASt3R scene"""
    cameras = {}
    images_data = {}
    points3D = []
    
    if verbose:
        print("Extracting scene data...")
        print(f"Scene type: {type(scene)}")
        print(f"Scene attributes: {dir(scene)}")
    
    # sceneの構造を確認
    if hasattr(scene, 'imgs'):
        num_views = len(scene.imgs)
    elif hasattr(scene, 'views'):
        num_views = len(scene.views)
    else:
        num_views = 0
        if verbose:
            print("Warning: Cannot determine number of views")
    
    if verbose:
        print(f"Number of views: {num_views}")
    
    # Extract camera parameters and poses
    for idx in range(num_views):
        if verbose:
            print(f"\n=== Processing view {idx} ===")
        
        # viewを取得
        if hasattr(scene, 'imgs'):
            view = scene.imgs[idx]
        elif hasattr(scene, 'views'):
            view = scene.views[idx]
        else:
            if verbose:
                print(f"Warning: Cannot access view {idx}")
            continue
        
        # 画像サイズを取得
        try:
            if hasattr(view, 'shape'):
                if isinstance(view.shape, (list, tuple)) and len(view.shape) >= 2:
                    height, width = int(view.shape[0]), int(view.shape[1])
                else:
                    height, width = 512, 512
            elif hasattr(view, 'img'):
                if isinstance(view.img, np.ndarray):
                    height, width = view.img.shape[:2]
                elif torch.is_tensor(view.img):
                    shape = view.img.shape
                    if len(shape) >= 2:
                        height, width = int(shape[-2]), int(shape[-1])
                    else:
                        height, width = 512, 512
                else:
                    height, width = 512, 512
            else:
                height, width = 512, 512
                
            if verbose:
                print(f"  Image size: {width}x{height}")
        except Exception as e:
            if verbose:
                print(f"  Error getting image size: {e}, using default 512x512")
            height, width = 512, 512
        
        # Camera intrinsics
        fx = fy = 500.0
        cx = width / 2.0
        cy = height / 2.0
        
        try:
            # カメラパラメータの取得を試みる
            if hasattr(scene, 'get_intrinsics'):
                K = scene.get_intrinsics()
                if K is not None:
                    if isinstance(K, torch.Tensor):
                        K = K.detach().cpu().numpy()
                    if K.ndim >= 2:
                        K_view = K[idx] if K.ndim == 3 else K
                        if K_view.shape[0] >= 3 and K_view.shape[1] >= 3:
                            fx = float(K_view[0, 0])
                            fy = float(K_view[1, 1])
                            cx = float(K_view[0, 2])
                            cy = float(K_view[1, 2])
                            if verbose:
                                print(f"  Extracted intrinsics from scene: fx={fx:.2f}, fy={fy:.2f}")
            
            # 個別のviewからカメラパラメータを取得
            if hasattr(view, 'camera') and view.camera is not None:
                cam = view.camera
                if isinstance(cam, torch.Tensor):
                    cam = cam.detach().cpu().numpy()
                
                if cam.ndim == 2 and cam.shape[0] >= 3 and cam.shape[1] >= 3:
                    fx = float(cam[0, 0])
                    fy = float(cam[1, 1])
                    cx = float(cam[0, 2])
                    cy = float(cam[1, 2])
                    if verbose:
                        print(f"  Extracted intrinsics from view.camera: fx={fx:.2f}, fy={fy:.2f}")
                        
        except Exception as e:
            if verbose:
                print(f"  Error extracting camera intrinsics: {e}")
                print(f"  Using defaults: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
        # カメラIDとパラメータを保存
        cam_id = idx
        cameras[cam_id] = {
            'model': 'PINHOLE',
            'width': int(width),
            'height': int(height),
            'params': [fx, fy, cx, cy]
        }
        
        # カメラポーズを抽出
        qvec = np.array([1.0, 0.0, 0.0, 0.0])  # デフォルト（回転なし）
        tvec = np.array([0.0, 0.0, 0.0])  # デフォルト（平行移動なし）
        
        try:
            pose = None
            
            # シーンレベルでポーズを取得
            if hasattr(scene, 'get_im_poses'):
                poses = scene.get_im_poses()
                if poses is not None:
                    if isinstance(poses, (list, tuple)):
                        if idx < len(poses):
                            pose = poses[idx]
                    else:
                        if isinstance(poses, torch.Tensor):
                            poses = poses.detach().cpu().numpy()
                        if poses.ndim >= 2:
                            pose = poses[idx] if poses.ndim == 3 else poses
            
            # ビューレベルでポーズを取得
            if pose is None and hasattr(view, 'pose') and view.pose is not None:
                pose = view.pose
            
            # ポーズがTensorの場合はnumpy配列に変換
            if pose is not None and isinstance(pose, torch.Tensor):
                pose = pose.detach().cpu().numpy()
            
            # ポーズの処理
            if pose is not None:
                if verbose:
                    print(f"  Pose type: {type(pose)}")
                    print(f"  Pose shape: {pose.shape if hasattr(pose, 'shape') else 'N/A'}")
                
                # ポーズの次元をチェック
                if isinstance(pose, np.ndarray):
                    if pose.ndim == 1:
                        if verbose:
                            print(f"  Warning: 1D pose array (shape {pose.shape}), using identity pose")
                        # デフォルト値を維持
                        
                    elif pose.ndim == 2:
                        if pose.shape == (4, 4):
                            # 正しい4x4行列
                            if verbose:
                                print(f"  Processing 4x4 pose matrix")
                            try:
                                # 行列が特異でないかチェック
                                det = np.linalg.det(pose)
                                if abs(det) < 1e-10:
                                    if verbose:
                                        print(f"  Warning: Near-singular matrix (det={det}), using identity pose")
                                else:
                                    # MASt3Rのポーズはworld-to-camera、COLMAPはcamera-to-worldが必要
                                    pose_inv = np.linalg.inv(pose)
                                    qvec, tvec = matrix_to_quaternion_translation(pose_inv)
                                    if verbose:
                                        print(f"  Successfully extracted pose")
                            except np.linalg.LinAlgError as e:
                                if verbose:
                                    print(f"  LinAlgError: {e}, using identity pose")
                                    
                        elif pose.shape == (3, 4):
                            # 3x4行列を4x4に拡張
                            if verbose:
                                print(f"  Processing 3x4 pose matrix")
                            pose_4x4 = np.eye(4)
                            pose_4x4[:3, :] = pose
                            try:
                                det = np.linalg.det(pose_4x4)
                                if abs(det) < 1e-10:
                                    if verbose:
                                        print(f"  Warning: Near-singular matrix (det={det}), using identity pose")
                                else:
                                    pose_inv = np.linalg.inv(pose_4x4)
                                    qvec, tvec = matrix_to_quaternion_translation(pose_inv)
                                    if verbose:
                                        print(f"  Successfully extracted pose from 3x4 matrix")
                            except np.linalg.LinAlgError as e:
                                if verbose:
                                    print(f"  LinAlgError: {e}, using identity pose")
                                    
                        elif pose.shape == (3, 3):
                            # 3x3回転行列のみ
                            if verbose:
                                print(f"  Processing 3x3 rotation matrix")
                            pose_4x4 = np.eye(4)
                            pose_4x4[:3, :3] = pose
                            try:
                                pose_inv = np.linalg.inv(pose_4x4)
                                qvec, tvec = matrix_to_quaternion_translation(pose_inv)
                                if verbose:
                                    print(f"  Successfully extracted pose from 3x3 matrix")
                            except np.linalg.LinAlgError as e:
                                if verbose:
                                    print(f"  LinAlgError: {e}, using identity pose")
                        else:
                            if verbose:
                                print(f"  Warning: Unexpected 2D pose shape {pose.shape}, using identity pose")
                    else:
                        if verbose:
                            print(f"  Warning: Unexpected pose dimensions (ndim={pose.ndim}), using identity pose")
                else:
                    if verbose:
                        print(f"  Warning: Pose is not a numpy array, using identity pose")
            else:
                if verbose:
                    print(f"  No pose found, using identity pose")
                    
        except Exception as e:
            if verbose:
                print(f"  Error extracting pose: {e}")
                print(f"  Using identity pose")
            import traceback
            traceback.print_exc()
        
        # 画像データを保存
        img_id = idx + 1
        images_data[img_id] = {
            'qvec': qvec,
            'tvec': tvec,
            'camera_id': cam_id,
            'name': f'image_{idx:04d}.jpg',
            'xys': np.array([]),  # 空の2D点配列
            'point3D_ids': np.array([])  # 空の3D点ID配列
        }
        
        if verbose:
            print(f"  Final - Camera {cam_id}: {width}x{height}")
            print(f"  Final - Image {img_id}: qvec={qvec[:4]}, tvec={tvec[:3]}")


    # 3D点を抽出（extract_scene_data関数内の該当部分を置き換え）
    if verbose:
        print("\n=== Extracting 3D points ===")
    
    try:
        # MASt3Rシーンから3D点を取得
        if hasattr(scene, 'get_pts3d'):
            pts3d = scene.get_pts3d()
            if pts3d is not None:
                if verbose:
                    print(f"  pts3d type: {type(pts3d)}")
                
                # リストの場合の処理
                if isinstance(pts3d, list):
                    if verbose:
                        print(f"  pts3d is a list with {len(pts3d)} elements")
                    
                    # リストの各要素を処理
                    all_points = []
                    for i, pts in enumerate(pts3d):
                        if isinstance(pts, torch.Tensor):
                            pts = pts.detach().cpu().numpy()
                        if isinstance(pts, np.ndarray):
                            all_points.append(pts.reshape(-1, 3))
                            if verbose and i < 3:  # 最初の3つだけ表示
                                print(f"    Element {i} shape: {pts.shape}")
                    
                    if all_points:
                        pts3d_combined = np.vstack(all_points)
                        if verbose:
                            print(f"  Combined pts3d shape: {pts3d_combined.shape}")
                    else:
                        pts3d_combined = None
                        
                # Tensorまたはnumpy配列の場合
                elif isinstance(pts3d, torch.Tensor):
                    pts3d_combined = pts3d.detach().cpu().numpy()
                    if verbose:
                        print(f"  pts3d shape (from tensor): {pts3d_combined.shape}")
                elif isinstance(pts3d, np.ndarray):
                    pts3d_combined = pts3d
                    if verbose:
                        print(f"  pts3d shape (numpy): {pts3d_combined.shape}")
                else:
                    pts3d_combined = None
                    if verbose:
                        print(f"  Unexpected pts3d type: {type(pts3d)}")
                
                # 信頼度フィルタリング
                if pts3d_combined is not None:
                    # 信頼度を取得
                    conf = None
                    if hasattr(scene, 'get_conf'):
                        conf = scene.get_conf()
                    elif hasattr(scene, 'im_conf'):
                        conf = scene.im_conf
                    
                    if conf is not None:
                        if verbose:
                            print(f"  conf type: {type(conf)}")
                        
                        # confもリストの可能性
                        if isinstance(conf, list):
                            all_conf = []
                            for c in conf:
                                if isinstance(c, torch.Tensor):
                                    c = c.detach().cpu().numpy()
                                if isinstance(c, np.ndarray):
                                    all_conf.append(c.flatten())
                            if all_conf:
                                conf_combined = np.concatenate(all_conf)
                            else:
                                conf_combined = None
                        elif isinstance(conf, torch.Tensor):
                            conf_combined = conf.detach().cpu().numpy().flatten()
                        elif isinstance(conf, np.ndarray):
                            conf_combined = conf.flatten()
                        else:
                            conf_combined = None
                        
                        if conf_combined is not None:
                            if verbose:
                                print(f"  conf shape: {conf_combined.shape}")
                            
                            # 点群を平坦化
                            pts3d_flat = pts3d_combined.reshape(-1, 3)
                            
                            # サイズを合わせる
                            min_size = min(len(pts3d_flat), len(conf_combined))
                            pts3d_flat = pts3d_flat[:min_size]
                            conf_combined = conf_combined[:min_size]
                            
                            # フィルタリング
                            mask = conf_combined >= min_conf_thr
                            pts3d_filtered = pts3d_flat[mask]
                            
                            if verbose:
                                print(f"  Points before filtering: {len(pts3d_flat)}")
                                print(f"  Points after filtering (conf >= {min_conf_thr}): {len(pts3d_filtered)}")
                        else:
                            pts3d_filtered = pts3d_combined.reshape(-1, 3)
                            if verbose:
                                print(f"  No valid confidence, using all {len(pts3d_filtered)} points")
                    else:
                        # 信頼度がない場合は全ての点を使用
                        pts3d_filtered = pts3d_combined.reshape(-1, 3)
                        if verbose:
                            print(f"  No confidence values, using all {len(pts3d_filtered)} points")
                    
                    # COLMAP形式に変換
                    for i, pt in enumerate(pts3d_filtered):
                        # 無効な点をスキップ（NaNやInf）
                        if not np.all(np.isfinite(pt)):
                            continue
                        
                        points3D.append({
                            'xyz': pt,
                            'rgb': np.array([128, 128, 128]),  # デフォルトグレー
                            'error': 0.0,
                            'image_ids': np.array([]),
                            'point2D_idxs': np.array([])
                        })
                else:
                    if verbose:
                        print("  No valid pts3d data")
        else:
            if verbose:
                print("  Warning: Scene has no get_pts3d method")
                
    except Exception as e:
        if verbose:
            print(f"  Error extracting 3D points: {e}")
        import traceback
        traceback.print_exc()

    
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

def save_image_data(scene, images_dir, depth_dir, normal_dir, mask_dir, min_conf_thr, verbose):
    """Save RGB images, depth maps, normal maps, and masks"""
    if verbose:
        print("\nSaving image data...")
    
    # ディレクトリが存在することを確認（既に作成済みのはず）
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # ビュー数を取得
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
    
    for idx in range(num_views):
        try:
            # RGB画像を保存
            img_path = images_dir / f'image_{idx:04d}.jpg'
            
            # 画像データを取得
            if hasattr(imgs[idx], 'img'):
                img = imgs[idx].img
            elif hasattr(imgs[idx], 'image'):
                img = imgs[idx].image
            else:
                img = imgs[idx]
            
            # Tensorの場合はnumpy配列に変換
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            
            # 画像を正しい形式に変換
            if isinstance(img, np.ndarray):
                # (C, H, W) -> (H, W, C)に変換
                if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                    img = np.transpose(img, (1, 2, 0))
                
                # 値の範囲を[0, 255]に正規化
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                
                # グレースケールの場合はRGBに変換
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                elif img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                
                # 画像を保存
                from PIL import Image
                Image.fromarray(img).save(img_path)
                
                if verbose and idx < 3:
                    print(f"  Saved image {idx}: {img_path}")
            
            # デプスマップを保存（もし利用可能なら）
            try:
                if hasattr(scene, 'get_depthmaps'):
                    depthmaps = scene.get_depthmaps()
                    if depthmaps is not None and idx < len(depthmaps):
                        depth = depthmaps[idx]
                        if isinstance(depth, torch.Tensor):
                            depth = depth.detach().cpu().numpy()
                        
                        if isinstance(depth, np.ndarray):
                            depth_path = depth_dir / f'depth_{idx:04d}.npy'
                            np.save(depth_path, depth)
                            
                            if verbose and idx < 3:
                                print(f"  Saved depth {idx}: {depth_path}")
            except Exception as e:
                if verbose and idx == 0:
                    print(f"  Note: Could not save depth maps: {e}")
            
            # マスクを保存（もし利用可能なら）
            try:
                if hasattr(scene, 'get_masks'):
                    masks = scene.get_masks()
                    if masks is not None and idx < len(masks):
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
                if verbose and idx == 0:
                    print(f"  Note: Could not save masks: {e}")
                    
        except Exception as e:
            if verbose:
                print(f"  Error saving data for view {idx}: {e}")
    
    if verbose:
        print(f"  Saved {num_views} images")
