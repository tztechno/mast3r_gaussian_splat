"""
process3_01.py
"""

import numpy as np
import os
from pathlib import Path
import struct
import cv2
from typing import Dict, List, Tuple, Optional


# =============================================================================
# 1. メイン変換関数
# =============================================================================

def convert_mast3r_to_colmap(
    scene,
    output_dir: str,
    min_conf_thr: float = 2.0,
    clean_depth: bool = False,
    mask_images: bool = True,
    verbose: bool = True
) -> str:
    """
    MASt3RシーンをCOLMAPフォーマットに変換（メイン関数）
    
    Args:
        scene: MASt3Rのシーンオブジェクト
        output_dir: 出力ディレクトリ
        min_conf_thr: 最小信頼度閾値
        clean_depth: デプスマップをクリーンにするか
        mask_images: 信頼度マスクを保存するか
        verbose: 詳細出力
        
    Returns:
        COLMAPディレクトリのパス
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリ作成
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
    
    # カメラ・画像・3D点を抽出
    cameras, images_data, points3D = extract_scene_data(scene, min_conf_thr, verbose)
    
    if verbose:
        print(f"Extracted {len(cameras)} cameras")
        print(f"Extracted {len(images_data)} images")
        print(f"Extracted {len(points3D)} 3D points")
    
    # 画像とデプス/ノーマルマップを保存
    save_image_data(scene, images_dir, depth_dir, normal_dir, mask_dir, min_conf_thr, verbose)
    
    # COLMAPバイナリファイルを書き込み
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


# =============================================================================
# 2. シーンからのデータ抽出
# =============================================================================
def extract_scene_data(scene, min_conf_thr: float, verbose: bool) -> Tuple[Dict, Dict, Dict]:
    """
    MASt3Rシーンからカメラ、画像、3D点を抽出
    """
    cameras = {}
    images_data = {}
    
    num_images = len(scene.imgs)
    
    # 全画像の信頼度を一度に取得
    all_confidences = scene.get_conf()  # インデックスなし
    
    # 各画像に対してカメラパラメータとポーズを抽出
    for idx in range(num_images):
        img = scene.imgs[idx]
        h, w = img.shape[:2]
        
        # カメラ内部パラメータ
        camera_id = 1
        
        if camera_id not in cameras:
            focal_length = max(w, h) * 1.2
            cx = w / 2.0
            cy = h / 2.0
            
            cameras[camera_id] = {
                'id': camera_id,
                'model': 'PINHOLE',
                'width': w,
                'height': h,
                'params': np.array([focal_length, focal_length, cx, cy])
            }
        
        # カメラポーズ推定
        pts3d = scene.get_pts3d(idx)
        confidence = all_confidences[idx]  # リストから取得
        
        pose = estimate_camera_pose(pts3d, confidence, min_conf_thr)
        qvec, tvec = matrix_to_quaternion_translation(pose)
        
        image_name = f"image_{idx:04d}.jpg"
        
        images_data[idx + 1] = {
            'id': idx + 1,
            'qvec': qvec,
            'tvec': tvec,
            'camera_id': camera_id,
            'name': image_name,
            'xys': np.array([]),
            'point3D_ids': np.array([])
        }
    
    # 3D点を抽出
    points3D = extract_3d_points(scene, min_conf_thr, verbose)
    
    return cameras, images_data, points3D


def estimate_camera_pose(pts3d: np.ndarray, confidence: np.ndarray, min_conf_thr: float) -> np.ndarray:
    """
    3D点からカメラポーズを推定
    
    Args:
        pts3d: 3D点の配列 [H, W, 3]
        confidence: 信頼度マップ [H, W]
        min_conf_thr: 最小信頼度閾値
        
    Returns:
        4x4変換行列
    """
    # テンソルをNumPy配列に変換
    if hasattr(pts3d, 'cpu'):  # PyTorchテンソルの場合
        pts3d = pts3d.cpu().numpy()
    if hasattr(confidence, 'cpu'):  # PyTorchテンソルの場合
        confidence = confidence.cpu().numpy()
    
    mask = confidence > min_conf_thr
    valid_pts = pts3d[mask]
    
    if len(valid_pts) < 4:
        return np.eye(4)
    
    center = np.median(valid_pts, axis=0)
    pose = np.eye(4)
    pose[:3, 3] = -center
    
    return pose



def matrix_to_quaternion_translation(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    4x4変換行列をクォータニオンと並進ベクトルに変換
    
    Args:
        matrix: 4x4変換行列
        
    Returns:
        (qvec, tvec) - クォータニオン[w,x,y,z]と並進ベクトル[x,y,z]
    """
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    
    # 回転行列をクォータニオンに変換
    qw = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
    
    qvec = np.array([qw, qx, qy, qz])
    
    return qvec, t


def extract_3d_points(scene, min_conf_thr: float, verbose: bool) -> Dict:
    """シーンから3D点を抽出"""
    points3D = {}
    point_id = 1
    
    num_images = len(scene.imgs)
    all_confidences = scene.get_conf()
    
    for idx in range(num_images):
        pts3d = scene.get_pts3d(idx)
        confidence = all_confidences[idx]
        img = scene.imgs[idx]
        
        # テンソルをNumPy配列に変換
        if hasattr(pts3d, 'cpu'):
            pts3d = pts3d.cpu().numpy()
        if hasattr(confidence, 'cpu'):
            confidence = confidence.cpu().numpy()
        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()
        
        h, w = pts3d.shape[:2]
        pts_flat = pts3d.reshape(-1, 3)
        conf_flat = confidence.reshape(-1)
        
        # 色情報
        if len(img.shape) == 3:
            colors = img.reshape(-1, 3)
        else:
            colors = np.stack([img.reshape(-1)] * 3, axis=1)
        
        # 信頼度でフィルタ
        mask = conf_flat > min_conf_thr
        
        # サンプリング（点が多すぎる場合）
        if mask.sum() > 10000:
            indices = np.where(mask)[0]
            sampled_indices = np.random.choice(indices, size=10000, replace=False)
            mask = np.zeros_like(mask, dtype=bool)
            mask[sampled_indices] = True
        
        valid_pts = pts_flat[mask]
        valid_colors = colors[mask]
        
        for pt, color in zip(valid_pts, valid_colors):
            points3D[point_id] = {
                'id': point_id,
                'xyz': pt,
                'rgb': color.astype(np.uint8),
                'error': 0.0,
                'image_ids': np.array([idx + 1]),
                'point2D_idxs': np.array([0])
            }
            point_id += 1
    
    if verbose:
        print(f"Extracted {len(points3D)} 3D points")
    
    return points3D


def save_image_data(
    scene,
    images_dir: Path,
    depth_dir: Path,
    normal_dir: Path,
    mask_dir: Optional[Path],
    min_conf_thr: float,
    verbose: bool
):
    """画像、デプスマップ、ノーマルマップ、信頼度マスクを保存"""
    num_images = len(scene.imgs)
    all_confidences = scene.get_conf()
    
    for idx in range(num_images):
        image_name = f"image_{idx:04d}.jpg"
        
        # 画像を保存
        img = scene.imgs[idx]
        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()
        
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        cv2.imwrite(str(images_dir / image_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # デプスマップを保存
        pts3d = scene.get_pts3d(idx)
        if hasattr(pts3d, 'cpu'):
            pts3d = pts3d.cpu().numpy()
        
        depth = np.linalg.norm(pts3d, axis=2)
        depth_name = image_name.replace('.jpg', '.geometric.bin')
        save_depth_map(depth, depth_dir / depth_name)
        
        # ノーマルマップを保存
        normals = compute_normals_from_depth(pts3d)
        normal_name = image_name.replace('.jpg', '.geometric.bin')
        save_normal_map(normals, normal_dir / normal_name)
        
        # 信頼度マスクを保存
        if mask_dir is not None:
            confidence = all_confidences[idx]
            if hasattr(confidence, 'cpu'):
                confidence = confidence.cpu().numpy()
            
            mask = (confidence > min_conf_thr).astype(np.uint8) * 255
            mask_name = image_name.replace('.jpg', '.png')
            cv2.imwrite(str(mask_dir / mask_name), mask)
    
    if verbose:
        print(f"Saved {num_images} images with depth/normal maps")



def compute_normals_from_depth(pts3d: np.ndarray) -> np.ndarray:
    """
    3D点から表面法線を計算
    
    Args:
        pts3d: 3D点の配列 [H, W, 3]
        
    Returns:
        法線ベクトルの配列 [H, W, 3]
    """
    h, w = pts3d.shape[:2]
    normals = np.zeros_like(pts3d)
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            px = pts3d[i, j + 1] - pts3d[i, j - 1]
            py = pts3d[i + 1, j] - pts3d[i - 1, j]
            normal = np.cross(px, py)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normals[i, j] = normal / norm
    
    return normals


def save_depth_map(depth: np.ndarray, path: Path):
    """
    デプスマップをCOLMAPバイナリ形式で保存
    
    Args:
        depth: デプスマップ [H, W]
        path: 保存先パス
    """
    h, w = depth.shape
    
    with open(path, 'wb') as f:
        f.write(struct.pack('i', w))
        f.write(struct.pack('i', h))
        f.write(struct.pack('i', 1))  # チャンネル数
        depth_flat = depth.astype(np.float32).flatten()
        f.write(depth_flat.tobytes())


def save_normal_map(normals: np.ndarray, path: Path):
    """
    ノーマルマップをCOLMAPバイナリ形式で保存
    
    Args:
        normals: ノーマルマップ [H, W, 3]
        path: 保存先パス
    """
    h, w = normals.shape[:2]
    
    with open(path, 'wb') as f:
        f.write(struct.pack('i', w))
        f.write(struct.pack('i', h))
        f.write(struct.pack('i', 3))  # チャンネル数
        normals_flat = normals.astype(np.float32).reshape(-1)
        f.write(normals_flat.tobytes())


# =============================================================================
# 4. COLMAPバイナリファイルの書き込み
# =============================================================================

def write_cameras_binary(cameras: Dict, path: Path):
    """
    cameras.binをCOLMAPバイナリ形式で書き込み
    
    Args:
        cameras: カメラ辞書
        path: 保存先パス
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('Q', len(cameras)))
        
        for camera in cameras.values():
            f.write(struct.pack('i', camera['id']))
            f.write(struct.pack('i', 1))  # PINHOLE = 1
            f.write(struct.pack('Q', camera['width']))
            f.write(struct.pack('Q', camera['height']))
            
            for param in camera['params']:
                f.write(struct.pack('d', param))


def write_images_binary(images: Dict, path: Path):
    """
    images.binをCOLMAPバイナリ形式で書き込み
    
    Args:
        images: 画像辞書
        path: 保存先パス
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('Q', len(images)))
        
        for img in images.values():
            f.write(struct.pack('i', img['id']))
            
            # クォータニオン (qw, qx, qy, qz)
            for q in img['qvec']:
                f.write(struct.pack('d', q))
            
            # 並進ベクトル
            for t in img['tvec']:
                f.write(struct.pack('d', t))
            
            f.write(struct.pack('i', img['camera_id']))
            
            # 画像名（null終端文字列）
            name_bytes = img['name'].encode('utf-8') + b'\x00'
            f.write(name_bytes)
            
            # 2D点
            f.write(struct.pack('Q', len(img['xys'])))
            for xy, p3d_id in zip(img['xys'], img['point3D_ids']):
                f.write(struct.pack('dd', xy[0], xy[1]))
                f.write(struct.pack('Q', p3d_id))


def write_points3d_binary(points3D: Dict, path: Path):
    """
    points3D.binをCOLMAPバイナリ形式で書き込み
    
    Args:
        points3D: 3D点辞書
        path: 保存先パス
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('Q', len(points3D)))
        
        for pt in points3D.values():
            f.write(struct.pack('Q', pt['id']))
            
            # XYZ座標
            for coord in pt['xyz']:
                f.write(struct.pack('d', coord))
            
            # RGB色
            for c in pt['rgb']:
                f.write(struct.pack('B', c))
            
            # エラー
            f.write(struct.pack('d', pt['error']))
            
            # トラック
            f.write(struct.pack('Q', len(pt['image_ids'])))
            for img_id, pt2d_idx in zip(pt['image_ids'], pt['point2D_idxs']):
                f.write(struct.pack('i', img_id))
                f.write(struct.pack('i', pt2d_idx))



