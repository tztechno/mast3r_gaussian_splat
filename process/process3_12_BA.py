import numpy as np
import cv2
from pathlib import Path
import struct
from scipy.spatial.transform import Rotation
import torch
from PIL import Image
import pycolmap  # 束調整(BA)の実行に必要です

# ============================================================================
# COLMAP Binary Writers (Binary Format Helper)
# ============================================================================

def write_next_bytes(fid, data, format_str):
    if isinstance(data, (list, tuple, np.ndarray)):
        fid.write(struct.pack("<" + format_str, *data))
    else:
        fid.write(struct.pack("<" + format_str, data))

def matrix_to_quaternion_translation(matrix: np.ndarray):
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # [x, y, z, w]
    qvec = np.array([quat[3], quat[0], quat[1], quat[2]]) # COLMAP format [w, x, y, z]
    return qvec, t

def write_cameras_binary(cameras, path_to_model_file):
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
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for image_id, img in images.items():
            write_next_bytes(fid, image_id, "I")
            write_next_bytes(fid, img['qvec'], "dddd")
            write_next_bytes(fid, img['tvec'], "ddd")
            write_next_bytes(fid, img['camera_id'], "I")
            for char in img['name']:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img['xys']), "Q")
            for xy, point3D_id in zip(img['xys'], img['point3D_ids']):
                write_next_bytes(fid, xy, "dd")
                write_next_bytes(fid, point3D_id, "Q")

def write_points3d_binary(points3D, path_to_model_file):
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for point_id, point in enumerate(points3D if isinstance(points3D, list) else points3D.values()):
            pid = point['id'] if isinstance(point, dict) and 'id' in point else point_id
            write_next_bytes(fid, pid, "Q")
            write_next_bytes(fid, point['xyz'], "ddd")
            write_next_bytes(fid, point['rgb'], "BBB")
            write_next_bytes(fid, point['error'], "d")
            track_length = len(point['image_ids'])
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_idx in zip(point['image_ids'], point['point2D_idxs']):
                write_next_bytes(fid, int(image_id), "I")
                write_next_bytes(fid, int(point2D_idx), "I")

# ============================================================================
# Optimization Engine (Bundle Adjustment)
# ============================================================================

def run_global_bundle_adjustment(sparse_dir):
    """
    COLMAPバイナリを読み込み、カメラ位置と3D点群を同時に最適化(BA)して上書き保存する
    """
    print("\n" + "="*50)
    print("RUNNING GLOBAL BUNDLE ADJUSTMENT")
    print("="*50)
    
    if not (sparse_dir / "cameras.bin").exists():
        print("Error: Model files not found for BA.")
        return

    # メモリにCOLMAPモデルをロード
    reconstruction = pycolmap.Reconstruction(sparse_dir)
    
    # 束調整のオプション設定
    options = pycolmap.BundleAdjustmentOptions()
    options.solver_options.num_threads = 8  # CPUスレッド数
    options.solver_options.max_num_iterations = 100 # 収束までの最大イテレーション
    
    print(f"Initial Mean Reprojection Error: {reconstruction.compute_mean_reprojection_error():.4f} pixels")
    
    # 最適化実行
    pycolmap.bundle_adjustment(reconstruction, options)
    
    print(f"Final Mean Reprojection Error: {reconstruction.compute_mean_reprojection_error():.4f} pixels")
    
    # 最適化されたモデルを上書き
    reconstruction.write(sparse_dir)
    print("✓ Model successfully refined and saved.")
    print("="*50 + "\n")

# ============================================================================
# Core Logic
# ============================================================================

def extract_scene_data(scene, min_conf_thr, verbose):
    """(既存の抽出ロジック) sceneからcamera, image, point3Dを抽出"""
    # ※ここは元コードの extract_scene_data 関数をそのまま配置してください
    # 紙面の都合上省略しますが、実際の運用では全行必要です。
    pass

def save_image_data(scene, images_dir, depth_dir, normal_dir, mask_dir, min_conf_thr, verbose, processed_image_paths=None):
    """(既存の画像保存ロジック)"""
    # ※ここは元コードの save_image_data 関数をそのまま配置してください。
    pass

def convert_mast3r_to_colmap(scene, output_dir, min_conf_thr=1.5, clean_depth=True, 
                            mask_images=True, verbose=True, processed_image_paths=None,
                            max_points=100000):
    
    output_dir = Path(output_dir)
    sparse_dir = output_dir / "sparse" / "0"
    images_dir = output_dir / "images"
    
    # ディレクトリ作成
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\nConverting to COLMAP: {output_dir}")
    
    # 1. データの抽出
    cameras, images_data, points3D = extract_scene_data(scene, min_conf_thr, verbose)

    # 2. ダウンサンプリング処理
    if max_points is not None and len(points3D) > max_points:
        if verbose: print(f"Downsampling points to {max_points}...")
        sampled_indices = np.random.choice(len(points3D), max_points, replace=False)
        points3D = [points3D[i] for i in sampled_indices]
    
    # 3. 画像データの保存
    save_image_data(scene, images_dir, output_dir/"depth", output_dir/"normal", 
                    output_dir/"mask", min_conf_thr, verbose, processed_image_paths)
    
    # 4. バイナリファイルの書き出し（暫定モデル）
    write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    write_images_binary(images_data, sparse_dir / "images.bin")
    write_points3d_binary(points3D, sparse_dir / "points3D.bin")
    
    # 5. 【追加】束調整(Bundle Adjustment)の実行
    # ここでバラバラに重なっていた像が、再投影誤差が最小になる位置へシュッと収束します。
    try:
        run_global_bundle_adjustment(sparse_dir)
    except Exception as e:
        print(f"Bundle Adjustment failed: {e}")
        print("Note: Check if 'pycolmap' is installed or if tracks are valid.")
    
    if verbose:
        print("✓ Process complete!")
    
    return output_dir
