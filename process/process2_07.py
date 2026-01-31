# process2_06_optimized.py
# 最適化版: 重複・無駄を削除し、3つのbin出力に特化

import struct
import numpy as np
from pathlib import Path
from PIL import Image
import os
import torch


# =====================================================================
# CORE UTILITIES
# =====================================================================

def rotmat_to_qvec(R):
    """回転行列をクォータニオン(w,x,y,z)に変換"""
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


# =====================================================================
# BINARY WRITERS
# =====================================================================

def write_cameras_binary(cameras_dict, image_size, output_file):
    """cameras.binを出力（PINHOLEモデル: fx, fy, cx, cy）"""
    width, height = image_size
    num_cameras = len(cameras_dict)
    PINHOLE = 1

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', num_cameras))

        for camera_id, (img_id, cam_params) in enumerate(cameras_dict.items(), start=1):
            focal = cam_params['focal']

            # fx, fy取得
            if isinstance(focal, (tuple, list)):
                fx, fy = focal
            else:
                fx = fy = focal

            # 主点取得
            if 'pp' in cam_params:
                pp = cam_params['pp']
                cx = float(pp[0])
                cy = float(pp[1])
            else:
                cx = width / 2.0
                cy = height / 2.0

            # バイナリ書き込み
            f.write(struct.pack('I', camera_id))
            f.write(struct.pack('i', PINHOLE))
            f.write(struct.pack('Q', width))
            f.write(struct.pack('Q', height))
            f.write(struct.pack('d', fx))
            f.write(struct.pack('d', fy))
            f.write(struct.pack('d', cx))
            f.write(struct.pack('d', cy))

    print(f"✓ cameras.bin saved: {num_cameras} cameras")


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

            # 画像名
            name_bytes = img_id.encode('utf-8') + b'\x00'
            f.write(name_bytes)
            f.write(struct.pack('Q', 0))  # Points2D count

    print(f"✓ images.bin saved: {num_images} images")


def write_points3D_binary_with_colors(pts3d, confidence, colors, output_file):
    """points3D.binを色付きで出力"""
    num_points = len(pts3d)

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', num_points))

        for point_id, (pt, color) in enumerate(zip(pts3d, colors), start=1):
            x, y, z = pt

            f.write(struct.pack('Q', point_id))
            f.write(struct.pack('d', x))
            f.write(struct.pack('d', y))
            f.write(struct.pack('d', z))

            # RGB色
            r = int(np.clip(color[0], 0, 255))
            g = int(np.clip(color[1], 0, 255))
            b = int(np.clip(color[2], 0, 255))

            f.write(struct.pack('B', r))
            f.write(struct.pack('B', g))
            f.write(struct.pack('B', b))

            # エラー推定
            if confidence is not None and point_id <= len(confidence):
                error = 1.0 / max(confidence[point_id-1], 0.001)
            else:
                error = 1.0
            f.write(struct.pack('d', error))

            # track_length
            f.write(struct.pack('Q', 0))

    print(f"✓ points3D.bin saved: {num_points} points with colors")


# =====================================================================
# CAMERA PARAMETER EXTRACTION
# =====================================================================

def extract_camera_params_process2(scene, image_paths, conf_threshold=1.5):
    """
    カメラパラメータと3D点を抽出（最適化版）
    
    Returns:
        cameras_dict: カメラパラメータ辞書
        pts3d: フィルタ済み3D点 (N, 3)
        confidence: フィルタ済み信頼度 (N,)
        all_pts3d_raw: フィルタ前の全3D点（色抽出用）
        all_conf_raw: フィルタ前の全信頼度（色抽出用）
    """
    print("\n=== Extracting Camera Parameters ===")

    cameras_dict = {}
    all_pts3d = []
    all_confidence = []

    # カメラ情報取得
    try:
        poses = scene.get_im_poses() if hasattr(scene, 'get_im_poses') else scene.im_poses if hasattr(scene, 'im_poses') else None
        focals = scene.get_focals() if hasattr(scene, 'get_focals') else scene.im_focals if hasattr(scene, 'im_focals') else None
        pps = scene.get_principal_points() if hasattr(scene, 'get_principal_points') else scene.im_pp if hasattr(scene, 'im_pp') else None
    except Exception as e:
        print(f"⚠️ Error getting camera parameters: {e}")
        poses = focals = pps = None

    mast3r_size = 224.0
    n_images = min(len(poses) if poses is not None else len(image_paths), len(image_paths))

    for idx in range(n_images):
        img_name = os.path.basename(image_paths[idx])

        try:
            # 画像サイズ取得
            img = Image.open(image_paths[idx])
            W, H = img.size
            img.close()

            scale = W / mast3r_size

            # Pose取得
            if poses is not None and idx < len(poses):
                pose_c2w = poses[idx]
                if isinstance(pose_c2w, torch.Tensor):
                    pose_c2w = pose_c2w.detach().cpu().numpy()
                if not isinstance(pose_c2w, np.ndarray) or pose_c2w.shape != (4, 4):
                    pose_c2w = np.eye(4)
                pose = np.linalg.inv(pose_c2w)
            else:
                pose = np.eye(4)

            # 焦点距離取得
            if focals is not None and idx < len(focals):
                focal_mast3r = focals[idx]
                if isinstance(focal_mast3r, torch.Tensor):
                    focal_mast3r = focal_mast3r.detach().cpu()

                if focals.shape[1] == 1:
                    focal_val = float(focal_mast3r) if focal_mast3r.numel() == 1 else float(focal_mast3r[0])
                    fx = fy = focal_val * scale
                else:
                    fx = float(focal_mast3r[0]) * scale
                    fy = float(focal_mast3r[1]) * scale
            else:
                fx = fy = 1000.0

            # 主点取得
            if pps is not None and idx < len(pps):
                pp_mast3r = pps[idx]
                if isinstance(pp_mast3r, torch.Tensor):
                    pp_mast3r = pp_mast3r.detach().cpu().numpy()
                pp = pp_mast3r * scale
            else:
                pp = np.array([W / 2.0, H / 2.0])

            # カメラ情報保存
            cameras_dict[img_name] = {
                'focal': (fx, fy),
                'pp': pp,
                'pose': pose,
                'rotation': pose[:3, :3],
                'translation': pose[:3, 3],
                'width': W,
                'height': H
            }

            # デバッグ情報（最初の画像のみ）
            if idx == 0:
                print(f"\nExample camera 0:")
                print(f"  Original size: {W}x{H}")
                print(f"  Scale factor: {scale:.3f}")
                print(f"  Scaled focal: fx={fx:.2f}, fy={fy:.2f}")
                print(f"  Scaled pp: [{pp[0]:.2f}, {pp[1]:.2f}]")

            # 3D点取得
            if hasattr(scene, 'im_pts3d') and idx < len(scene.im_pts3d):
                pts3d_img = scene.im_pts3d[idx]
            elif hasattr(scene, 'get_pts3d'):
                pts3d_all = scene.get_pts3d()
                pts3d_img = pts3d_all[idx] if idx < len(pts3d_all) else None
            else:
                pts3d_img = None

            # 信頼度取得
            if hasattr(scene, 'im_conf') and idx < len(scene.im_conf):
                conf_img = scene.im_conf[idx]
            elif hasattr(scene, 'get_conf'):
                conf_all = scene.get_conf()
                conf_img = conf_all[idx] if idx < len(conf_all) else None
            else:
                conf_img = None

            # 処理
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

    # 統合（フィルタ前データも保存）
    if all_pts3d:
        pts3d_raw = np.vstack(all_pts3d)
        conf_raw = np.concatenate(all_confidence)
    else:
        pts3d_raw = np.zeros((0, 3))
        conf_raw = np.zeros(0)

    print(f"✓ Extracted parameters for {len(cameras_dict)} cameras")
    print(f"✓ Total 3D points (raw): {len(pts3d_raw)}")

    # フィルタリング
    if len(conf_raw) > 0:
        valid_mask = conf_raw > conf_threshold
        pts3d = pts3d_raw[valid_mask]
        confidence = conf_raw[valid_mask]
        print(f"✓ Points after filtering (>{conf_threshold}): {len(pts3d)}")
    else:
        pts3d = pts3d_raw
        confidence = conf_raw

    return cameras_dict, pts3d, confidence, pts3d_raw, conf_raw


# =====================================================================
# COLOR EXTRACTION (最適化版)
# =====================================================================

def extract_colors_from_images_optimized(image_paths, pts3d, conf_raw, conf_threshold=1.5):
    """
    画像から色を抽出（最適化版: 3D点の再取得を削除）
    
    Args:
        image_paths: 画像パスリスト
        pts3d: フィルタ済み3D点 (N, 3)
        conf_raw: フィルタ前の全信頼度 (対応する色を抽出するため)
        conf_threshold: 信頼度閾値
    
    Returns:
        colors: (N, 3) RGB色 [0-255]
    """
    print("\n=== Extracting Colors from Images ===")

    # 画像サイズ取得
    first_img = Image.open(image_paths[0])
    W_orig, H_orig = first_img.size
    first_img.close()

    mast3r_size = 224

    # 色抽出
    print(f"Extracting colors from {len(image_paths)} images...")
    all_colors = []

    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        img_resized = img.resize((mast3r_size, mast3r_size), Image.BILINEAR)
        img_array = np.array(img_resized)
        img.close()

        colors_flat = img_array.reshape(-1, 3)
        all_colors.append(colors_flat)

        if idx == 0:
            print(f"  Example image 0:")
            print(f"    Original: {W_orig}x{H_orig} -> Resized: {mast3r_size}x{mast3r_size}")

    colors_all = np.vstack(all_colors)
    print(f"✓ Total colors extracted: {len(colors_all):,}")

    # 同じフィルタを適用
    valid_mask = conf_raw > conf_threshold
    colors_filtered = colors_all[valid_mask]

    print(f"✓ Colors after filtering: {len(colors_filtered):,}")

    # 検証
    if len(colors_filtered) != len(pts3d):
        print(f"⚠️ WARNING: Color/Point mismatch: {len(colors_filtered)} vs {len(pts3d)}")
        min_len = min(len(colors_filtered), len(pts3d))
        colors_filtered = colors_filtered[:min_len]
    else:
        print(f"✓ Colors match points: {len(colors_filtered):,}")

    unique_colors = len(np.unique(colors_filtered, axis=0))
    print(f"✓ Unique colors: {unique_colors:,}")

    return colors_filtered


# =====================================================================
# EXPORT FUNCTION
# =====================================================================

def export_colmap_binary_with_colors(cameras_dict, pts3d, confidence, colors,
                                     image_size, output_dir):
    """COLMAP バイナリファイル（3種類）を色付きで出力"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    write_cameras_binary(cameras_dict, image_size, output_path / 'cameras.bin')
    write_images_binary(cameras_dict, output_path / 'images.bin')
    write_points3D_binary_with_colors(pts3d, confidence, colors, output_path / 'points3D.bin')

    print(f"\n{'='*80}")
    print(f"✓ COLMAP files exported to: {output_dir}/")
    print(f"  - cameras.bin: {len(cameras_dict)} cameras (PINHOLE)")
    print(f"  - images.bin: {len(cameras_dict)} images")
    print(f"  - points3D.bin: {len(pts3d)} points WITH COLORS")
    print(f"{'='*80}")


# =====================================================================
# MAIN WORKFLOW
# =====================================================================

def create_process2_with_colors(scene, image_paths, output_dir, conf_threshold=1.5):
    """
    完全なワークフロー: Process2 with colors
    
    Usage:
        create_process2_with_colors(
            scene, 
            image_paths, 
            '/output/sparse/0',
            conf_threshold=1.5
        )
    """
    print("="*80)
    print("CREATING PROCESS2 COLMAP WITH COLORS (OPTIMIZED)")
    print("="*80)

    # Step 1: カメラパラメータと3D点を抽出
    cameras_dict, pts3d, confidence, pts3d_raw, conf_raw = extract_camera_params_process2(
        scene, image_paths, conf_threshold=conf_threshold
    )

    print(f"\n✓ Extracted:")
    print(f"  - {len(cameras_dict)} cameras")
    print(f"  - {len(pts3d):,} filtered 3D points")

    # Step 2: 色抽出（最適化版）
    colors = extract_colors_from_images_optimized(
        image_paths, pts3d, conf_raw, conf_threshold
    )

    # Step 3: 画像サイズ取得
    img = Image.open(image_paths[0])
    image_size = img.size
    img.close()

    # Step 4: エクスポート
    export_colmap_binary_with_colors(
        cameras_dict, pts3d, confidence, colors,
        image_size, output_dir
    )

    print("\n✓ COMPLETE!")
    print(f"\nOutput: {output_dir}")

    return cameras_dict, pts3d, confidence, colors


# =====================================================================
# OPTIONAL: PLY Export (最小限)
# =====================================================================

def write_colored_ply(pts3d, colors, output_path):
    """
    オプション: PLYファイル出力（高さベースではなく実際の色）
    """
    print(f"Writing colored PLY to {output_path}...")

    with open(output_path, 'w') as f:
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

        for pt, color in zip(pts3d, colors):
            r = int(np.clip(color[0], 0, 255))
            g = int(np.clip(color[1], 0, 255))
            b = int(np.clip(color[2], 0, 255))
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {r} {g} {b}\n")

    print(f"✓ Wrote PLY with {len(pts3d)} colored points")



##################################################
##################################################
##################################################



