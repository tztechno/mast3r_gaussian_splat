# process2_17: simplified process2_16 w/deepseek

import struct
import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch

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
    PINHOLE = 1

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', num_cameras))

        for camera_id, (img_id, cam_params) in enumerate(cameras_dict.items(), start=1):
            focal = cam_params['focal']

            # 焦点距離の取得
            if isinstance(focal, (tuple, list)):
                fx, fy = focal
            else:
                fx = fy = focal

            # 主点の取得
            if 'pp' in cam_params:
                pp = cam_params['pp']
                cx = float(pp[0])
                cy = float(pp[1])
            else:
                cx = width / 2.0
                cy = height / 2.0

            f.write(struct.pack('I', camera_id))
            f.write(struct.pack('i', PINHOLE))
            f.write(struct.pack('Q', width))
            f.write(struct.pack('Q', height))
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


def write_points3D_binary(pts3d, confidence, colors=None, output_file):
    """
    points3D.binを出力（色付きまたはグレー）
    """
    num_points = len(pts3d)

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', num_points))

        for point_id, pt in enumerate(pts3d, start=1):
            x, y, z = pt

            f.write(struct.pack('Q', point_id))
            f.write(struct.pack('d', x))
            f.write(struct.pack('d', y))
            f.write(struct.pack('d', z))

            # RGB Color
            if colors is not None and point_id <= len(colors):
                # 実際の色を使用
                color = colors[point_id-1]
                r = int(np.clip(color[0], 0, 255))
                g = int(np.clip(color[1], 0, 255))
                b = int(np.clip(color[2], 0, 255))
            else:
                # グレー（デフォルト）
                r = g = b = 128

            f.write(struct.pack('B', r))
            f.write(struct.pack('B', g))
            f.write(struct.pack('B', b))

            # 誤差
            if confidence is not None and point_id <= len(confidence):
                error = 1.0 / max(confidence[point_id-1], 0.001)
            else:
                error = 1.0
            f.write(struct.pack('d', error))

            # トラック長
            f.write(struct.pack('Q', 0))

    print(f"COLMAP points3D.bin saved to {output_file}")


def export_colmap_binary(cameras_dict, pts3d, confidence, image_size, output_dir, colors=None):
    """COLMAPバイナリファイルを出力（色付きまたはグレー）"""
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
        colors,
        output_path / 'points3D.bin'
    )

    color_status = "WITH COLORS" if colors is not None else "gray"
    print(f"\n✓ COLMAP binary files exported to {output_dir}/")
    print(f"  - cameras.bin: {len(cameras_dict)} cameras (PINHOLE model)")
    print(f"  - images.bin: {len(cameras_dict)} images")
    print(f"  - points3D.bin: {len(pts3d)} points {color_status}")


def extract_camera_params_process2(
    scene, image_paths, conf_threshold=1.5, max_points=100000):
    """
    Extracts camera parameters and 3D points from the scene.
    """
    print("\n=== Extracting Camera Parameters ===")

    cameras_dict = {}
    all_pts3d = []
    all_confidence = []

    # シーンからパラメータを取得
    try:
        poses = scene.get_im_poses() if hasattr(scene, 'get_im_poses') else getattr(scene, 'im_poses', None)
        focals = scene.get_focals() if hasattr(scene, 'get_focals') else getattr(scene, 'im_focals', None)
        pps = scene.get_principal_points() if hasattr(scene, 'get_principal_points') else getattr(scene, 'im_pp', None)
    except Exception as e:
        print(f"⚠️ Error getting camera parameters: {e}")
        poses = focals = pps = None

    mast3r_size = 224.0
    n_images = min(len(poses) if poses is not None else len(image_paths), len(image_paths))

    for idx in range(n_images):
        img_name = os.path.basename(image_paths[idx])

        try:
            img = Image.open(image_paths[idx])
            W, H = img.size
            img.close()

            scale = W / mast3r_size

            # 姿勢の取得と変換
            if poses is not None and idx < len(poses):
                pose_c2w = poses[idx]
                if isinstance(pose_c2w, torch.Tensor):
                    pose_c2w = pose_c2w.detach().cpu().numpy()
                if not isinstance(pose_c2w, np.ndarray) or pose_c2w.shape != (4, 4):
                    pose_c2w = np.eye(4)
                pose = np.linalg.inv(pose_c2w)
            else:
                pose = np.eye(4)

            # 焦点距離の取得とスケーリング
            if focals is not None and idx < len(focals):
                focal_mast3r = focals[idx]
                if isinstance(focal_mast3r, torch.Tensor):
                    focal_mast3r = focal_mast3r.detach().cpu()

                # 等方性または異方性カメラの処理
                if focals.shape[1] == 1:
                    focal_val = float(focal_mast3r) if focal_mast3r.numel() == 1 else float(focal_mast3r[0])
                    fx = fy = focal_val * scale
                else:
                    fx = float(focal_mast3r[0]) * scale
                    fy = float(focal_mast3r[1]) * scale
            else:
                fx = fy = 1000.0

            # 主点の取得とスケーリング
            if pps is not None and idx < len(pps):
                pp_mast3r = pps[idx]
                if isinstance(pp_mast3r, torch.Tensor):
                    pp_mast3r = pp_mast3r.detach().cpu().numpy()
                pp = pp_mast3r * scale
            else:
                pp = np.array([W / 2.0, H / 2.0])

            # カメラパラメータの保存
            cameras_dict[img_name] = {
                'focal': (fx, fy),
                'pp': pp,
                'pose': pose,
                'rotation': pose[:3, :3],
                'translation': pose[:3, 3],
                'width': W,
                'height': H
            }

            # 3D点の抽出
            pts3d_img = None
            if hasattr(scene, 'im_pts3d') and idx < len(scene.im_pts3d):
                pts3d_img = scene.im_pts3d[idx]
            elif hasattr(scene, 'get_pts3d'):
                pts3d_all = scene.get_pts3d()
                pts3d_img = pts3d_all[idx] if idx < len(pts3d_all) else None

            # 信頼度の抽出
            conf_img = None
            if hasattr(scene, 'im_conf') and idx < len(scene.im_conf):
                conf_img = scene.im_conf[idx]
            elif hasattr(scene, 'get_conf'):
                conf_all = scene.get_conf()
                conf_img = conf_all[idx] if idx < len(conf_all) else None

            # 3D点と信頼度の処理
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

    # 3D点の統合
    if all_pts3d:
        pts3d = np.vstack(all_pts3d)
        confidence = np.concatenate(all_confidence)
    else:
        pts3d = np.zeros((0, 3))
        confidence = np.zeros(0)

    print(f"✓ Extracted parameters for {len(cameras_dict)} cameras")
    print(f"✓ Total 3D points before filtering: {len(pts3d)}")

    # フィルタリングとサンプリング
    if len(confidence) > 0:
        valid_mask = confidence > conf_threshold
        pts3d = pts3d[valid_mask]
        confidence = confidence[valid_mask]

        if len(pts3d) > max_points:
            print(f"  ! Sampling points from {len(pts3d):,} to {max_points:,}...")
            idx = np.random.choice(len(pts3d), max_points, replace=False)
            pts3d = pts3d[idx]
            confidence = confidence[idx]

        print(f"✓ Points after filtering and sampling: {len(pts3d):,}")

    return cameras_dict, pts3d, confidence


def extract_colors_from_images(scene, image_paths, pts3d, confidence, conf_threshold=1.5):
    """
    Extract colors from images that match the filtered pts3d.
    """
    print("\n=== Extracting Colors from Images ===")

    # すべての3D点を取得（フィルタリング前）
    all_pts3d = []
    for idx in range(len(image_paths)):
        pts3d_img = None
        if hasattr(scene, 'im_pts3d') and idx < len(scene.im_pts3d):
            pts3d_img = scene.im_pts3d[idx]
        elif hasattr(scene, 'get_pts3d'):
            pts3d_all = scene.get_pts3d()
            pts3d_img = pts3d_all[idx] if idx < len(pts3d_all) else None

        if pts3d_img is not None:
            if isinstance(pts3d_img, torch.Tensor):
                pts3d_img = pts3d_img.detach().cpu().numpy()
            pts3d_flat = pts3d_img.reshape(-1, 3) if pts3d_img.ndim == 3 else pts3d_img
            all_pts3d.append(pts3d_flat)

    # 最初の画像からサイズを取得
    first_img = Image.open(image_paths[0])
    W_orig, H_orig = first_img.size
    first_img.close()

    mast3r_size = 224
    all_colors = []

    # 各画像から色を抽出
    print(f"Extracting colors from {len(image_paths)} images...")
    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        img_resized = img.resize((mast3r_size, mast3r_size), Image.BILINEAR)
        img_array = np.array(img_resized)
        img.close()

        colors_flat = img_array.reshape(-1, 3)
        all_colors.append(colors_flat)

    colors_all = np.vstack(all_colors)
    print(f"✓ Total colors extracted: {len(colors_all):,}")

    # すべての信頼度を取得
    all_conf = []
    for idx in range(len(image_paths)):
        conf_img = None
        if hasattr(scene, 'im_conf') and idx < len(scene.im_conf):
            conf_img = scene.im_conf[idx]
        elif hasattr(scene, 'get_conf'):
            conf_all = scene.get_conf()
            conf_img = conf_all[idx] if idx < len(conf_all) else None

        if conf_img is not None:
            if isinstance(conf_img, torch.Tensor):
                conf_img = conf_img.detach().cpu().numpy()
            conf_flat = conf_img.reshape(-1) if conf_img.ndim > 1 else conf_img
        else:
            conf_flat = np.ones(len(all_pts3d[idx]) if idx < len(all_pts3d) else 0)

        all_conf.append(conf_flat)

    conf_all = np.concatenate(all_conf)

    # 3D点と同じフィルタリングを適用
    valid_mask = conf_all > conf_threshold
    colors_filtered = colors_all[valid_mask]

    print(f"✓ Colors after confidence filtering (>{conf_threshold}): {len(colors_filtered):,}")

    # 形状の一致を確認
    if len(colors_filtered) != len(pts3d):
        print(f"⚠️ WARNING: Color count ({len(colors_filtered)}) != Point count ({len(pts3d)})")
        min_len = min(len(colors_filtered), len(pts3d))
        colors_filtered = colors_filtered[:min_len]
        pts3d = pts3d[:min_len]
        confidence = confidence[:min_len]
    else:
        print(f"✓ Colors match points: {len(colors_filtered):,} colors for {len(pts3d):,} points")

    unique_colors = len(np.unique(colors_filtered, axis=0))
    print(f"✓ Unique colors: {unique_colors:,}")

    if unique_colors < 100:
        print(f"⚠️ WARNING: Very few unique colors!")

    return colors_filtered


def create_process2_with_colors(scene, image_paths, output_dir, conf_threshold=1.5, max_points=100000):
    """
    Complete workflow: Process2 with color extraction.
    """
    print("="*80)
    print("CREATING PROCESS2 COLMAP WITH COLORS")
    print("="*80)

    # ステップ1: カメラパラメータと3D点の抽出
    cameras_dict, pts3d, confidence = extract_camera_params_process2(
        scene, image_paths, conf_threshold=conf_threshold, max_points=max_points
    )

    print(f"\n✓ Extracted:")
    print(f"  - {len(cameras_dict)} cameras")
    print(f"  - {len(pts3d):,} 3D points")

    # ステップ2: 色の抽出
    colors = extract_colors_from_images(
        scene, image_paths, pts3d, confidence, conf_threshold
    )

    # ステップ3: 画像サイズの取得
    img = Image.open(image_paths[0])
    image_size = img.size
    img.close()

    # ステップ4: 色付きでエクスポート
    export_colmap_binary(
        cameras_dict, pts3d, confidence, image_size, output_dir, colors
    )

    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print("\nOutput directory:", output_dir)

    return cameras_dict, pts3d, confidence, colors
