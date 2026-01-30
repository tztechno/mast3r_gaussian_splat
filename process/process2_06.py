import struct
import numpy as np
import os
from pathlib import Path

# =====================================================================
# 汎用COLMAP bin → PLY変換関数
# =====================================================================

def generate_pointcloud_from_colmap_bins(colmap_sparse_dir, output_ply_path, colorize_by_height=True):
    """
    汎用的なCOLMAP → PLY変換関数
    
    cameras.bin, images.bin, points3D.bin から point_cloud.ply を生成
    Process 1, 2, 3 すべてで使用可能
    
    Args:
        colmap_sparse_dir: .binファイルが入っているディレクトリ
        output_ply_path: 出力PLYファイルのパス
        colorize_by_height: Trueなら高さベースで着色
    """
    import struct
    import numpy as np
    
    print("="*60)
    print("汎用COLMAP → PLY変換")
    print("="*60)
    print(f"入力: {colmap_sparse_dir}")
    print(f"出力: {output_ply_path}")
    
    points3d_bin = os.path.join(colmap_sparse_dir, "points3D.bin")
    
    if not os.path.exists(points3d_bin):
        print(f"❌ エラー: {points3d_bin} が見つかりません")
        return False
    
    # points3D.binを読み込む
    points = []
    colors = []
    
    with open(points3d_bin, 'rb') as f:
        num_points = struct.unpack('Q', f.read(8))[0]
        print(f"点の数: {num_points:,}")
        
        if num_points == 0:
            print("❌ 点群が空です")
            return False
        
        for i in range(num_points):
            point_id = struct.unpack('Q', f.read(8))[0]
            xyz = struct.unpack('ddd', f.read(24))
            rgb = struct.unpack('BBB', f.read(3))
            error = struct.unpack('d', f.read(8))[0]
            
            track_length = struct.unpack('Q', f.read(8))[0]
            for _ in range(track_length):
                struct.unpack('II', f.read(8))
            
            points.append(xyz)
            colors.append(rgb)
            
            if (i + 1) % 100000 == 0:
                print(f"  進捗: {i+1:,} / {num_points:,}")
    
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    
    print(f"✓ {len(points):,} 点を読み込み")
    
    # 高さベース着色
    if colorize_by_height:
        print("高さベースで着色中...")
        z_norm = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-8)
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 0] = np.clip(255 * 2 * z_norm, 0, 255).astype(np.uint8)
        colors[:, 1] = np.clip(255 * 2 * (1 - np.abs(z_norm - 0.5)), 0, 255).astype(np.uint8)
        colors[:, 2] = np.clip(255 * 2 * (1 - z_norm), 0, 255).astype(np.uint8)
    
    # PLY保存
    os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
    
    with open(output_ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for pt, color in zip(points, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {color[0]} {color[1]} {color[2]}\n")
    
    print(f"✓ PLY保存完了: {output_ply_path}")
    print(f"  ファイルサイズ: {os.path.getsize(output_ply_path) / (1024*1024):.2f} MB")
    print("="*60)
    
    return True


# =====================================================================
# Process2用: BINファイルのみ生成（TXTは作らない）
# =====================================================================

def write_colmap_bins_only(cameras_dict, pts3d, confidence, image_paths, output_dir):
    """
    COLMAP .bin ファイルのみを生成（.txt は作らない）
    """
    print(f"=== COLMAP バイナリファイル生成: {output_dir} ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 画像名→IDマッピング
    image_name_to_id = {}
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        image_name_to_id[img_name] = idx
    
    # cameras.bin
    cameras_file = os.path.join(output_dir, "cameras.bin")
    with open(cameras_file, 'wb') as f:
        f.write(struct.pack('Q', len(cameras_dict)))
        
        for cam_name, cam_params in cameras_dict.items():
            cam_id = image_name_to_id.get(cam_name, len(image_name_to_id))
            
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
            
            f.write(struct.pack('I', cam_id))
            f.write(struct.pack('i', 1))  # PINHOLE
            f.write(struct.pack('Q', int(width)))
            f.write(struct.pack('Q', int(height)))
            f.write(struct.pack('d', fx))
            f.write(struct.pack('d', fy))
            f.write(struct.pack('d', cx))
            f.write(struct.pack('d', cy))
    
    print(f"✓ cameras.bin: {len(cameras_dict)} cameras")
    
    # images.bin
    images_file = os.path.join(output_dir, "images.bin")
    with open(images_file, 'wb') as f:
        f.write(struct.pack('Q', len(cameras_dict)))
        
        for cam_name, cam_params in cameras_dict.items():
            cam_id = image_name_to_id.get(cam_name, len(image_name_to_id))
            
            if 'rotation' in cam_params:
                R = cam_params['rotation']
                quat = rotation_matrix_to_quaternion(R)
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            
            if 'translation' in cam_params:
                t = cam_params['translation']
                if isinstance(t, torch.Tensor):
                    t = t.cpu().numpy()
                tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
            else:
                tx, ty, tz = 0.0, 0.0, 0.0
            
            f.write(struct.pack('I', cam_id))
            f.write(struct.pack('d', quat[0]))
            f.write(struct.pack('d', quat[1]))
            f.write(struct.pack('d', quat[2]))
            f.write(struct.pack('d', quat[3]))
            f.write(struct.pack('d', tx))
            f.write(struct.pack('d', ty))
            f.write(struct.pack('d', tz))
            f.write(struct.pack('I', cam_id))
            
            name_bytes = cam_name.encode('utf-8')
            f.write(name_bytes)
            f.write(b'\x00')
            f.write(struct.pack('Q', 0))
    
    print(f"✓ images.bin: {len(cameras_dict)} images")
    
    # points3D.bin
    points_file = os.path.join(output_dir, "points3D.bin")
    with open(points_file, 'wb') as f:
        f.write(struct.pack('Q', len(pts3d)))
        
        for i, (pt, conf) in enumerate(zip(pts3d, confidence)):
            color_val = int(np.clip(conf * 50, 0, 255))
            
            f.write(struct.pack('Q', i))
            f.write(struct.pack('d', float(pt[0])))
            f.write(struct.pack('d', float(pt[1])))
            f.write(struct.pack('d', float(pt[2])))
            f.write(struct.pack('B', color_val))
            f.write(struct.pack('B', color_val))
            f.write(struct.pack('B', color_val))
            f.write(struct.pack('d', 0.0))
            f.write(struct.pack('Q', 0))
    
    print(f"✓ points3D.bin: {len(pts3d)} points")


def rotation_matrix_to_quaternion(R):
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
