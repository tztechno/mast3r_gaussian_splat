

def generate_pointcloud_from_colmap(colmap_sparse_dir, output_ply_path, colorize_by_height=True):
    """
    汎用的なCOLMAP → PLY変換関数
    
    どのプロセス（Process 1, 2, 3...）でも、
    cameras.bin, images.bin, points3D.binさえあれば使用可能
    
    Args:
        colmap_sparse_dir: cameras.bin等が入っているディレクトリ (例: output/sparse/0/)
        output_ply_path: 出力するPLYファイルのパス
        colorize_by_height: Trueなら高さベースで着色、Falseなら元の色を使用
    
    Returns:
        成功したらTrue、失敗したらFalse
    """
    import struct
    import numpy as np
    import os
    
    print("="*60)
    print("汎用COLMAP → PLY変換")
    print("="*60)
    print(f"入力ディレクトリ: {colmap_sparse_dir}")
    print(f"出力PLYファイル: {output_ply_path}")
    
    # 必要なファイルの確認
    cameras_bin = os.path.join(colmap_sparse_dir, "cameras.bin")
    images_bin = os.path.join(colmap_sparse_dir, "images.bin")
    points3d_bin = os.path.join(colmap_sparse_dir, "points3D.bin")
    
    missing_files = []
    if not os.path.exists(cameras_bin):
        missing_files.append("cameras.bin")
    if not os.path.exists(images_bin):
        missing_files.append("images.bin")
    if not os.path.exists(points3d_bin):
        missing_files.append("points3D.bin")
    
    if missing_files:
        print(f"❌ エラー: 以下のファイルが見つかりません:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    print("✓ 必要なファイルを確認しました")
    
    # points3D.binを読み込む
    print("\npoints3D.binを読み込み中...")
    points = []
    colors = []
    
    try:
        with open(points3d_bin, 'rb') as f:
            # 点の数を読み込む
            num_points = struct.unpack('Q', f.read(8))[0]
            print(f"  点の数: {num_points:,}")
            
            if num_points == 0:
                print("❌ エラー: 点群が空です")
                return False
            
            # 各点を読み込む
            for i in range(num_points):
                # Point3D構造:
                # - point_id (uint64)
                # - xyz (3 * double)
                # - rgb (3 * uint8)
                # - error (double)
                # - track (可変長)
                
                point_id = struct.unpack('Q', f.read(8))[0]
                xyz = struct.unpack('ddd', f.read(24))
                rgb = struct.unpack('BBB', f.read(3))
                error = struct.unpack('d', f.read(8))[0]
                
                # Track情報を読み飛ばす
                track_length = struct.unpack('Q', f.read(8))[0]
                for _ in range(track_length):
                    struct.unpack('II', f.read(8))  # image_id, point2D_idx
                
                points.append(xyz)
                colors.append(rgb)
                
                # 進捗表示
                if (i + 1) % 100000 == 0:
                    print(f"  進捗: {i+1:,} / {num_points:,} ({100*(i+1)/num_points:.1f}%)")
        
        print(f"✓ {len(points):,} 点を読み込みました")
        
    except Exception as e:
        print(f"❌ エラー: points3D.binの読み込みに失敗: {e}")
        return False
    
    # NumPy配列に変換
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    
    # 統計情報
    print("\n点群の統計:")
    print(f"  X範囲: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y範囲: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z範囲: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # 高さベースで着色する場合
    if colorize_by_height:
        print("\n高さベースの着色を適用中...")
        z_values = points[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        z_norm = (z_values - z_min) / (z_max - z_min + 1e-8)
        
        # カラーマップ: 青(低) → シアン → 緑 → 黄 → 赤(高)
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 0] = np.clip(255 * 2 * z_norm, 0, 255).astype(np.uint8)  # R
        colors[:, 1] = np.clip(255 * 2 * (1 - np.abs(z_norm - 0.5)), 0, 255).astype(np.uint8)  # G
        colors[:, 2] = np.clip(255 * 2 * (1 - z_norm), 0, 255).astype(np.uint8)  # B
        print("✓ 着色完了")
    else:
        print("\nCOLMAPの元の色を使用")
    
    # PLYファイルとして保存
    print(f"\nPLYファイルを書き込み中: {output_ply_path}")
    
    try:
        os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
        
        with open(output_ply_path, 'w') as f:
            # PLYヘッダー
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
            
            # データ
            for pt, color in zip(points, colors):
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {color[0]} {color[1]} {color[2]}\n")
        
        print(f"✓ PLYファイルを保存しました: {output_ply_path}")
        print(f"  ファイルサイズ: {os.path.getsize(output_ply_path) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"❌ エラー: PLYファイルの書き込みに失敗: {e}")
        return False
    
    print("="*60)
    print("✓ 変換完了")
    print("="*60)
    
    return True


def generate_pointcloud_from_colmap_binary(colmap_sparse_dir, output_ply_path, colorize_by_height=True):
    """
    上記のバイナリPLY版（ファイルサイズが小さく、読み込みが速い）
    """
    import struct
    import numpy as np
    import os
    
    print("="*60)
    print("汎用COLMAP → PLY変換（バイナリ形式）")
    print("="*60)
    
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
    
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    
    # 高さベース着色
    if colorize_by_height:
        z_norm = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-8)
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 0] = np.clip(255 * 2 * z_norm, 0, 255).astype(np.uint8)
        colors[:, 1] = np.clip(255 * 2 * (1 - np.abs(z_norm - 0.5)), 0, 255).astype(np.uint8)
        colors[:, 2] = np.clip(255 * 2 * (1 - z_norm), 0, 255).astype(np.uint8)
    
    # バイナリPLYとして保存
    os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
    
    with open(output_ply_path, 'wb') as f:
        # ヘッダー
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))
        
        # データ（バイナリ）
        for pt, color in zip(points, colors):
            f.write(struct.pack('fff', pt[0], pt[1], pt[2]))
            f.write(struct.pack('BBB', color[0], color[1], color[2]))
    
    print(f"✓ バイナリPLY保存完了: {output_ply_path}")
    print(f"  ファイルサイズ: {os.path.getsize(output_ply_path) / (1024*1024):.2f} MB")
    print("="*60)
    
    return True
