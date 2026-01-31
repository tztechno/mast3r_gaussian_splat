def generate_pointcloud_from_colmap_bins_robust(colmap_sparse_dir, output_ply_path, colorize_by_height=False):
    """
    Robust COLMAP to PLY conversion function (Enhanced error handling version).
    
    Args:
        colmap_sparse_dir (str): Path to the directory containing points3D.bin.
        output_ply_path (str): Path where the resulting .ply file will be saved.
        colorize_by_height (bool): If True, replaces original colors with a height-based gradient.
    """
    import struct
    import numpy as np
    import os
    
    print("="*60)
    print("Universal COLMAP -> PLY Conversion (Robust Version)")
    print("="*60)
    print(f"Input: {colmap_sparse_dir}")
    print(f"Output: {output_ply_path}")
    
    points3d_bin = os.path.join(colmap_sparse_dir, "points3D.bin")
    
    if not os.path.exists(points3d_bin):
        print(f"❌ Error: {points3d_bin} not found.")
        return False
    
    points = []
    colors = []
    
    try:
        with open(points3d_bin, 'rb') as f:
            # Read the number of points
            num_points_bytes = f.read(8)
            if len(num_points_bytes) < 8:
                print(f"❌ Error: File is truncated (too short).")
                return False
            
            num_points = struct.unpack('Q', num_points_bytes)[0]
            print(f"Total points: {num_points:,}")
            
            if num_points == 0:
                print("❌ Error: Point cloud is empty.")
                return False
            
            # Read each point
            for i in range(num_points):
                try:
                    # Point ID (8 bytes)
                    point_id_bytes = f.read(8)
                    if len(point_id_bytes) < 8:
                        print(f"⚠️ Point {i}: Failed to read Point ID")
                        break
                    point_id = struct.unpack('Q', point_id_bytes)[0]
                    
                    # XYZ (24 bytes)
                    xyz_bytes = f.read(24)
                    if len(xyz_bytes) < 24:
                        print(f"⚠️ Point {i}: Failed to read XYZ coordinates")
                        break
                    xyz = struct.unpack('ddd', xyz_bytes)
                    
                    # RGB (3 bytes)
                    rgb_bytes = f.read(3)
                    if len(rgb_bytes) < 3:
                        print(f"⚠️ Point {i}: Failed to read RGB data")
                        break
                    rgb = struct.unpack('BBB', rgb_bytes)
                    
                    # Error (8 bytes)
                    error_bytes = f.read(8)
                    if len(error_bytes) < 8:
                        print(f"⚠️ Point {i}: Failed to read Error value")
                        break
                    error = struct.unpack('d', error_bytes)[0]
                    
                    # Track length (8 bytes)
                    track_len_bytes = f.read(8)
                    if len(track_len_bytes) < 8:
                        print(f"⚠️ Point {i}: Failed to read Track length")
                        break
                    track_length = struct.unpack('Q', track_len_bytes)[0]
                    
                    # Track data (track_length * 8 bytes)
                    if track_length > 0:
                        track_bytes = f.read(track_length * 8)
                        if len(track_bytes) < track_length * 8:
                            print(f"⚠️ Point {i}: Failed to read Track data")
                            break
                        # Track data is skipped as it is not needed for PLY
                    
                    points.append(xyz)
                    colors.append(rgb)
                    
                    # Progress update
                    if (i + 1) % 100000 == 0:
                        print(f"  Progress: {i+1:,} / {num_points:,} ({100*(i+1)/num_points:.1f}%)")
                
                except struct.error as e:
                    print(f"⚠️ Point {i}: Struct unpacking error: {e}")
                    break
                except Exception as e:
                    print(f"⚠️ Point {i}: Unexpected error: {e}")
                    break
        
        if len(points) == 0:
            print("❌ Error: No valid points found.")
            return False
        
        print(f"✓ Successfully loaded {len(points):,} points.")
        
    except Exception as e:
        print(f"❌ Error: Failed to read file: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Convert to NumPy arrays
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    
    # Statistics
    print("\nPoint Cloud Statistics:")
    print(f"  X Range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y Range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z Range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # Color statistics
    unique_colors = len(np.unique(colors, axis=0))
    print(f"  Unique colors: {unique_colors:,}")
    print(f"  Sample colors (first 5): {colors[:5].tolist()}")
    
    # Apply height-based colorization if requested
    if colorize_by_height:
        print("\nApplying height-based colorization...")
        z_values = points[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        z_norm = (z_values - z_min) / (z_max - z_min + 1e-8)
        
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 0] = np.clip(255 * 2 * z_norm, 0, 255).astype(np.uint8)
        colors[:, 1] = np.clip(255 * 2 * (1 - np.abs(z_norm - 0.5)), 0, 255).astype(np.uint8)
        colors[:, 2] = np.clip(255 * 2 * (1 - z_norm), 0, 255).astype(np.uint8)
        print("✓ Colorization complete.")
    else:
        print("\nUsing original COLMAP colors.")
    
    # Write to PLY file
    print(f"\nWriting PLY file: {output_ply_path}")
    
    try:
        os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
        
        with open(output_ply_path, 'w') as f:
            # PLY Header
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
            
            # Data
            for pt, color in zip(points, colors):
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {color[0]} {color[1]} {color[2]}\n")
        
        print(f"✓ PLY file saved: {output_ply_path}")
        print(f"  File size: {os.path.getsize(output_ply_path) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"❌ Error: Failed to write PLY file: {e}")
        return False
    
    print("="*60)
    print("✓ Conversion Complete!")
    print("="*60)
    
    return True


def generate_pointcloud_from_colmap_bins_robust(colmap_sparse_dir, output_ply_path, colorize_by_height=False):
    """
    堅牢なCOLMAP → PLY変換関数（エラー処理強化版）
    """
    import struct
    import numpy as np
    import os
    
    print("="*60)
    print("汎用COLMAP → PLY変換（堅牢版）")
    print("="*60)
    print(f"入力: {colmap_sparse_dir}")
    print(f"出力: {output_ply_path}")
    
    points3d_bin = os.path.join(colmap_sparse_dir, "points3D.bin")
    
    if not os.path.exists(points3d_bin):
        print(f"❌ エラー: {points3d_bin} が見つかりません")
        return False
    
    points = []
    colors = []
    
    try:
        with open(points3d_bin, 'rb') as f:
            # 点の数を読み込む
            num_points_bytes = f.read(8)
            if len(num_points_bytes) < 8:
                print(f"❌ エラー: ファイルが短すぎます")
                return False
            
            num_points = struct.unpack('Q', num_points_bytes)[0]
            print(f"点の数: {num_points:,}")
            
            if num_points == 0:
                print("❌ 点群が空です")
                return False
            
            # 各点を読み込む
            for i in range(num_points):
                try:
                    # Point ID (8 bytes)
                    point_id_bytes = f.read(8)
                    if len(point_id_bytes) < 8:
                        print(f"⚠️ 点 {i}: Point ID読み込み失敗")
                        break
                    point_id = struct.unpack('Q', point_id_bytes)[0]
                    
                    # XYZ (24 bytes)
                    xyz_bytes = f.read(24)
                    if len(xyz_bytes) < 24:
                        print(f"⚠️ 点 {i}: XYZ読み込み失敗")
                        break
                    xyz = struct.unpack('ddd', xyz_bytes)
                    
                    # RGB (3 bytes)
                    rgb_bytes = f.read(3)
                    if len(rgb_bytes) < 3:
                        print(f"⚠️ 点 {i}: RGB読み込み失敗")
                        break
                    rgb = struct.unpack('BBB', rgb_bytes)
                    
                    # Error (8 bytes)
                    error_bytes = f.read(8)
                    if len(error_bytes) < 8:
                        print(f"⚠️ 点 {i}: Error読み込み失敗")
                        break
                    error = struct.unpack('d', error_bytes)[0]
                    
                    # Track length (8 bytes)
                    track_len_bytes = f.read(8)
                    if len(track_len_bytes) < 8:
                        print(f"⚠️ 点 {i}: Track length読み込み失敗")
                        break
                    track_length = struct.unpack('Q', track_len_bytes)[0]
                    
                    # Track data (track_length * 8 bytes)
                    if track_length > 0:
                        track_bytes = f.read(track_length * 8)
                        if len(track_bytes) < track_length * 8:
                            print(f"⚠️ 点 {i}: Track data読み込み失敗")
                            break
                        # Track dataは使わないのでスキップ
                    
                    points.append(xyz)
                    colors.append(rgb)
                    
                    # 進捗表示
                    if (i + 1) % 100000 == 0:
                        print(f"  進捗: {i+1:,} / {num_points:,} ({100*(i+1)/num_points:.1f}%)")
                
                except struct.error as e:
                    print(f"⚠️ 点 {i}: 構造体エラー: {e}")
                    break
                except Exception as e:
                    print(f"⚠️ 点 {i}: 予期しないエラー: {e}")
                    break
        
        if len(points) == 0:
            print("❌ エラー: 有効な点が見つかりませんでした")
            return False
        
        print(f"✓ {len(points):,} 点を読み込み")
        
    except Exception as e:
        print(f"❌ エラー: ファイル読み込み失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # NumPy配列に変換
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    
    # 統計情報
    print("\n点群の統計:")
    print(f"  X範囲: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y範囲: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z範囲: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # 色の統計
    unique_colors = len(np.unique(colors, axis=0))
    print(f"  ユニーク色数: {unique_colors:,}")
    print(f"  サンプル色: {colors[:5].tolist()}")
    
    # 高さベースで着色する場合
    if colorize_by_height:
        print("\n高さベースの着色を適用中...")
        z_values = points[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        z_norm = (z_values - z_min) / (z_max - z_min + 1e-8)
        
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 0] = np.clip(255 * 2 * z_norm, 0, 255).astype(np.uint8)
        colors[:, 1] = np.clip(255 * 2 * (1 - np.abs(z_norm - 0.5)), 0, 255).astype(np.uint8)
        colors[:, 2] = np.clip(255 * 2 * (1 - z_norm), 0, 255).astype(np.uint8)
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



def generate_pointcloud_from_colmap(colmap_sparse_dir, output_ply_path, colorize_by_height=False):
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
