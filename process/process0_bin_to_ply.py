def generate_pointcloud_from_colmap_bins_robust(colmap_sparse_dir, output_ply_path, colorize_by_height=False):
    """
    完全修正版: COLMAP to PLY conversion function
    
    主な修正点:
    1. ❌ 削除: num_points_bytes = f.read(8) - points3D.binにはヘッダーが存在しない！
    2. ✅ 修正: EOFまでループで読み込む方式に変更
    3. ✅ 追加: Inf/NaN値の検出と除去
    4. ✅ 修正: エンディアン指定 ('<' をstructに追加)
    5. ✅ 強化: エラーハンドリングと進捗表示
    
    Args:
        colmap_sparse_dir (str): Path to the directory containing points3D.bin.
        output_ply_path (str): Path where the resulting .ply file will be saved.
        colorize_by_height (bool): If True, replaces original colors with a height-based gradient.
    """
    import struct
    import numpy as np
    import os
    
    print("="*60)
    print("FIXED COLMAP -> PLY Conversion")
    print("="*60)
    print(f"Input: {colmap_sparse_dir}")
    print(f"Output: {output_ply_path}")
    
    points3d_bin = os.path.join(colmap_sparse_dir, "points3D.bin")
    
    if not os.path.exists(points3d_bin):
        print(f"❌ Error: {points3d_bin} not found.")
        return False
    
    # ファイルサイズチェック
    file_size = os.path.getsize(points3d_bin)
    print(f"File size: {file_size:,} bytes ({file_size/(1024**2):.2f} MB)")
    
    if file_size == 0:
        print("❌ Error: File is empty.")
        return False
    
    points = []
    colors = []
    
    try:
        with open(points3d_bin, 'rb') as f:
            # ❌ 削除されたバグコード:
            # num_points_bytes = f.read(8)  # <- これが間違い！
            # num_points = struct.unpack('Q', num_points_bytes)[0]
            
            # ✅ 正しい方法: EOFまでループ
            point_count = 0
            
            while True:
                # Point ID (8 bytes)
                point_id_bytes = f.read(8)
                
                # EOF チェック
                if len(point_id_bytes) == 0:
                    # 正常終了
                    break
                
                if len(point_id_bytes) < 8:
                    print(f"⚠️ Point {point_count}: Incomplete Point ID (EOF)")
                    break
                
                try:
                    # ✅ エンディアン指定を追加 '<' = little-endian
                    point_id = struct.unpack('<Q', point_id_bytes)[0]
                    
                    # XYZ (24 bytes = 3 × 8 bytes)
                    xyz_bytes = f.read(24)
                    if len(xyz_bytes) < 24:
                        print(f"⚠️ Point {point_count}: Failed to read XYZ coordinates")
                        break
                    xyz = struct.unpack('<3d', xyz_bytes)  # ✅ '<3d' に修正
                    
                    # RGB (3 bytes)
                    rgb_bytes = f.read(3)
                    if len(rgb_bytes) < 3:
                        print(f"⚠️ Point {point_count}: Failed to read RGB data")
                        break
                    rgb = struct.unpack('<3B', rgb_bytes)  # ✅ '<3B' に修正
                    
                    # Error (8 bytes)
                    error_bytes = f.read(8)
                    if len(error_bytes) < 8:
                        print(f"⚠️ Point {point_count}: Failed to read Error value")
                        break
                    error = struct.unpack('<d', error_bytes)[0]  # ✅ '<d' に修正
                    
                    # Track length (8 bytes)
                    track_len_bytes = f.read(8)
                    if len(track_len_bytes) < 8:
                        print(f"⚠️ Point {point_count}: Failed to read Track length")
                        break
                    track_length = struct.unpack('<Q', track_len_bytes)[0]  # ✅ '<Q' に修正
                    
                    # Track data (track_length * 8 bytes)
                    # 各エントリ: image_id (uint32=4bytes) + point2D_idx (uint32=4bytes) = 8 bytes
                    if track_length > 0:
                        track_bytes_size = track_length * 8
                        track_bytes = f.read(track_bytes_size)
                        if len(track_bytes) < track_bytes_size:
                            print(f"⚠️ Point {point_count}: Failed to read Track data (expected {track_bytes_size}, got {len(track_bytes)})")
                            break
                        # Track data is skipped as it is not needed for PLY
                    
                    points.append(xyz)
                    colors.append(rgb)
                    point_count += 1
                    
                    # Progress update
                    if point_count % 100000 == 0:
                        print(f"  Progress: {point_count:,} points loaded...")
                
                except struct.error as e:
                    print(f"⚠️ Point {point_count}: Struct unpacking error: {e}")
                    break
                except Exception as e:
                    print(f"⚠️ Point {point_count}: Unexpected error: {e}")
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
    # ✅ float64で保持（精度を維持）
    points = np.array(points, dtype=np.float64)
    colors = np.array(colors, dtype=np.uint8)
    
    # ✅ Inf/NaN値のチェックと除去
    print("\nValidating point cloud data...")
    
    has_inf = np.isinf(points).any()
    has_nan = np.isnan(points).any()
    
    if has_inf:
        inf_count = np.isinf(points).sum()
        print(f"⚠️ WARNING: {inf_count} infinite values detected!")
        valid_mask = ~np.isinf(points).any(axis=1)
        points = points[valid_mask]
        colors = colors[valid_mask]
        print(f"  → Removed {(~valid_mask).sum()} points with inf values")
        print(f"  → Remaining: {len(points):,} points")
    
    if has_nan:
        nan_count = np.isnan(points).sum()
        print(f"⚠️ WARNING: {nan_count} NaN values detected!")
        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]
        colors = colors[valid_mask]
        print(f"  → Removed {(~valid_mask).sum()} points with NaN values")
        print(f"  → Remaining: {len(points):,} points")
    
    if not has_inf and not has_nan:
        print("✓ All points are valid (no inf/nan)")
    
    # Statistics
    print("\nPoint Cloud Statistics:")
    print(f"  Total points: {len(points):,}")
    print(f"  X Range: [{points[:, 0].min():.6f}, {points[:, 0].max():.6f}]")
    print(f"  Y Range: [{points[:, 1].min():.6f}, {points[:, 1].max():.6f}]")
    print(f"  Z Range: [{points[:, 2].min():.6f}, {points[:, 2].max():.6f}]")
    
    # Color statistics
    unique_colors = len(np.unique(colors, axis=0))
    print(f"  Unique colors: {unique_colors:,}")
    print(f"  Sample colors (first 5): {colors[:min(5, len(colors))].tolist()}")
    
    # Apply height-based colorization if requested
    if colorize_by_height:
        print("\nApplying height-based colorization...")
        z_values = points[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        
        if z_max > z_min:
            z_norm = (z_values - z_min) / (z_max - z_min)
        else:
            z_norm = np.zeros_like(z_values)
        
        # カラーグラデーション: 青(低) -> 緑(中) -> 赤(高)
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 0] = np.clip(255 * z_norm, 0, 255).astype(np.uint8)  # Red
        colors[:, 1] = np.clip(255 * (1 - np.abs(2*z_norm - 1)), 0, 255).astype(np.uint8)  # Green
        colors[:, 2] = np.clip(255 * (1 - z_norm), 0, 255).astype(np.uint8)  # Blue
        print("✓ Colorization complete.")
    else:
        print("\nUsing original COLMAP colors.")
    
    # Write to PLY file
    print(f"\nWriting PLY file: {output_ply_path}")
    
    try:
        # ディレクトリ作成
        output_dir = os.path.dirname(output_ply_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
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
            write_count = 0
            for pt, color in zip(points, colors):
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
                write_count += 1
                
                if write_count % 100000 == 0:
                    print(f"  Written: {write_count:,} / {len(points):,} points...")
        
        output_size = os.path.getsize(output_ply_path)
        print(f"\n✓ PLY file saved: {output_ply_path}")
        print(f"  File size: {output_size / (1024*1024):.2f} MB")
        print(f"  Points written: {write_count:,}")
        
    except Exception as e:
        print(f"❌ Error: Failed to write PLY file: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("="*60)
    print("✓ Conversion Complete!")
    print("="*60)
    
    return True


# 使用例
if __name__ == "__main__":
    # 元の関数と同じインターフェース
    success = generate_pointcloud_from_colmap_bins_robust(
        colmap_sparse_dir="/kaggle/working/output/sparse/0",
        output_ply_path="/kaggle/working/output/point_cloud.ply",
        colorize_by_height=False
    )
    
    if success:
        print("\n✅ SUCCESS: PLY file generated successfully!")
    else:
        print("\n❌ FAILED: Could not generate PLY file")
