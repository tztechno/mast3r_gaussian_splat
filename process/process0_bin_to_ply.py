

!pip install pycolmap

def convert_colmap_bin_to_ply(sparse_dir, output_ply_path):
    """
    Generate a PLY file from COLMAP binary files using pycolmap.
    
    Args:
        sparse_dir: Path to the sparse/0 directory.
        output_ply_path: Path where the output PLY file will be saved.
    """
    import pycolmap
    from plyfile import PlyData, PlyElement
    import numpy as np
    
    print(f"\n=== Converting COLMAP bin to PLY ===")
    
    # Load the reconstruction using pycolmap
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    
    print(f"Loaded reconstruction:")
    print(f"  - {len(reconstruction.cameras)} cameras")
    print(f"  - {len(reconstruction.images)} images")
    print(f"  - {len(reconstruction.points3D)} points")
    
    if len(reconstruction.points3D) == 0:
        print("❌ No 3D points found in reconstruction!")
        return 0
    
    # Extract 3D points and colors
    points = []
    colors = []
    
    for point3D_id, point3D in reconstruction.points3D.items():
        points.append(point3D.xyz)
        colors.append(point3D.color)
    
    points = np.array(points)
    colors = np.array(colors)
    
    print(f"\nPoint cloud statistics:")
    print(f"  Total points: {len(points)}")
    print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # Save as a PLY file
    vertices = np.array(
        [(p[0], p[1], p[2], c[0], c[1], c[2]) 
         for p, c in zip(points, colors)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(output_ply_path)
    
    print(f"✓ Saved PLY file to {output_ply_path}")
    
    return len(points)


'''
# Example usage:
colmap_output_dir = '/kaggle/working/output/colmap'
sparse_dir = os.path.join(colmap_output_dir, 'sparse', '0')
ply_path = os.path.join(colmap_output_dir, 'point_cloud.ply')
num_points = convert_colmap_bin_to_ply(sparse_dir, ply_path)
'''

#--------------------------------------------------------------

!pip install pycolmap

def convert_colmap_bin_to_ply(sparse_dir, output_ply_path):
    """
    pycolmapを使ってCOLMAP binファイルからPLYを生成
    
    Args:
        sparse_dir: sparse/0ディレクトリのパス
        output_ply_path: 出力PLYファイルのパス
    """
    import pycolmap
    from plyfile import PlyData, PlyElement
    import numpy as np
    
    print(f"\n=== Converting COLMAP bin to PLY ===")
    
    # pycolmapでreconstructionを読み込む
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    
    print(f"Loaded reconstruction:")
    print(f"  - {len(reconstruction.cameras)} cameras")
    print(f"  - {len(reconstruction.images)} images")
    print(f"  - {len(reconstruction.points3D)} points")
    
    if len(reconstruction.points3D) == 0:
        print("❌ No 3D points in reconstruction!")
        return 0
    
    # 3D点を抽出
    points = []
    colors = []
    
    for point3D_id, point3D in reconstruction.points3D.items():
        points.append(point3D.xyz)
        colors.append(point3D.color)
    
    points = np.array(points)
    colors = np.array(colors)
    
    print(f"\nPoint cloud statistics:")
    print(f"  Total points: {len(points)}")
    print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # PLYファイルとして保存
    vertices = np.array(
        [(p[0], p[1], p[2], c[0], c[1], c[2]) 
         for p, c in zip(points, colors)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(output_ply_path)
    
    print(f"✓ Saved PLY file to {output_ply_path}")
    
    return len(points)


'''
colmap_output_dir='/kaggle/working/output/colmap'
sparse_dir = os.path.join(colmap_output_dir, 'sparse', '0')
ply_path = os.path.join(colmap_output_dir, 'point_cloud.ply')
num_points = convert_colmap_bin_to_ply(sparse_dir, ply_path)
'''

