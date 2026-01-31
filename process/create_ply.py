def create_point_cloud_visualization(colmap_dir, output_path):
    """Create and save point cloud visualization from COLMAP output (CPU version)"""
    print("=== Creating Point Cloud Visualization (CPU) ===")
    
    import open3d as o3d
    
    # Path to points3D.bin
    points3d_path = Path(colmap_dir) / "sparse" / "0" / "points3D.bin"
    
    if not points3d_path.exists():
        print(f"Error: {points3d_path} does not exist")
        return None
    
    # Read the COLMAP binary file
    with open(points3d_path, 'rb') as f:
        # Read the number of points (uint64)
        num_points = struct.unpack('Q', f.read(8))[0]
        
        if num_points == 0:
            print("Warning: No 3D points found in points3D.bin")
            print("Creating empty point cloud file")

            pcd = o3d.geometry.PointCloud()
            o3d.io.write_point_cloud(str(output_path), pcd)
            print(f"Empty point cloud saved to {output_path}")
            return pcd
        
        print(f"Reading {num_points} points...")
        
        points = []
        colors = []
        
        for _ in range(num_points):
            # Read point data
            point_id = struct.unpack('Q', f.read(8))[0]
            xyz = struct.unpack('ddd', f.read(24))
            rgb = struct.unpack('BBB', f.read(3))
            error = struct.unpack('d', f.read(8))[0]
            
            # Read track
            track_length = struct.unpack('Q', f.read(8))[0]
            for _ in range(track_length):
                image_id = struct.unpack('I', f.read(4))[0]
                point2D_idx = struct.unpack('I', f.read(4))[0]
            
            points.append(xyz)
            colors.append([c / 255.0 for c in rgb])
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # Save to PLY
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Point cloud saved to {output_path}")
    print(f"Total points: {len(points)}")
    
    # Visualize
    try:
        print("Displaying point cloud...")
        o3d.visualization.draw_geometries([pcd])
    except Exception as e:
        print(f"Note: Could not display visualization: {e}")
    
    return pcd
# Example Usage
output_ply = '/kaggle/working/output/point_cloud.ply'
create_point_cloud_visualization(colmap_dir, output_ply)
