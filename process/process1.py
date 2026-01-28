
# ===== Traditional Method: extract_colmap_data =====
def extract_colmap_data_traditional(scene, image_paths, max_points=1000000):
    """
    Traditional Method: Extract COLMAP-compatible data from a MASt3R scene.
    (Derived from dino-mast3r-gs-kg-34oo.ipynb)
    """
    print("\n=== [TRADITIONAL] Extracting COLMAP-compatible data ===")

    # Extract point cloud
    pts_all = scene.get_pts3d()
    print(f"pts_all type: {type(pts_all)}")

    if isinstance(pts_all, list):
        print(f"pts_all is a list with {len(pts_all)} elements")
        if len(pts_all) > 0:
            print(f"First element type: {type(pts_all[0])}")
            if hasattr(pts_all[0], 'shape'):
                print(f"First element shape: {pts_all[0].shape}")

        pts_all = torch.stack([p if isinstance(p, torch.Tensor) else torch.tensor(p)
                              for p in pts_all])
        print(f"pts_all shape after conversion: {pts_all.shape}")

    if len(pts_all.shape) == 4:
        print(f"Found batched point cloud: {pts_all.shape}")
        B, H, W, _ = pts_all.shape
        pts3d = pts_all.reshape(-1, 3).detach().cpu().numpy()

        # Extract colors
        colors = []
        for img_path in image_paths:
            img = Image.open(img_path).resize((W, H))
            colors.append(np.array(img))
        colors = np.stack(colors).reshape(-1, 3) / 255.0
    else:
        pts3d = pts_all.detach().cpu().numpy() if isinstance(pts_all, torch.Tensor) else pts_all
        colors = np.ones((len(pts3d), 3)) * 0.5

    print(f"✓ Extracted {len(pts3d)} 3D points from {len(image_paths)} images")

    # Downsample points
    if len(pts3d) > max_points:
        print(f"\n⚠ Downsampling from {len(pts3d)} to {max_points} points...")
        valid_mask = ~(np.isnan(pts3d).any(axis=1) | np.isinf(pts3d).any(axis=1))
        pts3d_valid = pts3d[valid_mask]
        colors_valid = colors[valid_mask]
        
        # Count excluded points
        num_excluded = len(pts3d_valid) - max_points
        
        indices = np.random.choice(len(pts3d_valid), size=max_points, replace=False)
        pts3d = pts3d_valid[indices]
        colors = colors_valid[indices]
        print(f"✓ Downsampled to {len(pts3d)} points")
        print(f"⚠ Excluded {num_excluded} points due to max_points limit")

    # Extract camera parameters
    print("Extracting camera parameters...")

    # [Important] Convert camera-to-world (C2W) to world-to-camera (W2C)
    poses_c2w = scene.get_im_poses().detach().cpu().numpy()
    print(f"Retrieved camera-to-world poses: shape {poses_c2w.shape}")

    poses = []
    for i, pose_c2w in enumerate(poses_c2w):
        pose_w2c = np.linalg.inv(pose_c2w)
        poses.append(pose_w2c)
    poses = np.array(poses)
    print("Converted to world-to-camera poses for COLMAP")

    focals = scene.get_focals().detach().cpu().numpy()
    pp = scene.get_principal_points().detach().cpu().numpy()
    print(f"Focals shape: {focals.shape}")
    print(f"Principal points shape: {pp.shape}")

    mast3r_size = 224.0

    cameras = []
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        W, H = img.size
        scale = W / mast3r_size

        if focals.shape[1] == 1:
            focal_mast3r = float(focals[i, 0])
            fx = fy = focal_mast3r * scale
        else:
            fx = float(focals[i, 0]) * scale
            fy = float(focals[i, 1]) * scale

        cx = float(pp[i, 0]) * scale
        cy = float(pp[i, 1]) * scale

        camera = {
            'camera_id': i + 1,
            'model': 'PINHOLE',
            'width': W,
            'height': H,
            'params': [fx, fy, cx, cy]
        }
        cameras.append(camera)

        if i == 0:
            print(f"\nExample camera 0:")
            print(f"  Image size: {W}x{H}")
            print(f"  MASt3R focal: {focal_mast3r:.2f}, pp: ({pp[i,0]:.2f}, {pp[i,1]:.2f})")
            print(f"  Scaled fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            print(f"  Pose (first row): {poses[i][0]}")

    print(f"\n✓ Extracted {len(cameras)} cameras and {len(poses)} poses")

    pts3d = pts3d.reshape(-1, 3)
    colors = np.ones((len(pts3d), 3)) * 0.5
    
    return pts3d, colors, cameras, poses


# ===== Traditional Method: rotmat2qvec =====
def rotmat2qvec_traditional(R):
    """Traditional Method: Convert rotation matrix to quaternion."""
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


# ===== Traditional Method: Save Functions =====
def write_cameras_binary_traditional(cameras, output_file):
    """Traditional Method: Write cameras.bin."""
    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', len(cameras)))

        for i, cam in enumerate(cameras):
            camera_id = cam.get('camera_id', i + 1)
            model_id = 1  # PINHOLE
            width = cam['width']
            height = cam['height']
            params = cam['params']

            f.write(struct.pack('i', camera_id))
            f.write(struct.pack('i', model_id))
            f.write(struct.pack('Q', width))
            f.write(struct.pack('Q', height))

            for param in params[:4]:
                f.write(struct.pack('d', param))


def write_images_binary_traditional(image_paths, cameras, poses, output_file):
    """Traditional Method: Write images.bin."""
    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', len(image_paths)))

        for i, (img_path, pose) in enumerate(zip(image_paths, poses)):
            image_id = i + 1
            camera_id = cameras[i].get('camera_id', i + 1)
            image_name = os.path.basename(img_path)

            R = pose[:3, :3]
            t = pose[:3, 3]
            qvec = rotmat2qvec_traditional(R)
            tvec = t

            f.write(struct.pack('i', image_id))
            for q in qvec:
                f.write(struct.pack('d', float(q)))
            for tv in tvec:
                f.write(struct.pack('d', float(tv)))
            f.write(struct.pack('i', camera_id))
            f.write(image_name.encode('utf-8') + b'\x00')
            f.write(struct.pack('Q', 0))


def write_points3d_binary_traditional(pts3d, colors, output_file):
    """Traditional Method: Write points3D.bin."""
    valid_indices = []
    invalid_count = 0
    
    for i, pt in enumerate(pts3d):
        if not (np.isnan(pt).any() or np.isinf(pt).any()):
            valid_indices.append(i)
        else:
            invalid_count += 1

    with open(output_file, 'wb') as f:
        f.write(struct.pack('Q', len(valid_indices)))

        for idx, point_id in enumerate(valid_indices):
            pt = pts3d[point_id]
            color = colors[point_id]

            f.write(struct.pack('Q', point_id))
            for coord in np.asarray(pt).ravel(): 
                    f.write(struct.pack('d', float(coord)))

            col_int = (color * 255).astype(np.uint8)
            for c in col_int:
                f.write(struct.pack('B', int(c)))

            f.write(struct.pack('d', 0.0))
            f.write(struct.pack('Q', 0))

    if invalid_count > 0:
        print(f"  ⚠ Excluded {invalid_count} invalid points (NaN/Inf)")

    return len(valid_indices)


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].
    """
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
    
    return np.array([w, x, y, z])


def save_colmap_reconstruction_traditional(pts3d, colors, cameras, poses, colmap_dir):
    """
    Save COLMAP reconstruction in traditional format.
    
    Args:
        pts3d: 3D points array (N, 3)
        colors: Color array (N, 3)
        cameras: List of camera dictionaries with intrinsics
        poses: Camera poses (world-to-camera)
        colmap_dir: Directory to save COLMAP files
    """
    # 修正：output_dir を colmap_dir に変更
    output_path = os.path.join(colmap_dir, "sparse", "0")
    os.makedirs(output_path, exist_ok=True)
    
    # Write cameras.txt
    with open(os.path.join(output_path, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam in cameras:
            params_str = " ".join(map(str, cam['params']))
            f.write(f"{cam['camera_id']} {cam['model']} {cam['width']} {cam['height']} {params_str}\n")
    
    # Write images.txt
    with open(os.path.join(output_path, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, pose in enumerate(poses):
            # Convert rotation matrix to quaternion
            R = pose[:3, :3]
            t = pose[:3, 3]
            quat = rotation_matrix_to_quaternion(R)
            
            f.write(f"{i+1} {quat[0]} {quat[1]} {quat[2]} {quat[3]} ")
            f.write(f"{t[0]} {t[1]} {t[2]} {i+1} image_{i:04d}.jpg\n")
            f.write("\n")  # Empty line for POINTS2D
    
    # Write points3D.txt
    with open(os.path.join(output_path, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, (pt, color) in enumerate(zip(pts3d, colors)):
            r, g, b = (color * 255).astype(int)
            f.write(f"{i+1} {pt[0]} {pt[1]} {pt[2]} {r} {g} {b} 0.0\n")
    
    print(f"✓ Saved COLMAP reconstruction to {output_path}")
    return output_path
