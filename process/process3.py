
class StandaloneCOLMAPConverter:
    """
    Standalone COLMAP converter that doesn't depend on colmap_dataset_utils.
    Directly writes COLMAP binary format files from MASt3R output.
    """
    
    def __init__(self):
        pass
    
    def convert_mast3r_to_colmap(
        self,
        scene,
        output_dir: str,
        min_conf_thr: float = 2.0,
        clean_depth: bool = False,
        mask_images: bool = True,
        verbose: bool = True
    ) -> str:
        """
        Convert MASt3R scene to COLMAP format without external dependencies.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        sparse_dir = output_path / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        depth_dir = output_path / "stereo" / "depth_maps"
        depth_dir.mkdir(parents=True, exist_ok=True)
        
        normal_dir = output_path / "stereo" / "normal_maps"
        normal_dir.mkdir(parents=True, exist_ok=True)
        
        if mask_images:
            mask_dir = output_path / "stereo" / "confidence_maps"
            mask_dir.mkdir(parents=True, exist_ok=True)
        else:
            mask_dir = None
        
        if verbose:
            print(f"Converting MASt3R scene to COLMAP format...")
            print(f"Output directory: {output_dir}")
        
        # Extract camera parameters and poses from scene
        cameras, images_data, points3D = self._extract_from_scene(
            scene, min_conf_thr, verbose
        )
        
        if verbose:
            print(f"Extracted {len(cameras)} cameras")
            print(f"Extracted {len(images_data)} images")
            print(f"Extracted {len(points3D)} 3D points")
        
        # Save images and depth/normal maps
        self._save_image_data(
            scene, images_dir, depth_dir, normal_dir, mask_dir,
            min_conf_thr, verbose
        )
        
        # Write COLMAP binary files
        self._write_cameras_binary(cameras, sparse_dir / "cameras.bin")
        self._write_images_binary(images_data, sparse_dir / "images.bin")
        self._write_points3D_binary(points3D, sparse_dir / "points3D.bin")
        
        if verbose:
            print(f"✓ COLMAP conversion completed")
            print(f"  Sparse model: {sparse_dir}")
            print(f"  Images: {images_dir}")
            print(f"  Depth maps: {depth_dir}")
            print(f"  Normal maps: {normal_dir}")
        
        return str(output_path)
    
    def _extract_from_scene(self, scene, min_conf_thr: float, verbose: bool):
        """Extract camera parameters, image data, and 3D points from MASt3R scene."""
        
        cameras = {}
        images_data = {}
        points3D = {}
        
        num_images = len(scene.imgs)
        
        for idx in range(num_images):
            # Get image info
            img = scene.imgs[idx]
            h, w = img.shape[:2]
            
            # Get camera intrinsics
            camera_id = 1
            
            if camera_id not in cameras:
                focal_length = max(w, h) * 1.2
                cx = w / 2.0
                cy = h / 2.0
                
                cameras[camera_id] = {
                    'id': camera_id,
                    'model': 'PINHOLE',
                    'width': w,
                    'height': h,
                    'params': np.array([focal_length, focal_length, cx, cy])
                }
            
            # Get camera pose
            pts3d = scene.get_pts3d(idx)
            confidence = scene.get_conf(idx)
            
            pose = self._estimate_camera_pose(pts3d, confidence, min_conf_thr)
            qvec, tvec = self._matrix_to_quaternion_translation(pose)
            
            image_name = f"image_{idx:04d}.jpg"
            
            images_data[idx + 1] = {
                'id': idx + 1,
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': image_name,
                'xys': np.array([]),
                'point3D_ids': np.array([])
            }
        
        # Extract 3D points
        points3D = self._extract_3d_points(scene, min_conf_thr, verbose)
        
        return cameras, images_data, points3D
    
    def _estimate_camera_pose(self, pts3d: np.ndarray, confidence: np.ndarray, min_conf_thr: float):
        """Estimate camera pose from 3D points."""
        mask = confidence > min_conf_thr
        valid_pts = pts3d[mask]
        
        if len(valid_pts) < 4:
            return np.eye(4)
        
        center = np.median(valid_pts, axis=0)
        pose = np.eye(4)
        pose[:3, 3] = -center
        
        return pose
    
    def _matrix_to_quaternion_translation(self, matrix: np.ndarray):
        """Convert 4x4 transformation matrix to quaternion and translation."""
        R = matrix[:3, :3]
        t = matrix[:3, 3]
        
        qw = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
        qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
        
        qvec = np.array([qw, qx, qy, qz])
        return qvec, t
    
    def _extract_3d_points(self, scene, min_conf_thr: float, verbose: bool):
        """Extract 3D points from scene."""
        points3D = {}
        point_id = 1
        
        num_images = len(scene.imgs)
        
        for idx in range(num_images):
            pts3d = scene.get_pts3d(idx)
            confidence = scene.get_conf(idx)
            img = scene.imgs[idx]
            
            h, w = pts3d.shape[:2]
            pts_flat = pts3d.reshape(-1, 3)
            conf_flat = confidence.reshape(-1)
            
            if len(img.shape) == 3:
                colors = img.reshape(-1, 3)
            else:
                colors = np.stack([img.reshape(-1)] * 3, axis=1)
            
            mask = conf_flat > min_conf_thr
            
            if mask.sum() > 10000:
                indices = np.where(mask)[0]
                sampled_indices = np.random.choice(indices, size=10000, replace=False)
                mask = np.zeros_like(mask, dtype=bool)
                mask[sampled_indices] = True
            
            valid_pts = pts_flat[mask]
            valid_colors = colors[mask]
            
            for pt, color in zip(valid_pts, valid_colors):
                points3D[point_id] = {
                    'id': point_id,
                    'xyz': pt,
                    'rgb': color.astype(np.uint8),
                    'error': 0.0,
                    'image_ids': np.array([idx + 1]),
                    'point2D_idxs': np.array([0])
                }
                point_id += 1
        
        return points3D
    
    def _save_image_data(self, scene, images_dir, depth_dir, normal_dir, mask_dir, min_conf_thr, verbose):
        """Save images, depth maps, normal maps, and confidence masks."""
        num_images = len(scene.imgs)
        
        for idx in range(num_images):
            image_name = f"image_{idx:04d}.jpg"
            
            # Save image
            img = scene.imgs[idx]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            cv2.imwrite(str(images_dir / image_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Save depth map
            pts3d = scene.get_pts3d(idx)
            depth = np.linalg.norm(pts3d, axis=2)
            depth_name = image_name.replace('.jpg', '.geometric.bin')
            self._save_depth_map(depth, depth_dir / depth_name)
            
            # Save normal map
            normals = self._compute_normals_from_depth(pts3d)
            normal_name = image_name.replace('.jpg', '.geometric.bin')
            self._save_normal_map(normals, normal_dir / normal_name)
            
            # Save confidence mask
            if mask_dir is not None:
                confidence = scene.get_conf(idx)
                mask = (confidence > min_conf_thr).astype(np.uint8) * 255
                mask_name = image_name.replace('.jpg', '.png')
                cv2.imwrite(str(mask_dir / mask_name), mask)
        
        if verbose:
            print(f"Saved {num_images} images with depth/normal maps")
    
    def _compute_normals_from_depth(self, pts3d: np.ndarray):
        """Compute surface normals from 3D points."""
        h, w = pts3d.shape[:2]
        normals = np.zeros_like(pts3d)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                px = pts3d[i, j + 1] - pts3d[i, j - 1]
                py = pts3d[i + 1, j] - pts3d[i - 1, j]
                normal = np.cross(px, py)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normals[i, j] = normal / norm
        
        return normals
    
    def _save_depth_map(self, depth: np.ndarray, path: Path):
        """Save depth map in COLMAP binary format."""
        h, w = depth.shape
        
        with open(path, 'wb') as f:
            f.write(struct.pack('i', w))
            f.write(struct.pack('i', h))
            f.write(struct.pack('i', 1))
            depth_flat = depth.astype(np.float32).flatten()
            f.write(depth_flat.tobytes())
    
    def _save_normal_map(self, normals: np.ndarray, path: Path):
        """Save normal map in COLMAP binary format."""
        h, w = normals.shape[:2]
        
        with open(path, 'wb') as f:
            f.write(struct.pack('i', w))
            f.write(struct.pack('i', h))
            f.write(struct.pack('i', 3))
            normals_flat = normals.astype(np.float32).reshape(-1)
            f.write(normals_flat.tobytes())
    
    def _write_cameras_binary(self, cameras: Dict, path: Path):
        """Write cameras.bin in COLMAP binary format."""
        with open(path, 'wb') as f:
            f.write(struct.pack('Q', len(cameras)))
            
            for camera in cameras.values():
                f.write(struct.pack('i', camera['id']))
                f.write(struct.pack('i', 1))  # PINHOLE = 1
                f.write(struct.pack('Q', camera['width']))
                f.write(struct.pack('Q', camera['height']))
                
                for param in camera['params']:
                    f.write(struct.pack('d', param))
    
    def _write_images_binary(self, images: Dict, path: Path):
        """Write images.bin in COLMAP binary format."""
        with open(path, 'wb') as f:
            f.write(struct.pack('Q', len(images)))
            
            for img in images.values():
                f.write(struct.pack('i', img['id']))
                
                for q in img['qvec']:
                    f.write(struct.pack('d', q))
                
                for t in img['tvec']:
                    f.write(struct.pack('d', t))
                
                f.write(struct.pack('i', img['camera_id']))
                
                name_bytes = img['name'].encode('utf-8') + b'\x00'
                f.write(name_bytes)
                
                f.write(struct.pack('Q', len(img['xys'])))
                for xy, p3d_id in zip(img['xys'], img['point3D_ids']):
                    f.write(struct.pack('dd', xy[0], xy[1]))
                    f.write(struct.pack('Q', p3d_id))
    
    def _write_points3D_binary(self, points3D: Dict, path: Path):
        """Write points3D.bin in COLMAP binary format."""
        with open(path, 'wb') as f:
            f.write(struct.pack('Q', len(points3D)))
            
            for pt in points3D.values():
                f.write(struct.pack('Q', pt['id']))
                
                for coord in pt['xyz']:
                    f.write(struct.pack('d', coord))
                
                for c in pt['rgb']:
                    f.write(struct.pack('B', c))
                
                f.write(struct.pack('d', pt['error']))
                
                f.write(struct.pack('Q', len(pt['image_ids'])))
                for img_id, pt2d_idx in zip(pt['image_ids'], pt['point2D_idxs']):
                    f.write(struct.pack('i', img_id))
                    f.write(struct.pack('i', pt2d_idx))



'''

from process3_standalone import main_pipeline_process3_standalone

gs_model = main_pipeline_process3_standalone(
    image_dir="/kaggle/input/two-dogs/bike15",
    output_dir="/kaggle/working/output",
    square_size=512,
    iterations=30000,
    max_images=None,
    min_conf_thr=2.0,
    clean_depth=False,
    mask_images=True,
    verbose=True
)

'''
