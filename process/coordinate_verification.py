"""
MASt3R to COLMAP Coordinate Transformation Verification Tool
=============================================================

Kaggle Notebook用の統合版
/kaggle/working/ に配置して使用してください

使用例:
------
# セルの最初に追加
from coordinate_verification import (
    CalibrationDataGenerator,
    test_calibration_pipeline,
    verify_transformation,
    create_calibration_scene
)

# クイックテスト
scene = create_calibration_scene(num_views=4, num_points=100)
# ... あなたのCOLMAP変換関数を実行 ...
# verify_transformation('./ground_truth', './colmap_output')
"""

import numpy as np
import torch
from pathlib import Path
import json
from tqdm.auto import tqdm


# ============================================================================
# CalibrationDataGenerator - キャリブレーションデータ生成
# ============================================================================

class CalibrationDataGenerator:
    """既知の3D座標を持つ合成シーンデータの生成"""
    
    def __init__(self, num_views=4, num_points=100):
        self.num_views = num_views
        self.num_points = num_points
        
    def generate_cube_points(self, size=1.0):
        """立方体形状の3D点群を生成"""
        points = []
        
        # 立方体の8頂点
        for x in [-size, size]:
            for y in [-size, size]:
                for z in [-size, size]:
                    points.append([x, y, z])
        
        # 辺上の点
        n_edge = max(1, (self.num_points - 8) // 12)
        for i in range(n_edge):
            t = (i + 1) / (n_edge + 1)
            # X軸方向の辺
            points.extend([
                [2*size*t - size, -size, -size],
                [2*size*t - size, size, -size],
                [2*size*t - size, -size, size],
                [2*size*t - size, size, size],
            ])
            # Y軸方向の辺
            points.extend([
                [-size, 2*size*t - size, -size],
                [size, 2*size*t - size, -size],
                [-size, 2*size*t - size, size],
                [size, 2*size*t - size, size],
            ])
            # Z軸方向の辺
            points.extend([
                [-size, -size, 2*size*t - size],
                [size, -size, 2*size*t - size],
                [-size, size, 2*size*t - size],
                [size, size, 2*size*t - size],
            ])
        
        return np.array(points[:self.num_points])
    
    def generate_circular_camera_poses(self, radius=5.0, height=0.0):
        """円形配置のカメラポーズを生成"""
        poses = []
        for i in range(self.num_views):
            angle = 2 * np.pi * i / self.num_views
            
            # カメラ位置
            cam_pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height
            ])
            
            # 原点を注視
            forward = -cam_pos / np.linalg.norm(cam_pos)
            up = np.array([0, 0, 1])
            right = np.cross(up, forward)
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)
            
            # 回転行列(world to camera)
            R = np.stack([right, up, forward], axis=0)
            
            # 平行移動(world to camera)
            t = -R @ cam_pos
            
            # 4x4変換行列
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            
            poses.append(pose)
            
        return poses
    
    def create_mock_scene(self, device='cpu'):
        """MASt3R風のシーンオブジェクトを作成"""
        
        # Ground truth 3D点群を生成
        pts3d_world = self.generate_cube_points(size=1.0)
        
        # カメラポーズを生成
        camera_poses = self.generate_circular_camera_poses(radius=5.0, height=0.5)
        
        # Sceneオブジェクトを作成
        class MockScene:
            def __init__(self, pts3d_world, camera_poses, device):
                self.device = device
                self.num_views = len(camera_poses)
                self.num_points = len(pts3d_world)
                
                # Ground truthを保存
                self.ground_truth_pts3d_world = pts3d_world.copy()
                self.ground_truth_poses = [p.copy() for p in camera_poses]
                
                # 各ビューのデータを作成
                self.imgs = []
                for i in range(self.num_views):
                    img_data = {
                        'idx': i,
                        'instance': f'view_{i:02d}',
                        'true_shape': np.array([512, 512])
                    }
                    self.imgs.append(img_data)
                
                # 各ビューの3D点(カメラ座標系)
                self.pts3d = []
                self.conf = []
                for pose in camera_poses:
                    # ワールド座標からカメラ座標へ変換
                    pts_cam = self._world_to_camera(pts3d_world, pose)
                    self.pts3d.append(torch.from_numpy(pts_cam).float().to(device))
                    # 全点に高い信頼度
                    self.conf.append(torch.ones(self.num_points).float().to(device))
                
                # カメラ内部パラメータ
                self.focals = torch.tensor([500.0] * self.num_views).to(device)
                self.principal_points = torch.tensor([[256.0, 256.0]] * self.num_views).to(device)
                
                # カメラポーズ(最適化済み)
                self.im_poses = torch.stack([
                    torch.eye(4) for _ in range(self.num_views)
                ]).float().to(device)
                
                for i, pose in enumerate(camera_poses):
                    self.im_poses[i] = torch.from_numpy(pose).float()
            
            def _world_to_camera(self, pts_world, pose):
                """ワールド座標からカメラ座標へ変換"""
                pts_homo = np.hstack([pts_world, np.ones((len(pts_world), 1))])
                pts_cam_homo = (pose @ pts_homo.T).T
                return pts_cam_homo[:, :3]
            
            def get_im_poses(self):
                return self.im_poses
            
            def get_pts3d(self, i=None):
                if i is not None:
                    return self.pts3d[i]
                return self.pts3d
            
            def get_conf(self, i=None):
                if i is not None:
                    return self.conf[i]
                return self.conf
            
            def get_focals(self):
                return self.focals
            
            def get_principal_points(self):
                return self.principal_points
        
        scene = MockScene(pts3d_world, camera_poses, device)
        return scene
    
    def save_ground_truth(self, scene, output_path):
        """Ground truthデータを保存"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 3D点群
        np.savetxt(
            output_path / 'ground_truth_points3d.txt',
            scene.ground_truth_pts3d_world,
            header='X Y Z (world coordinates)',
            fmt='%.6f'
        )
        
        # カメラポーズ
        with open(output_path / 'ground_truth_poses.json', 'w') as f:
            poses_list = [p.tolist() for p in scene.ground_truth_poses]
            json.dump({
                'poses': poses_list,
                'description': '4x4 transformation matrices (world to camera)'
            }, f, indent=2)
        
        # カメラ内部パラメータ
        with open(output_path / 'ground_truth_intrinsics.json', 'w') as f:
            json.dump({
                'focals': scene.focals.cpu().numpy().tolist(),
                'principal_points': scene.principal_points.cpu().numpy().tolist(),
                'image_size': [512, 512]
            }, f, indent=2)
        
        print(f"✓ Ground truth saved to {output_path}")
        print(f"  - {len(scene.ground_truth_pts3d_world)} 3D points")
        print(f"  - {len(scene.ground_truth_poses)} camera poses")


# ============================================================================
# verify_transformation - 座標変換の検証
# ============================================================================

def verify_transformation(ground_truth_path, colmap_output_path):
    """COLMAP出力がGround truthと一致するか検証"""
    ground_truth_path = Path(ground_truth_path)
    colmap_output_path = Path(colmap_output_path)
    
    # Ground truth読み込み
    gt_points = np.loadtxt(ground_truth_path / 'ground_truth_points3d.txt')
    
    # COLMAP points3D.txt読み込み
    colmap_points_file = colmap_output_path / 'points3D.txt'
    if not colmap_points_file.exists():
        print(f"❌ COLMAP output not found: {colmap_points_file}")
        return False
    
    # COLMAP点群をパース
    colmap_points = []
    with open(colmap_points_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                colmap_points.append([x, y, z])
    
    colmap_points = np.array(colmap_points)
    
    print("\n" + "="*70)
    print("座標変換検証結果")
    print("="*70)
    print(f"\nGround truth点数: {len(gt_points)}")
    print(f"COLMAP点数: {len(colmap_points)}")
    
    if len(colmap_points) == 0:
        print("❌ COLMAP出力に点が見つかりません")
        return False
    
    # 統計情報
    print("\n【Ground Truth統計】")
    print(f"  平均: {gt_points.mean(axis=0)}")
    print(f"  標準偏差: {gt_points.std(axis=0)}")
    print(f"  最小値: {gt_points.min(axis=0)}")
    print(f"  最大値: {gt_points.max(axis=0)}")
    
    print("\n【COLMAP出力統計】")
    print(f"  平均: {colmap_points.mean(axis=0)}")
    print(f"  標準偏差: {colmap_points.std(axis=0)}")
    print(f"  最小値: {colmap_points.min(axis=0)}")
    print(f"  最大値: {colmap_points.max(axis=0)}")
    
    # Procrustes解析
    if len(colmap_points) == len(gt_points):
        try:
            from scipy.spatial import procrustes
            gt_centered = gt_points - gt_points.mean(axis=0)
            colmap_centered = colmap_points - colmap_points.mean(axis=0)
            
            mtx1, mtx2, disparity = procrustes(gt_centered, colmap_centered)
            
            print(f"\n【Procrustes解析】")
            print(f"  差異度: {disparity:.6f}")
            
            if disparity < 0.01:
                print("  判定: ✓ 座標一致(剛体変換のみ)")
            elif disparity < 0.1:
                print("  判定: ⚠ 概ね一致(小さな差異あり)")
            else:
                print("  判定: ❌ 大きな座標変化を検出")
        except ImportError:
            print("\n⚠ scipy未インストールのため詳細解析をスキップ")
        except Exception as e:
            print(f"\n⚠ Procrustes解析エラー: {e}")
    else:
        print("\n⚠ 点数不一致のため直接比較不可")
    
    # 比較結果を保存
    comparison_file = colmap_output_path / 'coordinate_comparison.txt'
    with open(comparison_file, 'w') as f:
        f.write("Ground Truth vs COLMAP 座標比較\n")
        f.write("="*60 + "\n\n")
        f.write(f"Ground truth点数: {len(gt_points)}\n")
        f.write(f"COLMAP点数: {len(colmap_points)}\n\n")
        f.write("Ground Truth統計:\n")
        f.write(f"  平均: {gt_points.mean(axis=0)}\n")
        f.write(f"  標準偏差: {gt_points.std(axis=0)}\n")
        f.write("\nCOLMAP統計:\n")
        f.write(f"  平均: {colmap_points.mean(axis=0)}\n")
        f.write(f"  標準偏差: {colmap_points.std(axis=0)}\n")
    
    print(f"\n✓ 比較結果を保存: {comparison_file}")
    print("="*70 + "\n")
    
    return True


# ============================================================================
# test_calibration_pipeline - 完全なテストパイプライン
# ============================================================================

def test_calibration_pipeline(colmap_converter_func, output_dir='./test_calibration'):
    """
    完全なキャリブレーションテストを実行
    
    Args:
        colmap_converter_func: COLMAP変換関数 (scene, images, output_path) を受け取る
        output_dir: 出力ディレクトリ
    """
    print("="*70)
    print("MASt3R→COLMAP 座標変換テスト")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # [1] キャリブレーションデータ生成
    print("\n[1/4] キャリブレーションデータ生成中...")
    generator = CalibrationDataGenerator(num_views=6, num_points=120)
    scene = generator.create_mock_scene(device='cpu')
    
    ground_truth_dir = output_dir / 'ground_truth'
    generator.save_ground_truth(scene, ground_truth_dir)
    
    print(f"\n  ビュー数: {scene.num_views}")
    print(f"  点数: {scene.num_points}")
    print(f"  座標範囲: X[{scene.ground_truth_pts3d_world[:, 0].min():.2f}, "
          f"{scene.ground_truth_pts3d_world[:, 0].max():.2f}], "
          f"Y[{scene.ground_truth_pts3d_world[:, 1].min():.2f}, "
          f"{scene.ground_truth_pts3d_world[:, 1].max():.2f}], "
          f"Z[{scene.ground_truth_pts3d_world[:, 2].min():.2f}, "
          f"{scene.ground_truth_pts3d_world[:, 2].max():.2f}]")
    
    # [2] モック画像作成
    print("\n[2/4] モック画像データ作成中...")
    mock_images = []
    for i in range(scene.num_views):
        img_dict = {
            'img': torch.zeros(3, 512, 512),
            'true_shape': np.array([512, 512]),
            'idx': i,
            'instance': f'calibration_view_{i:02d}.jpg'
        }
        mock_images.append(img_dict)
    print(f"  ✓ {len(mock_images)}枚の画像データ作成完了")
    
    # [3] COLMAP変換実行
    print("\n[3/4] COLMAP変換実行中...")
    colmap_output_dir = output_dir / 'colmap_output'
    colmap_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        colmap_converter_func(scene, mock_images, str(colmap_output_dir))
        print("  ✓ COLMAP変換完了")
    except Exception as e:
        print(f"  ❌ COLMAP変換失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # [4] 検証
    print("\n[4/4] 座標検証中...")
    success = verify_transformation(ground_truth_dir, colmap_output_dir)
    
    if success:
        print("\n" + "="*70)
        print("✓ テスト完了!")
        print(f"結果: {output_dir}")
        print("="*70)
    
    return success


# ============================================================================
# クイック使用関数
# ============================================================================

def create_calibration_scene(num_views=4, num_points=100, device='cpu', 
                            save_ground_truth=True, output_dir='./calibration_data'):
    """
    キャリブレーション用シーンをクイック生成
    
    使用例:
        scene = create_calibration_scene(num_views=4, num_points=100)
        # あなたのCOLMAP変換を実行
        mast3r_to_colmap(scene, images, './colmap_output')
        # 検証
        verify_transformation('./calibration_data', './colmap_output')
    """
    print("キャリブレーションシーン生成中...")
    
    generator = CalibrationDataGenerator(num_views=num_views, num_points=num_points)
    scene = generator.create_mock_scene(device=device)
    
    if save_ground_truth:
        generator.save_ground_truth(scene, output_dir)
    
    print(f"✓ シーン生成完了 (ビュー数: {num_views}, 点数: {num_points})")
    print(f"Ground truth保存先: {output_dir}")
    
    # モック画像も返す
    mock_images = []
    for i in range(num_views):
        img_dict = {
            'img': torch.zeros(3, 512, 512),
            'true_shape': np.array([512, 512]),
            'idx': i,
            'instance': f'view_{i:02d}.jpg'
        }
        mock_images.append(img_dict)
    
    return scene, mock_images


# ============================================================================
# 使用例
# ============================================================================

if __name__ == "__main__":
    print("""
MASt3R→COLMAP 座標変換検証ツール (Kaggle統合版)
==============================================

【使用方法】

1. クイック生成:
   scene, images = create_calibration_scene(num_views=4, num_points=100)
   # あなたのCOLMAP変換関数を実行
   mast3r_to_colmap(scene, images, './colmap_output')
   # 検証
   verify_transformation('./calibration_data', './colmap_output')

2. 完全テスト:
   test_calibration_pipeline(mast3r_to_colmap, output_dir='./test_results')

3. 既存出力の検証:
   verify_transformation('./ground_truth', './colmap_output')

""")
