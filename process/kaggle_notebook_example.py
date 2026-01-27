"""
Kaggle Notebook での使用例
============================

このファイルの内容をKaggle Notebookの各セルにコピーして使用してください
"""

# ============================================================================
# セル 1: ファイルの配置と確認
# ============================================================================
"""
手順:
1. coordinate_verification.py を Kaggle の /kaggle/working/ にアップロード
2. このセルを実行してファイルの存在を確認
"""

import os
from pathlib import Path

# ファイルの存在確認
verification_file = Path('/kaggle/working/coordinate_verification.py')
if verification_file.exists():
    print("✓ coordinate_verification.py が見つかりました")
    print(f"  ファイルサイズ: {verification_file.stat().st_size} bytes")
else:
    print("❌ coordinate_verification.py が見つかりません")
    print("  /kaggle/working/ にファイルをアップロードしてください")


# ============================================================================
# セル 2: インポート
# ============================================================================

import sys
sys.path.insert(0, '/kaggle/working')

from coordinate_verification import (
    CalibrationDataGenerator,
    create_calibration_scene,
    verify_transformation,
    test_calibration_pipeline
)

import numpy as np
import torch

print("✓ モジュールのインポート完了")


# ============================================================================
# セル 3A: 方法1 - クイック使用(推奨)
# ============================================================================
"""
通常の使用フロー:
1. キャリブレーションシーンを生成
2. あなたのCOLMAP変換関数を実行
3. 結果を検証
"""

# キャリブレーションシーン生成
scene, images = create_calibration_scene(
    num_views=4,
    num_points=100,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_ground_truth=True,
    output_dir='/kaggle/working/calibration_data'
)

print("\nGround Truth座標(最初の5点):")
print(scene.ground_truth_pts3d_world[:5])

# ここであなたのCOLMAP変換関数を呼び出す
# 例: mast3r_to_colmap(scene, images, '/kaggle/working/colmap_output')

print("\n次のステップ:")
print("1. あなたのCOLMAP変換関数を実行")
print("2. verify_transformation()を実行して検証")


# ============================================================================
# セル 3B: 方法2 - 完全な自動テスト
# ============================================================================
"""
あなたのCOLMAP変換関数を自動でテストする場合
"""

# あなたのCOLMAP変換関数を定義
def your_colmap_converter(scene, images, output_path, masks=None):
    """
    あなたの既存のCOLMAP変換関数をここに配置
    """
    from pathlib import Path
    import torch
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 例: 簡単な実装(実際のコードに置き換えてください)
    
    # cameras.txt
    with open(output_path / 'cameras.txt', 'w') as f:
        f.write("# Camera list\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i in range(len(images)):
            focal = scene.get_focals()[i].item()
            pp = scene.get_principal_points()[i]
            f.write(f"{i+1} PINHOLE 512 512 {focal} {focal} {pp[0]} {pp[1]}\n")
    
    # images.txt
    with open(output_path / 'images.txt', 'w') as f:
        f.write("# Image list\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i in range(len(images)):
            pose = scene.get_im_poses()[i].cpu().numpy()
            # 回転行列をクォータニオンに変換(簡易版)
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0  # 実装を省略
            tx, ty, tz = pose[:3, 3]
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i+1} view_{i:02d}.jpg\n")
            f.write("\n")
    
    # points3D.txt
    with open(output_path / 'points3D.txt', 'w') as f:
        f.write("# 3D point list\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        
        # 各ビューの3D点を収集
        all_points = []
        for i in range(len(images)):
            pts = scene.get_pts3d(i)
            if torch.is_tensor(pts):
                pts = pts.cpu().numpy()
            # カメラ座標からワールド座標に変換(簡易版)
            # 実際にはpose行列の逆変換が必要
            all_points.append(pts)
        
        # 点を書き出し
        point_id = 1
        for pts in all_points:
            for pt in pts[:10]:  # 最初の10点のみ(例)
                f.write(f"{point_id} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 128 128 128 0.0\n")
                point_id += 1
    
    print(f"✓ COLMAP データを {output_path} に保存")

# テスト実行
test_calibration_pipeline(
    colmap_converter_func=your_colmap_converter,
    output_dir='/kaggle/working/test_results'
)


# ============================================================================
# セル 4: 検証のみ実行
# ============================================================================
"""
既にCOLMAP変換を実行済みの場合、検証のみを実行
"""

verify_transformation(
    ground_truth_path='/kaggle/working/calibration_data',
    colmap_output_path='/kaggle/working/colmap_output'
)


# ============================================================================
# セル 5: 詳細な座標比較
# ============================================================================
"""
Ground TruthとCOLMAP出力の座標を詳しく比較
"""

import numpy as np
from pathlib import Path

# Ground Truth読み込み
gt_points = np.loadtxt('/kaggle/working/calibration_data/ground_truth_points3d.txt')

# COLMAP出力読み込み
colmap_points = []
with open('/kaggle/working/colmap_output/points3D.txt', 'r') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            colmap_points.append([x, y, z])
colmap_points = np.array(colmap_points)

# 視覚化
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 5))

# Ground Truth
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], c='blue', label='Ground Truth')
ax1.set_title('Ground Truth')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# COLMAP出力
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(colmap_points[:, 0], colmap_points[:, 1], colmap_points[:, 2], c='red', label='COLMAP')
ax2.set_title('COLMAP Output')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()

# 重ね合わせ
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
           c='blue', alpha=0.6, label='Ground Truth')
ax3.scatter(colmap_points[:, 0], colmap_points[:, 1], colmap_points[:, 2], 
           c='red', alpha=0.6, label='COLMAP')
ax3.set_title('Overlay')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.legend()

plt.tight_layout()
plt.savefig('/kaggle/working/coordinate_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ 比較画像を保存: /kaggle/working/coordinate_comparison.png")


# ============================================================================
# セル 6: 実際のMASt3R出力との統合例
# ============================================================================
"""
実際のMASt3Rパイプラインにキャリブレーションシーンを組み込む例
"""

# 通常のMASt3R処理の代わりにキャリブレーションシーンを使用
USE_CALIBRATION = True  # Trueでキャリブレーションモード

if USE_CALIBRATION:
    print("【キャリブレーションモード】")
    # キャリブレーションシーンを使用
    scene, images = create_calibration_scene(
        num_views=4,
        num_points=100,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
else:
    print("【通常モード】")
    # 実際のMASt3R処理
    # scene, images = run_mast3r_pairs(model, image_paths, pairs, device)
    pass

# この後は通常通りCOLMAP変換を実行
# mast3r_to_colmap(scene, images, output_path)

# キャリブレーションモードの場合のみ検証
if USE_CALIBRATION:
    # COLMAP変換実行後
    # verify_transformation('/kaggle/working/calibration_data', output_path)
    pass


# ============================================================================
# セル 7: トラブルシューティング
# ============================================================================
"""
問題が発生した場合のデバッグ情報出力
"""

def debug_scene_info(scene):
    """シーンオブジェクトの詳細情報を出力"""
    print("=== Scene Debug Info ===")
    print(f"Device: {scene.device}")
    print(f"Views: {scene.num_views}")
    print(f"Points: {scene.num_points}")
    
    print("\nGround Truth Points (first 3):")
    if hasattr(scene, 'ground_truth_pts3d_world'):
        print(scene.ground_truth_pts3d_world[:3])
    
    print("\nCamera Poses (first camera):")
    if hasattr(scene, 'ground_truth_poses'):
        print(scene.ground_truth_poses[0])
    
    print("\nPer-view 3D points (first view, first 3 points):")
    pts = scene.get_pts3d(0)
    if torch.is_tensor(pts):
        print(pts[:3].cpu().numpy())
    else:
        print(pts[:3])
    
    print("\nFocals:")
    print(scene.get_focals())
    
    print("\nPrincipal Points:")
    print(scene.get_principal_points())
    
    print("\nImage Poses (first camera):")
    print(scene.get_im_poses()[0])

# 使用例
# scene, images = create_calibration_scene()
# debug_scene_info(scene)


# ============================================================================
# セル 8: カスタム設定での生成
# ============================================================================
"""
より多くの点や異なる配置でテスト
"""

# 高密度テスト
scene_dense, images_dense = create_calibration_scene(
    num_views=8,      # カメラ数を増やす
    num_points=500,   # 点数を増やす
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_dir='/kaggle/working/calibration_dense'
)

print("高密度キャリブレーションシーン生成完了")
print(f"  ビュー数: {scene_dense.num_views}")
print(f"  点数: {scene_dense.num_points}")


# ============================================================================
# まとめ
# ============================================================================
"""
基本的な使用フロー:

1. coordinate_verification.py を /kaggle/working/ に配置
2. インポート: from coordinate_verification import *
3. シーン生成: scene, images = create_calibration_scene()
4. COLMAP変換: your_function(scene, images, output_path)
5. 検証: verify_transformation(gt_path, colmap_path)

これにより、COLMAP変換過程で座標が変化しているかを確認できます。
"""
