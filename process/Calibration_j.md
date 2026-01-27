# MASt3R to COLMAP 座標変換検証ツール

## 概要

MASt3Rの出力をCOLMAP形式に変換する際に、3D座標が変化するかどうかを検証するためのツールです。

既知の3D座標を持つ合成シーンデータを生成し、変換パイプラインを通した後の座標と比較することで、変換の正確性を確認できます。

## ファイル構成

```
calibration_data_generator.py  # キャリブレーションデータ生成器
test_calibration_pipeline.py   # テストパイプライン実行
calibration_usage_example.py   # 使用例
CALIBRATION_README.md          # このファイル
```

## 主な機能

### 1. CalibrationDataGenerator

合成シーンデータを生成します:
- **既知の3D座標**: 立方体形状の点群(デフォルト100点)
- **カメラポーズ**: 円形配置された複数視点(デフォルト4視点)
- **Ground Truth保存**: 検証用の基準データを保存

```python
from calibration_data_generator import CalibrationDataGenerator

generator = CalibrationDataGenerator(num_views=4, num_points=100)
scene = generator.create_mock_scene(device='cpu')
generator.save_ground_truth(scene, './calibration_data')
```

### 2. test_calibration_pipeline

完全なテストパイプラインを実行:

```python
from test_calibration_pipeline import test_calibration_pipeline
from your_module import mast3r_to_colmap

# COLMAP変換関数をテスト
test_calibration_pipeline(mast3r_to_colmap, output_dir='./test_results')
```

### 3. verify_transformation

Ground TruthとCOLMAP出力を比較:

```python
from calibration_data_generator import verify_transformation

verify_transformation(
    ground_truth_path='./calibration_data',
    colmap_output_path='./colmap_output'
)
```

## 使用方法

### 方法A: 自動テスト(推奨)

```python
# 1. インポート
from test_calibration_pipeline import test_calibration_pipeline
from your_module import mast3r_to_colmap  # 既存の変換関数

# 2. テスト実行
test_calibration_pipeline(
    colmap_converter_func=mast3r_to_colmap,
    output_dir='./calibration_test'
)

# 3. 結果確認
# - 座標統計の比較
# - Procrustes alignmentによる類似度
# - 座標比較ファイル生成
```

### 方法B: 手動テスト

```python
# 1. キャリブレーションデータ生成
from calibration_data_generator import CalibrationDataGenerator

generator = CalibrationDataGenerator(num_views=6, num_points=120)
scene = generator.create_mock_scene(device='cpu')
generator.save_ground_truth(scene, './ground_truth')

# 2. 既存のパイプラインで処理
# (sceneオブジェクトを通常通り使用)
mast3r_to_colmap(scene, images, './colmap_output')

# 3. 検証
from calibration_data_generator import verify_transformation
verify_transformation('./ground_truth', './colmap_output')
```

### 方法C: 既存の出力を検証

```python
from calibration_data_generator import verify_transformation

# 既に生成済みのCOLMAP出力を検証
verify_transformation(
    ground_truth_path='./calibration_data',
    colmap_output_path='./existing_colmap_output'
)
```

## 生成されるファイル

### Ground Truthディレクトリ
```
calibration_data/
├── ground_truth_points3d.txt       # 3D座標(X Y Z)
├── ground_truth_poses.json         # カメラポーズ(4x4行列)
└── ground_truth_intrinsics.json    # カメラ内部パラメータ
```

### 検証結果
```
test_results/
├── ground_truth/                   # Ground Truthデータ
├── colmap_output/                  # COLMAP変換後データ
│   ├── cameras.txt
│   ├── images.txt
│   ├── points3D.txt
│   └── coordinate_comparison.txt   # 比較結果
```

## 検証項目

1. **点数の一致**: Ground TruthとCOLMAP出力の点数比較
2. **座標統計**: 平均、標準偏差、最小・最大値の比較
3. **Procrustes alignment**: 剛体変換後の類似度(0に近いほど良い)
4. **座標フレームの変化**: スケール、回転、平行移動の有無

## 検証結果の解釈

### ✓ 座標が保持されている場合
```
Procrustes alignment disparity: 0.000123
✓ Coordinates match after alignment (rigid transformation)
```
- 剛体変換(回転・平行移動)のみ
- 座標の相対的な関係は保持

### ⚠ 小さな差がある場合
```
Procrustes alignment disparity: 0.045678
⚠ Coordinates approximately match (small differences)
```
- 数値誤差や最適化による微小な変化
- 通常は許容範囲内

### ❌ 大きな変化がある場合
```
Procrustes alignment disparity: 0.523456
❌ Significant coordinate differences detected
```
- スケール変化、非線形変換、データ損失など
- 変換プロセスの見直しが必要

## 実装例

### シンプルな統合例

```python
import torch
from calibration_data_generator import CalibrationDataGenerator
from test_calibration_pipeline import test_calibration_pipeline

# 既存のCOLMAP変換関数があると仮定
def mast3r_to_colmap(scene, images, output_path, masks=None):
    # 変換処理
    pass

# テスト実行
if __name__ == "__main__":
    test_calibration_pipeline(
        colmap_converter_func=mast3r_to_colmap,
        output_dir='./test_output'
    )
```

### Jupyter Notebookでの使用

```python
# セル1: インポート
from calibration_data_generator import CalibrationDataGenerator, verify_transformation
import numpy as np
import torch

# セル2: データ生成
generator = CalibrationDataGenerator(num_views=4, num_points=100)
scene = generator.create_mock_scene(device='cuda')
generator.save_ground_truth(scene, './calibration_data')

print("Ground Truth 3D points (first 5):")
print(scene.ground_truth_pts3d_world[:5])

# セル3: COLMAP変換実行
# (既存のコードをそのまま使用、sceneオブジェクトを渡す)
mast3r_to_colmap(scene, images, './colmap_output')

# セル4: 検証
verify_transformation('./calibration_data', './colmap_output')
```

## トラブルシューティング

### Q: "No points found in COLMAP output"
A: COLMAP変換が正しく実行されていません。points3D.txtの生成を確認してください。

### Q: 点数が一致しない
A: フィルタリングや閾値処理で点が除外されている可能性があります。変換ロジックを確認してください。

### Q: 座標が大きく変化している
A: 以下を確認:
- 座標系の定義(カメラ座標 vs ワールド座標)
- スケール正規化の有無
- 変換行列の適用順序

## カスタマイズ

### 点群の形状変更

```python
# 立方体以外の形状
class CustomGenerator(CalibrationDataGenerator):
    def generate_sphere_points(self, radius=1.0):
        # 球面上の点を生成
        phi = np.random.uniform(0, 2*np.pi, self.num_points)
        theta = np.random.uniform(0, np.pi, self.num_points)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.column_stack([x, y, z])
```

### カメラ配置変更

```python
# 螺旋配置
def generate_spiral_camera_poses(self, radius=5.0, height_range=2.0, turns=2):
    poses = []
    for i in range(self.num_views):
        t = i / (self.num_views - 1)
        angle = 2 * np.pi * turns * t
        height = height_range * (2*t - 1)
        # ... (カメラポーズ計算)
    return poses
```

## 技術詳細

### 座標系
- **ワールド座標**: 立方体の中心が原点
- **カメラ座標**: 各カメラから見た座標
- **変換**: 4x4同次変換行列

### 生成される点群
- 立方体の8頂点
- 12辺上の等間隔点
- 合計: 指定した点数(デフォルト100)

### カメラモデル
- ピンホールカメラモデル
- 焦点距離: 500px
- 主点: (256, 256)
- 画像サイズ: 512×512

## 参考資料

- MASt3R: https://github.com/naver/mast3r
- COLMAP format: https://colmap.github.io/format.html
- Procrustes analysis: scipy.spatial.procrustes

## ライセンス

このツールは検証目的で自由に使用・改変できます。
