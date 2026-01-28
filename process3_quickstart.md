# Process 3 クイックスタートガイド

**5分でMASt3RからCOLMAP形式への変換を始める**

## 🚀 30秒でスタート

```python
from process3 import convert_mast3r_to_colmap

# MASt3Rで復元したsceneオブジェクトを変換
output_path = convert_mast3r_to_colmap(
    scene=scene,
    output_dir='output/colmap_data'
)
```

**これだけ！** 🎉

---

## 📋 前提条件

### 必要なもの

1. **MASt3Rのsceneオブジェクト**
   ```python
   from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
   scene = sparse_global_alignment(...)
   ```

2. **colmap_dataset_utils.py** 
   - Wild Gaussian Splattingの`src/`ディレクトリから取得
   - または`process3.py`と同じディレクトリに配置

---

## 🔧 セットアップ

### ステップ1: ファイル配置

```
your_project/
├── process3.py                  # ← ダウンロード
├── colmap_dataset_utils.py     # ← 必要
└── your_script.py               # ← あなたのコード
```

### ステップ2: インポート

```python
from process3 import convert_mast3r_to_colmap
```

---

## 💻 基本的な使い方

### パターン1: 最小限のコード

```python
# シーンを変換
output_path = convert_mast3r_to_colmap(scene, 'output/colmap')
```

### パターン2: パラメータ調整

```python
output_path = convert_mast3r_to_colmap(
    scene=scene,
    output_dir='output/colmap',
    min_conf_thr=2.0,      # 信頼度閾値
    clean_depth=False,      # 深度クリーニング
    mask_images=True,       # マスク画像保存
    verbose=True            # 進行状況表示
)
```

### パターン3: 統計情報取得

```python
from process3 import MASt3RToCOLMAPConverter

converter = MASt3RToCOLMAPConverter()
stats = converter.convert_with_summary(scene, 'output/colmap')

print(f"総3D点数: {stats['total_points']:,}")
```

---

## 📂 出力を確認

変換後のディレクトリ構造：

```
output/colmap/
├── images/          # 入力画像
├── masks/           # 信頼度マスク
└── sparse/0/        # COLMAP形式データ
    ├── cameras.bin
    ├── images.bin
    └── points3D.ply
```

---

## 🎯 次のステップ

### 3D Gaussian Splattingで使う

```python
# 1. COLMAP形式に変換
colmap_dir = convert_mast3r_to_colmap(scene, 'data/colmap')

# 2. 3DGSで学習（gaussian-splattingを使用）
import subprocess
subprocess.run([
    'python', 'train.py',
    '--source_path', colmap_dir,
    '--iterations', '30000'
])
```

---

## ⚙️ よく使うパラメータ

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `min_conf_thr` | **2.0** | デフォルト・バランス型 |
| | 1.5 | より多くの点を含める |
| | 2.5-3.0 | 高品質・厳格 |

---

## 🐛 トラブルシューティング

### エラー: `ImportError: colmap_dataset_utils not found`

**解決策1**: パスを指定
```python
converter = MASt3RToCOLMAPConverter(
    colmap_utils_path='path/to/colmap_dataset_utils.py'
)
```

**解決策2**: sys.pathに追加
```python
import sys
sys.path.append('path/to/utils/directory')
```

### 点群が少ない

**解決策**: 閾値を下げる
```python
convert_mast3r_to_colmap(scene, output_dir, min_conf_thr=1.5)
```

---

## 📚 より詳しく学ぶ

- **詳細ドキュメント**: `README_process3.md`
- **使用例**: `examples_process3.py`
- **コード内ドキュメント**: `process3.py`の docstring

---

## ✅ チェックリスト

変換を実行する前に：

- [ ] MASt3Rのsceneオブジェクトを用意
- [ ] `process3.py`をダウンロード
- [ ] `colmap_dataset_utils.py`を配置
- [ ] 出力ディレクトリを決定

---

## 🎓 完全な例

```python
#!/usr/bin/env python3
"""MASt3RからCOLMAPへの完全な変換例"""

import torch
from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from process3 import convert_mast3r_to_colmap

# 1. モデル読み込み
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AsymmetricMASt3R.from_pretrained(
    "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
).to(device)

# 2. 画像読み込み
image_files = ['img1.jpg', 'img2.jpg', 'img3.jpg']
imgs = load_images(image_files, size=512)

# 3. ペア作成
pairs = make_pairs(imgs, scene_graph='complete')

# 4. MASt3Rで3D復元
scene = sparse_global_alignment(
    filelist=image_files,
    pairs=pairs,
    cache_dir='cache',
    model=model,
    device=device,
    lr1=0.07,
    niter1=600,
    lr2=0.014,
    niter2=300
)

# 5. COLMAP形式に変換
output_path = convert_mast3r_to_colmap(
    scene=scene,
    output_dir='output/colmap_data',
    min_conf_thr=2.0,
    verbose=True
)

print(f"✅ 完了！ 出力: {output_path}")
```

---

**これで準備完了です！ 🚀**

質問があれば `README_process3.md` を参照してください。
