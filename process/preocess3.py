#!/usr/bin/env python3
"""
Process 3: Dense Depth Map + Normals方式
MASt3RのシーンオブジェクトをCOLMAP形式に変換

この方式の特徴：
- Dense depth mapから密な点群を生成
- 信頼度ベースのフィルタリング
- 法線情報を含む点群の保存
- 3D Gaussian Splattingに最適化された形式

使用例：
    from process3 import MASt3RToCOLMAPConverter
    
    # MASt3Rで復元
    scene = sparse_global_alignment(...)
    
    # COLMAP形式に変換
    converter = MASt3RToCOLMAPConverter(
        colmap_utils_path='path/to/colmap_dataset_utils.py'
    )
    converter.convert(
        scene=scene,
        output_dir='output/colmap_data',
        min_conf_thr=2.0,
        clean_depth=False,
        mask_images=True
    )
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import importlib.util


class MASt3RToCOLMAPConverter:
    """
    MASt3RのシーンオブジェクトをCOLMAP形式に変換するクラス
    
    Process 3方式：Dense depth mapから密な点群を生成し、
    法線情報と共にCOLMAP形式で保存する。
    """
    
    def __init__(self, colmap_utils_path: Optional[str] = None):
        """
        Args:
            colmap_utils_path: colmap_dataset_utils.pyへのパス
                               Noneの場合は自動検出を試みる
        """
        self.colmap_utils_path = colmap_utils_path
        self.colmap_utils = None
        
    def _import_colmap_utils(self):
        """colmap_dataset_utilsモジュールをインポート"""
        if self.colmap_utils is not None:
            return
            
        # パスが指定されている場合
        if self.colmap_utils_path:
            utils_dir = os.path.dirname(self.colmap_utils_path)
            if utils_dir not in sys.path:
                sys.path.insert(0, utils_dir)
            
            spec = importlib.util.spec_from_file_location(
                "colmap_dataset_utils", 
                self.colmap_utils_path
            )
            self.colmap_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.colmap_utils)
        else:
            # 自動検出を試みる
            try:
                import colmap_dataset_utils
                self.colmap_utils = colmap_dataset_utils
            except ImportError:
                # 一般的な場所を探索
                search_paths = [
                    '../wild-gaussian-splatting/src',
                    './wild-gaussian-splatting/src',
                    '../src',
                    './src',
                ]
                
                for search_path in search_paths:
                    utils_file = os.path.join(search_path, 'colmap_dataset_utils.py')
                    if os.path.exists(utils_file):
                        if search_path not in sys.path:
                            sys.path.insert(0, search_path)
                        import colmap_dataset_utils
                        self.colmap_utils = colmap_dataset_utils
                        print(f"Found colmap_dataset_utils at: {utils_file}")
                        break
                
                if self.colmap_utils is None:
                    raise ImportError(
                        "colmap_dataset_utils not found. "
                        "Please specify colmap_utils_path explicitly."
                    )
    
    def _extract_scene_data(
        self, 
        scene, 
        min_conf_thr: float = 2.0,
        clean_depth: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        MASt3Rのシーンオブジェクトから必要なデータを抽出
        
        Args:
            scene: MASt3Rのシーンオブジェクト
            min_conf_thr: 信頼度の最小閾値
            clean_depth: 深度マップのクリーニングを行うか
            
        Returns:
            cam2world: カメラ→ワールド変換行列 (N, 4, 4)
            world2cam: ワールド→カメラ変換行列 (N, 4, 4)
            principal_points: 主点 (N, 2)
            focals: 焦点距離 (N, 1) または (N, 2)
            imgs: 画像データ (N, H, W, 3)
            pts3d: 各画像のdense depth map (N, H, W, 3)
            masks: 信頼度マスク (N, H, W)
        """
        # カメラパラメータを取得
        cam2world = scene.get_im_poses().detach().cpu().numpy()
        principal_points = scene.get_principal_points().detach().cpu().numpy()
        focals = scene.get_focals().detach().cpu().numpy()[..., None]  # (N, 1) or (N, 2)
        
        # 画像データを取得
        imgs = np.array(scene.imgs)
        
        # Dense depth mapと信頼度を取得
        pts3d, _, confs = scene.get_dense_pts3d(clean_depth=clean_depth)
        
        # 画像形状に整形
        pts3d = [pt.detach().cpu().numpy().reshape(imgs[0].shape) for pt in pts3d]
        
        # 信頼度マスクを生成
        confs_np = [c.detach().cpu().numpy() if hasattr(c, 'detach') else c for c in confs]
        masks = [c > min_conf_thr for c in confs_np]
        
        # world2camを計算（逆行列）
        world2cam = self.colmap_utils.inv(cam2world)
        
        return cam2world, world2cam, principal_points, focals, imgs, pts3d, masks
    
    def convert(
        self,
        scene,
        output_dir: str,
        min_conf_thr: float = 2.0,
        clean_depth: bool = False,
        mask_images: bool = True,
        verbose: bool = True
    ) -> str:
        """
        MASt3RのシーンをCOLMAP形式に変換
        
        Args:
            scene: MASt3Rのシーンオブジェクト
            output_dir: 出力ディレクトリ
            min_conf_thr: 信頼度の最小閾値（デフォルト: 2.0）
            clean_depth: 深度マップのクリーニングを行うか
            mask_images: マスク画像を保存するか
            verbose: 進行状況を表示するか
            
        Returns:
            保存先のディレクトリパス
        """
        # colmap_utilsをインポート
        self._import_colmap_utils()
        
        if verbose:
            print(f"Converting MASt3R scene to COLMAP format...")
            print(f"Output directory: {output_dir}")
            print(f"Min confidence threshold: {min_conf_thr}")
            print(f"Clean depth: {clean_depth}")
        
        # シーンデータを抽出
        cam2world, world2cam, principal_points, focals, imgs, pts3d, masks = \
            self._extract_scene_data(scene, min_conf_thr, clean_depth)
        
        if verbose:
            print(f"Extracted data:")
            print(f"  - Number of images: {len(imgs)}")
            print(f"  - Image shape: {imgs[0].shape}")
            print(f"  - Number of 3D points per image: ~{np.sum(masks[0])}")
        
        # COLMAPディレクトリ構造を初期化
        save_path, images_path, masks_path, sparse_path = \
            self.colmap_utils.init_filestructure(output_dir)
        
        if verbose:
            print(f"\nSaving COLMAP data:")
            print(f"  - Images: {images_path}")
            print(f"  - Masks: {masks_path}")
            print(f"  - Sparse: {sparse_path}")
        
        # 画像とマスクを保存
        self.colmap_utils.save_images_masks(
            imgs, masks, images_path, masks_path, mask_images
        )
        
        # カメラパラメータを保存
        self.colmap_utils.save_cameras(
            focals, principal_points, sparse_path, imgs_shape=imgs.shape
        )
        
        # カメラポーズを保存
        self.colmap_utils.save_imagestxt(world2cam, sparse_path)
        
        # 法線付き点群を保存
        self.colmap_utils.save_pointcloud_with_normals(
            imgs, pts3d, masks, sparse_path
        )
        
        if verbose:
            print(f"\n✅ Successfully converted to COLMAP format!")
            print(f"Output directory: {save_path}")
        
        return save_path
    
    def convert_with_summary(
        self,
        scene,
        output_dir: str,
        **kwargs
    ) -> dict:
        """
        変換を実行し、詳細な統計情報を返す
        
        Returns:
            統計情報を含む辞書
        """
        # データ抽出
        min_conf_thr = kwargs.get('min_conf_thr', 2.0)
        clean_depth = kwargs.get('clean_depth', False)
        
        cam2world, world2cam, principal_points, focals, imgs, pts3d, masks = \
            self._extract_scene_data(scene, min_conf_thr, clean_depth)
        
        # 統計情報を計算
        total_points = sum(np.sum(mask) for mask in masks)
        avg_points_per_image = total_points / len(masks)
        
        stats = {
            'num_images': len(imgs),
            'image_shape': imgs[0].shape,
            'total_points': int(total_points),
            'avg_points_per_image': float(avg_points_per_image),
            'min_conf_threshold': min_conf_thr,
            'clean_depth': clean_depth,
            'output_dir': output_dir,
        }
        
        # 変換を実行
        save_path = self.convert(scene, output_dir, **kwargs)
        stats['save_path'] = save_path
        
        return stats


def convert_mast3r_to_colmap(
    scene,
    output_dir: str,
    colmap_utils_path: Optional[str] = None,
    min_conf_thr: float = 2.0,
    clean_depth: bool = False,
    mask_images: bool = True,
    verbose: bool = True
) -> str:
    """
    便利関数：MASt3RシーンをCOLMAP形式に変換
    
    Args:
        scene: MASt3Rのシーンオブジェクト
        output_dir: 出力ディレクトリ
        colmap_utils_path: colmap_dataset_utils.pyへのパス（Noneで自動検出）
        min_conf_thr: 信頼度の最小閾値
        clean_depth: 深度マップのクリーニング
        mask_images: マスク画像の保存
        verbose: 進行状況の表示
        
    Returns:
        保存先のディレクトリパス
    """
    converter = MASt3RToCOLMAPConverter(colmap_utils_path)
    return converter.convert(
        scene, output_dir, min_conf_thr, clean_depth, mask_images, verbose
    )


# ============================================================================
# 使用例
# ============================================================================

if __name__ == "__main__":
    """
    使用例のデモンストレーション
    """
    print("=" * 70)
    print("Process 3: MASt3R to COLMAP Converter")
    print("Dense Depth Map + Normals方式")
    print("=" * 70)
    print()
    
    print("【使用例1】基本的な使い方")
    print("-" * 70)
    print("""
    from process3 import convert_mast3r_to_colmap
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    
    # MASt3Rで3D復元
    scene = sparse_global_alignment(
        filelist=image_files,
        pairs=pairs,
        cache_dir=cache_dir,
        model=model,
        device=device,
        ...
    )
    
    # COLMAP形式に変換
    output_path = convert_mast3r_to_colmap(
        scene=scene,
        output_dir='output/colmap_data',
        min_conf_thr=2.0,      # 信頼度閾値
        clean_depth=False,      # 深度クリーニング
        mask_images=True,       # マスク画像を保存
        verbose=True
    )
    
    print(f"Saved to: {output_path}")
    """)
    
    print("\n【使用例2】クラスを使った詳細な制御")
    print("-" * 70)
    print("""
    from process3 import MASt3RToCOLMAPConverter
    
    # コンバータを初期化
    converter = MASt3RToCOLMAPConverter(
        colmap_utils_path='path/to/colmap_dataset_utils.py'
    )
    
    # 統計情報付きで変換
    stats = converter.convert_with_summary(
        scene=scene,
        output_dir='output/colmap_data',
        min_conf_thr=1.5,
        clean_depth=True
    )
    
    print(f"Number of images: {stats['num_images']}")
    print(f"Total 3D points: {stats['total_points']}")
    print(f"Average points per image: {stats['avg_points_per_image']:.1f}")
    """)
    
    print("\n【出力ディレクトリ構造】")
    print("-" * 70)
    print("""
    output/colmap_data/
    ├── images/              # 入力画像
    │   ├── 0000.jpg
    │   ├── 0001.jpg
    │   └── ...
    ├── masks/               # 信頼度マスク（オプション）
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    └── sparse/0/            # COLMAP形式のデータ
        ├── cameras.bin      # カメラ内部パラメータ
        ├── images.bin       # カメラポーズ
        └── points3D.ply     # 法線付き密な点群
    """)
    
    print("\n【Process 3の特徴】")
    print("-" * 70)
    print("""
    ✅ Dense depth mapから密な点群を生成
    ✅ 信頼度ベースのフィルタリング
    ✅ 法線情報を含む点群の保存
    ✅ 3D Gaussian Splattingに最適化
    ✅ Traditional方式より遥かに多くの点を生成
    
    💡 3DGSの初期化に最適：
       - 密な幾何情報で高品質な初期化
       - 法線情報でより正確なサーフェス推定
       - マスクで低品質領域を除外
    """)
    
    print("\n【他の方式との比較】")
    print("-" * 70)
    print("""
    方式           | 点群の種類 | データ量 | 2D-3D対応 | 用途
    ---------------|-----------|---------|-----------|------------------
    Traditional    | 疎（SfM） | 少ない  | あり      | 従来のCOLMAP
    Process 2      | 中間      | 中程度  | 部分的    | 実験的
    Process 3 ★    | 密（DM）  | 多い    | なし      | 3DGS初期化に最適
    
    DM = Depth Map
    """)
    
    print("\n" + "=" * 70)
    print("詳細なドキュメントはコード内のdocstringを参照してください")
    print("=" * 70)
