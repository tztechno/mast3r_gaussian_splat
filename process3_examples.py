#!/usr/bin/env python3
"""
Process 3 使用例スクリプト
MASt3RからCOLMAP形式への変換の実践例
"""

import os
import sys
from pathlib import Path

# process3をインポート
from process3 import MASt3RToCOLMAPConverter, convert_mast3r_to_colmap


# ============================================================================
# 例1: 最もシンプルな使い方
# ============================================================================

def example1_simple_conversion(scene, output_dir='output/example1'):
    """
    最もシンプルな変換例
    """
    print("=" * 70)
    print("例1: シンプルな変換")
    print("=" * 70)
    
    # これだけ！
    output_path = convert_mast3r_to_colmap(
        scene=scene,
        output_dir=output_dir,
        verbose=True
    )
    
    print(f"\n✅ 完了: {output_path}")
    return output_path


# ============================================================================
# 例2: パラメータを調整した変換
# ============================================================================

def example2_custom_parameters(scene, output_dir='output/example2'):
    """
    パラメータをカスタマイズした変換例
    """
    print("=" * 70)
    print("例2: カスタムパラメータ")
    print("=" * 70)
    
    output_path = convert_mast3r_to_colmap(
        scene=scene,
        output_dir=output_dir,
        min_conf_thr=1.5,      # より多くの点を含める
        clean_depth=True,       # 深度マップをクリーニング
        mask_images=True,       # マスク画像を保存
        verbose=True
    )
    
    print(f"\n✅ 完了: {output_path}")
    return output_path


# ============================================================================
# 例3: 統計情報付きの変換
# ============================================================================

def example3_with_statistics(scene, output_dir='output/example3'):
    """
    詳細な統計情報を取得する例
    """
    print("=" * 70)
    print("例3: 統計情報付き変換")
    print("=" * 70)
    
    # コンバータを初期化
    converter = MASt3RToCOLMAPConverter()
    
    # 統計情報付きで変換
    stats = converter.convert_with_summary(
        scene=scene,
        output_dir=output_dir,
        min_conf_thr=2.0,
        clean_depth=False,
        verbose=True
    )
    
    # 統計情報を表示
    print("\n" + "=" * 70)
    print("📊 変換統計:")
    print("-" * 70)
    print(f"画像数:                {stats['num_images']}")
    print(f"画像サイズ:            {stats['image_shape']}")
    print(f"総3D点数:              {stats['total_points']:,}")
    print(f"画像あたりの平均点数:  {stats['avg_points_per_image']:.1f}")
    print(f"信頼度閾値:            {stats['min_conf_threshold']}")
    print(f"深度クリーニング:      {stats['clean_depth']}")
    print(f"保存先:                {stats['save_path']}")
    print("=" * 70)
    
    return stats


# ============================================================================
# 例4: 異なる信頼度閾値で複数回変換
# ============================================================================

def example4_multiple_thresholds(scene, base_output_dir='output/example4'):
    """
    異なる信頼度閾値で複数回変換して比較する例
    """
    print("=" * 70)
    print("例4: 複数の信頼度閾値で変換")
    print("=" * 70)
    
    converter = MASt3RToCOLMAPConverter()
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    results = []
    
    print("\n信頼度閾値を変えて変換中...")
    print("-" * 70)
    
    for threshold in thresholds:
        output_dir = os.path.join(base_output_dir, f'threshold_{threshold}')
        
        stats = converter.convert_with_summary(
            scene=scene,
            output_dir=output_dir,
            min_conf_thr=threshold,
            verbose=False
        )
        
        results.append(stats)
        print(f"閾値 {threshold:3.1f}: {stats['total_points']:8,} 点")
    
    # 最適な閾値を推奨
    print("\n" + "=" * 70)
    print("📊 閾値比較結果:")
    print("-" * 70)
    print(f"{'閾値':<8} {'総点数':<12} {'平均点数/画像':<15}")
    print("-" * 70)
    
    for stats in results:
        thr = stats['min_conf_threshold']
        total = stats['total_points']
        avg = stats['avg_points_per_image']
        print(f"{thr:<8.1f} {total:<12,} {avg:<15.1f}")
    
    print("-" * 70)
    print("\n💡 推奨:")
    print("  - 3DGS用: 閾値 1.5〜2.0 (バランス型)")
    print("  - 高品質用: 閾値 2.5〜3.0 (厳格)")
    print("  - 大量点群用: 閾値 1.0〜1.5 (寛容)")
    print("=" * 70)
    
    return results


# ============================================================================
# 例5: 完全なパイプライン（MASt3R → COLMAP → 3DGS）
# ============================================================================

def example5_full_pipeline(image_files, output_base='output/example5'):
    """
    MASt3Rから3DGSまでの完全なパイプライン例
    （疑似コード）
    """
    print("=" * 70)
    print("例5: 完全なパイプライン")
    print("=" * 70)
    
    print("\n📝 パイプラインの流れ:")
    print("-" * 70)
    print("1. 画像読み込み")
    print("2. MASt3Rで3D復元")
    print("3. COLMAP形式に変換（Process 3）")
    print("4. 3D Gaussian Splattingで学習")
    print("-" * 70)
    
    # 疑似コード（実際のコードは環境に応じて調整）
    print("\n【擬似コード】")
    print("""
    # Step 1: 画像読み込み
    from dust3r.utils.image import load_images
    imgs = load_images(image_files, size=512)
    
    # Step 2: MASt3Rで3D復元
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from dust3r.image_pairs import make_pairs
    
    pairs = make_pairs(imgs, scene_graph='complete')
    scene = sparse_global_alignment(
        filelist=image_files,
        pairs=pairs,
        cache_dir='cache',
        model=model,
        device='cuda',
        lr1=0.07,
        niter1=600,
        lr2=0.014,
        niter2=300,
        shared_intrinsics=False
    )
    
    # Step 3: COLMAP形式に変換
    from process3 import convert_mast3r_to_colmap
    
    colmap_dir = convert_mast3r_to_colmap(
        scene=scene,
        output_dir='output/colmap_data',
        min_conf_thr=2.0,
        clean_depth=False,
        verbose=True
    )
    
    # Step 4: 3D Gaussian Splattingで学習
    # （gaussian-splattingのtrainスクリプトを使用）
    import subprocess
    
    subprocess.run([
        'python', 'train.py',
        '--source_path', colmap_dir,
        '--model_path', 'output/gs_model',
        '--iterations', '30000',
        '--position_lr_init', '0.00032',
        '--feature_lr', '0.0025',
    ])
    
    print("✅ パイプライン完了！")
    """)


# ============================================================================
# 例6: バッチ処理
# ============================================================================

def example6_batch_processing(scenes_dict, output_base='output/example6'):
    """
    複数のシーンを一括処理する例
    
    Args:
        scenes_dict: {'scene_name': scene_object} の辞書
    """
    print("=" * 70)
    print("例6: バッチ処理")
    print("=" * 70)
    
    converter = MASt3RToCOLMAPConverter()
    results = {}
    
    print(f"\n{len(scenes_dict)}個のシーンを処理中...")
    print("-" * 70)
    
    for scene_name, scene in scenes_dict.items():
        output_dir = os.path.join(output_base, scene_name)
        
        try:
            stats = converter.convert_with_summary(
                scene=scene,
                output_dir=output_dir,
                min_conf_thr=2.0,
                verbose=False
            )
            results[scene_name] = stats
            print(f"✅ {scene_name:<20} → {stats['total_points']:8,} 点")
            
        except Exception as e:
            print(f"❌ {scene_name:<20} → エラー: {e}")
            results[scene_name] = None
    
    # サマリー
    print("\n" + "=" * 70)
    print("📊 バッチ処理サマリー:")
    print("-" * 70)
    
    successful = sum(1 for r in results.values() if r is not None)
    total_points = sum(r['total_points'] for r in results.values() if r is not None)
    
    print(f"成功: {successful}/{len(scenes_dict)}")
    print(f"総3D点数: {total_points:,}")
    print("=" * 70)
    
    return results


# ============================================================================
# 例7: エラーハンドリング
# ============================================================================

def example7_error_handling(scene, output_dir='output/example7'):
    """
    エラーハンドリングの例
    """
    print("=" * 70)
    print("例7: エラーハンドリング")
    print("=" * 70)
    
    try:
        # COLMAP utilsのパスを明示的に指定
        converter = MASt3RToCOLMAPConverter(
            colmap_utils_path='path/to/colmap_dataset_utils.py'
        )
        
        stats = converter.convert_with_summary(
            scene=scene,
            output_dir=output_dir,
            min_conf_thr=2.0,
            verbose=True
        )
        
        print(f"\n✅ 変換成功: {stats['total_points']:,} 点")
        return stats
        
    except ImportError as e:
        print(f"\n❌ インポートエラー: {e}")
        print("\n解決策:")
        print("  1. colmap_dataset_utils.pyのパスを確認")
        print("  2. sys.pathに追加: sys.path.append('path/to/utils')")
        print("  3. 明示的にパスを指定:")
        print("     converter = MASt3RToCOLMAPConverter(")
        print("         colmap_utils_path='correct/path/to/colmap_dataset_utils.py'")
        print("     )")
        return None
        
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        print(f"エラータイプ: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# メイン実行部分
# ============================================================================

def main():
    """
    使用例のデモンストレーション
    """
    print("\n" + "=" * 70)
    print("Process 3: 使用例デモンストレーション")
    print("=" * 70)
    
    print("\n⚠️  注意: これらはデモ用の疑似コードです")
    print("実際に実行するには、MASt3Rのsceneオブジェクトが必要です")
    print()
    
    # 例を選択して実行
    examples = {
        '1': ('シンプルな変換', example1_simple_conversion),
        '2': ('カスタムパラメータ', example2_custom_parameters),
        '3': ('統計情報付き', example3_with_statistics),
        '4': ('複数の閾値で比較', example4_multiple_thresholds),
        '5': ('完全なパイプライン', example5_full_pipeline),
        '6': ('バッチ処理', example6_batch_processing),
        '7': ('エラーハンドリング', example7_error_handling),
    }
    
    print("実行可能な例:")
    print("-" * 70)
    for key, (desc, _) in examples.items():
        print(f"  {key}. {desc}")
    print("-" * 70)
    
    # すべての例の説明を表示
    for key, (desc, func) in examples.items():
        print(f"\n{'='*70}")
        print(f"例{key}: {desc}")
        print('='*70)
        
        # docstringを表示
        if func.__doc__:
            print(func.__doc__)
        
        print("\n使用方法:")
        print(f"  from examples import {func.__name__}")
        print(f"  {func.__name__}(scene)")
        print()


if __name__ == '__main__':
    main()
