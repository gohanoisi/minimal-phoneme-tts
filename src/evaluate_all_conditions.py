#!/usr/bin/env python3
"""
全条件の評価スクリプト
4条件すべてを評価し、結果をCSVとJSONで保存する
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from evaluate import evaluate_condition


# 評価対象の条件
CONDITIONS = [
    "train_80sent",
    "train_4sent_37phonemes",
    "train_4sent_random",
    "train_10sent_top",
]


def evaluate_all(
    project_root: Path,
    synth_base_dir: Path,
    reference_audio_dir: Path,
    results_dir: Path,
    fs: int = 24000,
) -> Dict[str, Dict]:
    """
    全条件を評価する
    
    Args:
        project_root: プロジェクトルートディレクトリ
        synth_base_dir: 合成音声のベースディレクトリ（条件名のサブディレクトリを含む）
        reference_audio_dir: 参照音声ディレクトリ
        results_dir: 結果保存ディレクトリ
        fs: サンプリングレート
        
    Returns:
        全条件の評価結果の辞書
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for condition in CONDITIONS:
        print("=" * 60)
        print(f"Evaluating condition: {condition}")
        print("=" * 60)
        
        # 合成音声ディレクトリ
        synth_audio_dir = synth_base_dir / condition
        
        if not synth_audio_dir.exists():
            logging.warning(f"Synthetic audio directory not found: {synth_audio_dir}")
            continue
        
        # 出力JSONファイル
        output_json = results_dir / f"evaluation_{condition}.json"
        
        try:
            # 評価実行
            condition_results = evaluate_condition(
                synth_audio_dir=synth_audio_dir,
                reference_audio_dir=reference_audio_dir,
                output_json=output_json,
                fs=fs,
            )
            
            all_results[condition] = condition_results
            
            print(f"MCD: {condition_results['mcd']:.4f} dB")
            print(f"log-F0 RMSE: {condition_results['log_f0_rmse']:.4f}")
            print(f"Number of files: {condition_results['num_files']}")
            print()
            
        except Exception as e:
            logging.error(f"Failed to evaluate {condition}: {e}")
            continue
    
    # 全結果をJSONで保存
    summary_json = results_dir / "evaluation_results.json"
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_json}")
    
    # CSV形式で保存
    summary_data = []
    for condition, results in all_results.items():
        summary_data.append({
            "condition": condition,
            "mcd": results["mcd"],
            "log_f0_rmse": results["log_f0_rmse"],
            "num_files": results["num_files"],
        })
    
    df = pd.DataFrame(summary_data)
    summary_csv = results_dir / "evaluation_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")
    
    # 結果表示
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    
    return all_results


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="Evaluate all conditions")
    parser.add_argument(
        "--synth_base_dir",
        type=str,
        default="outputs/audio/exp_text",
        help="Base directory containing synthetic audio files (default: outputs/audio/exp_text)"
    )
    parser.add_argument(
        "--reference_audio_dir",
        type=str,
        default="data/jvs002/parallel100/wav24kHz16bit",
        help="Directory containing reference audio files (default: data/jvs002/parallel100/wav24kHz16bit)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results (default: results)"
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=24000,
        help="Sampling rate (default: 24000)"
    )
    
    args = parser.parse_args()
    
    # プロジェクトルート
    project_root = Path(__file__).parent.parent
    
    # パス解決
    synth_base_dir = project_root / args.synth_base_dir
    reference_audio_dir = project_root / args.reference_audio_dir
    results_dir = project_root / args.results_dir
    
    # 評価実行
    evaluate_all(
        project_root=project_root,
        synth_base_dir=synth_base_dir,
        reference_audio_dir=reference_audio_dir,
        results_dir=results_dir,
        fs=args.fs,
    )


if __name__ == "__main__":
    main()
