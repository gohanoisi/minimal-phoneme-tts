#!/usr/bin/env python3
"""
結果可視化スクリプト
Phase 7: 評価結果の可視化（棒グラフ、散布図など）
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")


def load_evaluation_results(results_dir: Path) -> pd.DataFrame:
    """Load evaluation results"""
    summary_csv = results_dir / "evaluation_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Evaluation summary not found: {summary_csv}")
    
    df = pd.read_csv(summary_csv)
    return df


def load_corpus_selection(results_dir: Path) -> Dict:
    """Load corpus selection results"""
    corpus_json = results_dir / "corpus_selection.json"
    if not corpus_json.exists():
        raise FileNotFoundError(f"Corpus selection not found: {corpus_json}")
    
    with open(corpus_json, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    return corpus_data


def get_condition_labels() -> Dict[str, str]:
    """Get condition labels"""
    return {
        "train_80sent": "80 sentences",
        "train_4sent_37phonemes": "4 sentences (37 phonemes)",
        "train_4sent_random": "4 sentences (low coverage)",
        "train_10sent_top": "10 sentences (top)",
    }


def plot_mcd_comparison(df: pd.DataFrame, output_path: Path):
    """MCD比較棒グラフ"""
    condition_labels = get_condition_labels()
    
    # データ準備
    df_plot = df.copy()
    df_plot['condition_label'] = df_plot['condition'].map(condition_labels)
    
    # ソート（MCDの昇順）
    df_plot = df_plot.sort_values('mcd')
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df_plot['condition_label'], df_plot['mcd'], 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # 値のラベルを追加
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('MCD (dB)', fontsize=12)
    ax.set_title('MCD Comparison (All Conditions)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MCD comparison graph saved: {output_path}")


def plot_f0_comparison(df: pd.DataFrame, output_path: Path):
    """log-F0 RMSE比較棒グラフ"""
    condition_labels = get_condition_labels()
    
    # データ準備
    df_plot = df.copy()
    df_plot['condition_label'] = df_plot['condition'].map(condition_labels)
    
    # ソート（log-F0 RMSEの昇順）
    df_plot = df_plot.sort_values('log_f0_rmse')
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df_plot['condition_label'], df_plot['log_f0_rmse'],
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # 値のラベルを追加
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('log-F0 RMSE', fontsize=12)
    ax.set_title('log-F0 RMSE Comparison (All Conditions)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"log-F0 RMSE comparison graph saved: {output_path}")


def plot_coverage_vs_mcd(df: pd.DataFrame, corpus_data: Dict, output_path: Path):
    """音素カバレッジとMCDの散布図"""
    condition_labels = get_condition_labels()
    
    # 音素カバレッジを取得（38音素に対する割合）
    total_phonemes = 38
    coverage_map = {
        "train_80sent": corpus_data.get("E1_80sent", {}).get("coverage", {}).get("unique_phoneme_count", 38) / total_phonemes * 100,
        "train_4sent_37phonemes": corpus_data.get("E2_4sent_37phonemes", {}).get("coverage", {}).get("unique_phoneme_count", 38) / total_phonemes * 100,
        "train_4sent_random": corpus_data.get("E3_4sent_random", {}).get("coverage", {}).get("unique_phoneme_count", 25) / total_phonemes * 100,
        "train_10sent_top": corpus_data.get("E4_10sent_top", {}).get("coverage", {}).get("unique_phoneme_count", 38) / total_phonemes * 100,
    }
    
    # データ準備
    df_plot = df.copy()
    df_plot['coverage'] = df_plot['condition'].map(coverage_map)
    df_plot['condition_label'] = df_plot['condition'].map(condition_labels)
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 散布図
    scatter = ax.scatter(df_plot['coverage'], df_plot['mcd'], 
                        s=200, alpha=0.7, c=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # ラベルを追加
    for idx, row in df_plot.iterrows():
        ax.annotate(row['condition_label'], 
                   (row['coverage'], row['mcd']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Phoneme Coverage (%)', fontsize=12)
    ax.set_ylabel('MCD (dB)', fontsize=12)
    ax.set_title('Phoneme Coverage vs MCD', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Phoneme coverage vs MCD scatter plot saved: {output_path}")


def plot_datasize_vs_mcd(df: pd.DataFrame, output_path: Path):
    """データ量とMCDの散布図"""
    condition_labels = get_condition_labels()
    
    # データ量（文数）を取得
    datasize_map = {
        "train_80sent": 80,
        "train_4sent_37phonemes": 4,
        "train_4sent_random": 4,
        "train_10sent_top": 10,
    }
    
    # データ準備
    df_plot = df.copy()
    df_plot['datasize'] = df_plot['condition'].map(datasize_map)
    df_plot['condition_label'] = df_plot['condition'].map(condition_labels)
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 散布図
    scatter = ax.scatter(df_plot['datasize'], df_plot['mcd'],
                        s=200, alpha=0.7, c=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # ラベルを追加
    for idx, row in df_plot.iterrows():
        ax.annotate(row['condition_label'],
                   (row['datasize'], row['mcd']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Training Data Size (Number of Sentences)', fontsize=12)
    ax.set_ylabel('MCD (dB)', fontsize=12)
    ax.set_title('Data Size vs MCD', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')  # 対数スケール
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Data size vs MCD scatter plot saved: {output_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing evaluation results (default: results)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/figures",
        help="Directory to save figures (default: outputs/figures)"
    )
    
    args = parser.parse_args()
    
    # パス解決
    project_root = Path(__file__).parent.parent
    results_dir = project_root / args.results_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading evaluation results...")
    df = load_evaluation_results(results_dir)
    print(f"Evaluation results: {len(df)} conditions")
    
    print("Loading corpus selection results...")
    corpus_data = load_corpus_selection(results_dir)
    
    # Generate graphs
    print("\nGenerating graphs...")
    plot_mcd_comparison(df, output_dir / "mcd_comparison.png")
    plot_f0_comparison(df, output_dir / "f0_comparison.png")
    plot_coverage_vs_mcd(df, corpus_data, output_dir / "coverage_vs_mcd.png")
    plot_datasize_vs_mcd(df, output_dir / "datasize_vs_mcd.png")
    
    print("\nAll graphs generated successfully!")


if __name__ == "__main__":
    main()
