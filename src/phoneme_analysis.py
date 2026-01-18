#!/usr/bin/env python3
"""
音素分析スクリプト
JVS parallel100 jvs002話者の100文から音素分布を抽出し、37音素インベントリを確認する。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set
import pyopenjtalk
from collections import Counter
import pandas as pd


def extract_phonemes(text: str) -> List[str]:
    """
    テキストから音素列を抽出する
    
    Args:
        text: 入力テキスト
        
    Returns:
        音素列のリスト
    """
    try:
        # pyopenjtalkで音素列を取得（kana=Falseで音素列を取得）
        phonemes = pyopenjtalk.g2p(text, kana=False)
        # 空白で分割してリスト化
        phoneme_list = phonemes.split()
        return phoneme_list
    except Exception as e:
        print(f"Error processing text '{text}': {e}")
        return []


def analyze_phoneme_distribution(texts: Dict[str, str]) -> Dict:
    """
    100文の音素分布を分析する
    
    Args:
        texts: 文ID -> テキストの辞書
        
    Returns:
        音素分布データの辞書
    """
    all_phonemes = []
    sentence_phonemes = {}
    phoneme_inventory = set()
    
    for sentence_id, text in texts.items():
        phonemes = extract_phonemes(text)
        sentence_phonemes[sentence_id] = phonemes
        all_phonemes.extend(phonemes)
        phoneme_inventory.update(phonemes)
    
    # 全音素の頻度分布
    phoneme_freq = Counter(all_phonemes)
    
    # 各文のユニーク音素数
    sentence_stats = {}
    for sentence_id, phonemes in sentence_phonemes.items():
        unique_phonemes = set(phonemes)
        sentence_stats[sentence_id] = {
            "text": texts[sentence_id],
            "phonemes": phonemes,
            "unique_phoneme_count": len(unique_phonemes),
            "total_phoneme_count": len(phonemes),
            "unique_phonemes": list(unique_phonemes)
        }
    
    return {
        "phoneme_inventory": sorted(list(phoneme_inventory)),
        "phoneme_inventory_size": len(phoneme_inventory),
        "phoneme_frequency": dict(phoneme_freq),
        "sentence_stats": sentence_stats,
        "total_sentences": len(texts)
    }


def load_jvs_texts(data_dir: Path) -> Dict[str, str]:
    """
    JVS parallel100のテキストファイルを読み込む
    
    Args:
        data_dir: JVSデータのディレクトリパス
        
    Returns:
        文ID -> テキストの辞書
    """
    texts = {}
    
    # JVS parallel100の構造を想定: data/jvs002/parallel100/wav24kHz16bit/*.wav
    # テキストファイルは通常、同じディレクトリまたは親ディレクトリに配置される
    text_dir = data_dir / "parallel100"
    
    # テキストファイルのパターンを探す
    possible_text_files = [
        text_dir / "transcripts_utf8.txt",  # 一般的なJVSのテキストファイル名
        text_dir / "text.txt",
        data_dir / "transcripts_utf8.txt",
        data_dir / "text.txt"
    ]
    
    text_file = None
    for path in possible_text_files:
        if path.exists():
            text_file = path
            break
    
    if text_file is None:
        # テキストファイルが見つからない場合、音声ファイルから推測
        # または手動でテキストを読み込む必要がある
        print(f"Warning: Text file not found. Searching in {data_dir}")
        # ここでは空の辞書を返す（後で手動でテキストを追加する必要がある）
        return texts
    
    # テキストファイルを読み込む
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # JVSの形式: "文ID:テキスト" または "文ID\tテキスト" または "文ID テキスト"
            if ':' in line:
                parts = line.split(':', 1)
            elif '\t' in line:
                parts = line.split('\t', 1)
            else:
                parts = line.split(' ', 1)
            
            if len(parts) == 2:
                sentence_id, text = parts
                texts[sentence_id] = text
    
    return texts


def main():
    """メイン処理"""
    # パス設定
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "jvs002"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("音素分析スクリプト")
    print("=" * 60)
    
    # JVSデータの確認
    if not data_dir.exists():
        print(f"Warning: JVS data directory not found: {data_dir}")
        print("Please download JVS parallel100 data and place it in data/jvs002/")
        print("\nFor now, creating a sample structure...")
        # サンプルデータ構造を作成（後で実際のデータに置き換える）
        return
    
    # テキストファイルの読み込み
    print(f"Loading texts from: {data_dir}")
    texts = load_jvs_texts(data_dir)
    
    if not texts:
        print("Warning: No texts found. Please check the data directory structure.")
        print("Expected structure: data/jvs002/parallel100/transcripts_utf8.txt")
        return
    
    print(f"Loaded {len(texts)} sentences")
    
    # 音素分析
    print("\nAnalyzing phoneme distribution...")
    analysis_result = analyze_phoneme_distribution(texts)
    
    # 結果の表示
    print(f"\n音素インベントリサイズ: {analysis_result['phoneme_inventory_size']}")
    print(f"総文数: {analysis_result['total_sentences']}")
    print(f"\n音素一覧:")
    print(", ".join(analysis_result['phoneme_inventory']))
    
    # 結果の保存
    output_file = results_dir / "phoneme_distribution.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # CSV形式でも保存（可読性のため）
    csv_data = []
    for sentence_id, stats in analysis_result['sentence_stats'].items():
        csv_data.append({
            "sentence_id": sentence_id,
            "text": stats['text'],
            "unique_phoneme_count": stats['unique_phoneme_count'],
            "total_phoneme_count": stats['total_phoneme_count'],
            "unique_phonemes": ", ".join(stats['unique_phonemes'])
        })
    
    df = pd.DataFrame(csv_data)
    csv_file = results_dir / "phoneme_distribution.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"CSV results saved to: {csv_file}")
    
    print("\n" + "=" * 60)
    print("音素分析完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
