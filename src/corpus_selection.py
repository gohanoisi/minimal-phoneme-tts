#!/usr/bin/env python3
"""
コーパス選定スクリプト
音素分布データから4条件のコーパス（80文/37音素4文/ランダム4文/上位10文）を選定する。
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd


def load_phoneme_distribution(results_dir: Path) -> Dict:
    """
    音素分布データを読み込む
    
    Args:
        results_dir: 結果ディレクトリのパス
        
    Returns:
        音素分布データの辞書
    """
    distribution_file = results_dir / "phoneme_distribution.json"
    
    if not distribution_file.exists():
        raise FileNotFoundError(
            f"Phoneme distribution file not found: {distribution_file}\n"
            "Please run phoneme_analysis.py first."
        )
    
    with open(distribution_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def select_80_sentences(sentence_stats: Dict, seed: int = 42) -> List[str]:
    """
    80文コーパスを選定（8:2分割、ランダムシード固定）
    
    Args:
        sentence_stats: 文の統計情報
        seed: 乱数シード
        
    Returns:
        選定された文IDのリスト
    """
    all_sentence_ids = list(sentence_stats.keys())
    random.seed(seed)
    selected = random.sample(all_sentence_ids, 80)
    return sorted(selected)


def select_37_phoneme_4_sentences(
    sentence_stats: Dict,
    phoneme_inventory: List[str],
    seed: int = 42
) -> List[str]:
    """
    37音素を全てカバーする4文を選定（貪欲法）
    
    Args:
        sentence_stats: 文の統計情報
        phoneme_inventory: 全37音素のリスト
        seed: 乱数シード（同点の場合の選択に使用）
        
    Returns:
        選定された文IDのリスト（4文）
    """
    target_phonemes = set(phoneme_inventory)
    selected_sentences = []
    covered_phonemes = set()
    
    random.seed(seed)
    remaining_sentences = list(sentence_stats.items())
    random.shuffle(remaining_sentences)  # 同点の場合の順序をランダム化
    
    while len(covered_phonemes) < len(target_phonemes) and len(selected_sentences) < 4:
        best_sentence_id = None
        best_new_phonemes = set()
        
        for sentence_id, stats in remaining_sentences:
            if sentence_id in selected_sentences:
                continue
            
            sentence_phonemes = set(stats['unique_phonemes'])
            new_phonemes = sentence_phonemes - covered_phonemes
            
            if len(new_phonemes) > len(best_new_phonemes):
                best_sentence_id = sentence_id
                best_new_phonemes = new_phonemes
        
        if best_sentence_id is None:
            # これ以上カバーできる文がない
            break
        
        selected_sentences.append(best_sentence_id)
        covered_phonemes.update(best_new_phonemes)
        
        # 選択済みの文を除外
        remaining_sentences = [
            (sid, stats) for sid, stats in remaining_sentences
            if sid != best_sentence_id
        ]
    
    # 37音素を全てカバーできたか確認
    if len(covered_phonemes) < len(target_phonemes):
        missing = target_phonemes - covered_phonemes
        print(f"Warning: Could not cover all 37 phonemes with 4 sentences.")
        print(f"Missing phonemes: {missing}")
        print(f"Covered: {len(covered_phonemes)}/{len(target_phonemes)}")
    
    return sorted(selected_sentences)


def select_random_4_sentences(sentence_stats: Dict, seed: int = 123) -> List[str]:
    """
    ユニーク音素数が少ない4文を選定（低カバレッジの対照群）
    
    変更点: ランダム選定ではなく、ユニーク音素数が少ない文を優先
    目的: 音素カバレッジが低い場合の影響を確認する対照実験
    目標カバレッジ: 20-25音素（約50-65%）
    
    Args:
        sentence_stats: 文の統計情報
        seed: 乱数シード（同点の場合の順序ランダム化に使用）
        
    Returns:
        選定された文IDのリスト（4文）
    """
    # 各文のユニーク音素数でソート
    sentence_scores = [
        (sentence_id, len(stats['unique_phonemes']))
        for sentence_id, stats in sentence_stats.items()
    ]
    sentence_scores.sort(key=lambda x: x[1])  # 音素数が少ない順
    
    # シード固定でランダム性を加える（同点の場合の順序をランダム化）
    # ただし、音素数が少ない文を優先
    selected = []
    covered_phonemes = set()
    target_max_coverage = 25  # 目標: 25音素以下
    
    random.seed(seed)
    for sentence_id, phoneme_count in sentence_scores:
        if len(selected) >= 4:
            break
        
        stats = sentence_stats[sentence_id]
        sentence_phonemes = set(stats['unique_phonemes'])
        new_phonemes = sentence_phonemes - covered_phonemes
        new_coverage_count = len(covered_phonemes | sentence_phonemes)
        
        # 目標カバレッジを超えないように制御
        if new_coverage_count <= target_max_coverage:
            selected.append(sentence_id)
            covered_phonemes.update(sentence_phonemes)
        elif len(selected) == 0:
            # 最初の文は必ず追加（最小音素数の文）
            selected.append(sentence_id)
            covered_phonemes.update(sentence_phonemes)
    
    # 4文に満たない場合は、目標カバレッジを超えても追加
    if len(selected) < 4:
        for sentence_id, phoneme_count in sentence_scores:
            if sentence_id in selected:
                continue
            if len(selected) >= 4:
                break
            selected.append(sentence_id)
            stats = sentence_stats[sentence_id]
            covered_phonemes.update(stats['unique_phonemes'])
    
    return sorted(selected)


def calculate_phoneme_feature_score(
    sentence_id: str,
    sentence_stats: Dict,
    phoneme_frequency: Dict[str, int]
) -> float:
    """
    音素特徴量スコアを計算（ユニーク音素数 + レア音素数）
    
    Args:
        sentence_id: 文ID
        sentence_stats: 文の統計情報
        phoneme_frequency: 音素の頻度分布
        
    Returns:
        スコア（高いほど良い）
    """
    stats = sentence_stats[sentence_id]
    unique_phonemes = set(stats['unique_phonemes'])
    
    # ユニーク音素数
    unique_count = len(unique_phonemes)
    
    # レア音素の含有数（頻度が低い音素ほど高スコア）
    # 頻度の中央値を計算
    frequencies = list(phoneme_frequency.values())
    median_freq = sorted(frequencies)[len(frequencies) // 2] if frequencies else 0
    
    # レア音素（頻度が中央値以下の音素）の数
    rare_phoneme_count = sum(
        1 for phoneme in unique_phonemes
        if phoneme_frequency.get(phoneme, 0) <= median_freq
    )
    
    # スコア = ユニーク音素数 + レア音素数
    score = unique_count + rare_phoneme_count
    
    return score


def select_top_10_sentences(
    sentence_stats: Dict,
    phoneme_frequency: Dict[str, int]
) -> List[str]:
    """
    音素特徴量上位10文を選定
    
    Args:
        sentence_stats: 文の統計情報
        phoneme_frequency: 音素の頻度分布
        
    Returns:
        選定された文IDのリスト（10文）
    """
    scores = []
    
    for sentence_id in sentence_stats.keys():
        score = calculate_phoneme_feature_score(
            sentence_id, sentence_stats, phoneme_frequency
        )
        scores.append((sentence_id, score))
    
    # スコアでソート（降順）
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # 上位10文を取得
    top_10 = [sentence_id for sentence_id, _ in scores[:10]]
    
    return sorted(top_10)


def calculate_coverage(sentence_ids: List[str], sentence_stats: Dict) -> Dict:
    """
    選定されたコーパスの音素カバレッジを計算
    
    Args:
        sentence_ids: 選定された文IDのリスト
        sentence_stats: 文の統計情報
        
    Returns:
        カバレッジ情報の辞書
    """
    covered_phonemes = set()
    total_phonemes = []
    
    for sentence_id in sentence_ids:
        stats = sentence_stats[sentence_id]
        covered_phonemes.update(stats['unique_phonemes'])
        total_phonemes.extend(stats['phonemes'])
    
    return {
        "unique_phoneme_count": len(covered_phonemes),
        "total_phoneme_count": len(total_phonemes),
        "covered_phonemes": sorted(list(covered_phonemes)),
        "sentence_count": len(sentence_ids)
    }


def main():
    """メイン処理"""
    # パス設定
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("コーパス選定スクリプト")
    print("=" * 60)
    
    # 音素分布データの読み込み
    print("Loading phoneme distribution data...")
    distribution_data = load_phoneme_distribution(results_dir)
    
    sentence_stats = distribution_data['sentence_stats']
    phoneme_inventory = distribution_data['phoneme_inventory']
    phoneme_frequency = distribution_data['phoneme_frequency']
    
    print(f"Loaded {len(sentence_stats)} sentences")
    print(f"Phoneme inventory size: {len(phoneme_inventory)}")
    
    # 4条件のコーパス選定
    print("\n" + "-" * 60)
    print("Selecting corpora for 4 conditions...")
    print("-" * 60)
    
    # E1: 80文コーパス
    print("\n[E1] Selecting 80 sentences (8:2 split)...")
    e1_sentences = select_80_sentences(sentence_stats, seed=42)
    e1_coverage = calculate_coverage(e1_sentences, sentence_stats)
    print(f"  Selected {len(e1_sentences)} sentences")
    print(f"  Coverage: {e1_coverage['unique_phoneme_count']} unique phonemes")
    
    # E2: 37音素4文
    print("\n[E2] Selecting 4 sentences covering all 37 phonemes...")
    e2_sentences = select_37_phoneme_4_sentences(
        sentence_stats, phoneme_inventory, seed=42
    )
    e2_coverage = calculate_coverage(e2_sentences, sentence_stats)
    print(f"  Selected {len(e2_sentences)} sentences")
    print(f"  Coverage: {e2_coverage['unique_phoneme_count']} unique phonemes")
    if e2_coverage['unique_phoneme_count'] == len(phoneme_inventory):
        print(f"  ✓ All {len(phoneme_inventory)} phonemes covered!")
    else:
        print(f"  ⚠ Could not cover all phonemes")
    
    # E3: ランダム4文
    print("\n[E3] Selecting 4 random sentences...")
    e3_sentences = select_random_4_sentences(sentence_stats, seed=123)
    e3_coverage = calculate_coverage(e3_sentences, sentence_stats)
    print(f"  Selected {len(e3_sentences)} sentences")
    print(f"  Coverage: {e3_coverage['unique_phoneme_count']} unique phonemes")
    
    # E4: 上位10文
    print("\n[E4] Selecting top 10 sentences by phoneme features...")
    e4_sentences = select_top_10_sentences(sentence_stats, phoneme_frequency)
    e4_coverage = calculate_coverage(e4_sentences, sentence_stats)
    print(f"  Selected {len(e4_sentences)} sentences")
    print(f"  Coverage: {e4_coverage['unique_phoneme_count']} unique phonemes")
    
    # 結果の整理
    corpus_selection = {
        "E1_80sent": {
            "name": "80文コーパス",
            "sentence_ids": e1_sentences,
            "coverage": e1_coverage
        },
        "E2_4sent_37phonemes": {
            "name": "37音素4文",
            "sentence_ids": e2_sentences,
            "coverage": e2_coverage
        },
        "E3_4sent_random": {
            "name": "ランダム4文",
            "sentence_ids": e3_sentences,
            "coverage": e3_coverage
        },
        "E4_10sent_top": {
            "name": "上位10文",
            "sentence_ids": e4_sentences,
            "coverage": e4_coverage
        },
        "test_set": {
            "name": "テストセット",
            "sentence_ids": [],  # 後で設定
            "coverage": {}
        }
    }
    
    # テストセットの選定（4条件すべてで未学習となる文）
    all_train_sentences = set()
    for condition in ["E1_80sent", "E2_4sent_37phonemes", "E3_4sent_random", "E4_10sent_top"]:
        all_train_sentences.update(corpus_selection[condition]["sentence_ids"])
    
    all_sentence_ids = set(sentence_stats.keys())
    test_sentences = sorted(list(all_sentence_ids - all_train_sentences))
    
    # テストセットが少ない場合は、E1の20文をテストセットとして使用
    if len(test_sentences) < 10:
        print("\nWarning: Test set is too small. Using 20 sentences from E1 as test set.")
        all_sentence_ids_list = list(sentence_stats.keys())
        random.seed(999)
        test_sentences = random.sample(all_sentence_ids_list, 20)
        test_sentences = sorted(test_sentences)
    
    corpus_selection["test_set"]["sentence_ids"] = test_sentences
    corpus_selection["test_set"]["coverage"] = calculate_coverage(
        test_sentences, sentence_stats
    )
    
    print(f"\n[Test Set] Selected {len(test_sentences)} sentences")
    print(f"  Coverage: {corpus_selection['test_set']['coverage']['unique_phoneme_count']} unique phonemes")
    
    # 結果の保存
    output_file = results_dir / "corpus_selection.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(corpus_selection, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # サマリーの表示
    print("\n" + "=" * 60)
    print("Corpus Selection Summary")
    print("=" * 60)
    for condition_id, condition_data in corpus_selection.items():
        if condition_id == "test_set":
            continue
        print(f"\n{condition_data['name']} ({condition_id}):")
        print(f"  Sentences: {len(condition_data['sentence_ids'])}")
        print(f"  Unique phonemes: {condition_data['coverage']['unique_phoneme_count']}")
        print(f"  Total phonemes: {condition_data['coverage']['total_phoneme_count']}")
    
    print("\n" + "=" * 60)
    print("コーパス選定完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
