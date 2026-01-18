#!/usr/bin/env python3
"""
データ前処理スクリプト
JVS parallel100データをESPnet2形式に変換し、4条件それぞれの学習用データセットを準備する。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pyopenjtalk
import soundfile as sf
import librosa


def load_corpus_selection(results_dir: Path) -> Dict:
    """
    コーパス選定結果を読み込む
    
    Args:
        results_dir: 結果ディレクトリのパス
        
    Returns:
        コーパス選定データの辞書
    """
    selection_file = results_dir / "corpus_selection.json"
    
    if not selection_file.exists():
        raise FileNotFoundError(
            f"Corpus selection file not found: {selection_file}\n"
            "Please run corpus_selection.py first."
        )
    
    with open(selection_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_audio_files(data_dir: Path, sentence_ids: List[str]) -> Dict[str, Path]:
    """
    音声ファイルを検索する
    
    Args:
        data_dir: JVSデータのディレクトリパス
        sentence_ids: 文IDのリスト
        
    Returns:
        文ID -> 音声ファイルパスの辞書
    """
    audio_files = {}
    
    # JVS parallel100の構造を想定
    # data/jvs002/parallel100/wav24kHz16bit/*.wav
    wav_dir = data_dir / "parallel100" / "wav24kHz16bit"
    
    if not wav_dir.exists():
        # 別の構造を試す
        wav_dir = data_dir / "wav24kHz16bit"
    
    if not wav_dir.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {wav_dir}\n"
            "Please check the JVS data directory structure."
        )
    
    # 音声ファイルを検索
    for sentence_id in sentence_ids:
        # 文IDに対応する音声ファイルを探す
        # JVSの形式: "VOICEACTRESS100_001.wav" など
        # 文IDが既に "VOICEACTRESS100_001" の形式の場合
        possible_names = [
            f"{sentence_id}.wav",  # そのまま
            f"jvs002_{sentence_id.split('_')[-1]}.wav",  # 番号部分のみ
            f"VOICEACTRESS100_{sentence_id.split('_')[-1]}.wav"  # 番号部分のみ
        ]
        
        found = False
        for name in possible_names:
            audio_path = wav_dir / name
            if audio_path.exists():
                audio_files[sentence_id] = audio_path
                found = True
                break
        
        if not found:
            # ワイルドカードで検索
            pattern_files = list(wav_dir.glob(f"*{sentence_id}*.wav"))
            if pattern_files:
                audio_files[sentence_id] = pattern_files[0]
            else:
                print(f"Warning: Audio file not found for sentence {sentence_id}")
    
    return audio_files


def load_texts(data_dir: Path) -> Dict[str, str]:
    """
    テキストファイルを読み込む
    
    Args:
        data_dir: JVSデータのディレクトリパス
        
    Returns:
        文ID -> テキストの辞書
    """
    texts = {}
    
    # テキストファイルのパターンを探す
    possible_text_files = [
        data_dir / "parallel100" / "transcripts_utf8.txt",
        data_dir / "parallel100" / "text.txt",
        data_dir / "transcripts_utf8.txt",
        data_dir / "text.txt"
    ]
    
    text_file = None
    for path in possible_text_files:
        if path.exists():
            text_file = path
            break
    
    if text_file is None:
        raise FileNotFoundError(
            f"Text file not found. Searched in: {possible_text_files}"
        )
    
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


def text_to_phoneme(text: str) -> str:
    """
    テキストを音素列に変換する
    
    Args:
        text: 入力テキスト
        
    Returns:
        音素列（スペース区切り）
    """
    try:
        phonemes = pyopenjtalk.g2p(text, kana=False)
        # 空白で分割してリスト化し、再度結合（正規化）
        phoneme_list = phonemes.split()
        return " ".join(phoneme_list)
    except Exception as e:
        print(f"Error converting text to phoneme: {text}, error: {e}")
        return ""


def check_audio_file(audio_path: Path) -> Dict:
    """
    音声ファイルの情報を確認する
    
    Args:
        audio_path: 音声ファイルのパス
        
    Returns:
        音声ファイル情報の辞書
    """
    try:
        info = sf.info(audio_path)
        return {
            "samplerate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "frames": info.frames,
            "subtype": info.subtype
        }
    except Exception as e:
        print(f"Error reading audio file {audio_path}: {e}")
        return {}


def create_espnet_data_list(
    sentence_ids: List[str],
    audio_files: Dict[str, Path],
    texts: Dict[str, str],
    speaker_id: str = "jvs002"
) -> List[Dict]:
    """
    ESPnet2形式のデータリストを作成する
    
    Args:
        sentence_ids: 文IDのリスト
        audio_files: 文ID -> 音声ファイルパスの辞書
        texts: 文ID -> テキストの辞書
        speaker_id: 話者ID
        
    Returns:
        データリスト（辞書のリスト）
    """
    data_list = []
    
    for sentence_id in sentence_ids:
        if sentence_id not in audio_files:
            print(f"Warning: Audio file not found for {sentence_id}, skipping")
            continue
        
        if sentence_id not in texts:
            print(f"Warning: Text not found for {sentence_id}, skipping")
            continue
        
        audio_path = audio_files[sentence_id]
        text = texts[sentence_id]
        phoneme = text_to_phoneme(text)
        
        # 音声ファイルの絶対パス
        abs_audio_path = audio_path.resolve()
        
        data_list.append({
            "utt_id": sentence_id,
            "speaker": speaker_id,
            "text": text,
            "phoneme": phoneme,
            "audio": str(abs_audio_path)
        })
    
    return data_list


def save_data_list(data_list: List[Dict], output_file: Path):
    """
    データリストをESPnet2形式で保存する
    
    Args:
        data_list: データリスト
        output_file: 出力ファイルパス
    """
    # ESPnet2のdata.list形式（JSONL形式）
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    """メイン処理"""
    # パス設定
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "jvs002"
    results_dir = project_root / "results"
    output_data_dir = project_root / "data"
    
    print("=" * 60)
    print("データ前処理スクリプト")
    print("=" * 60)
    
    # JVSデータの確認
    if not data_dir.exists():
        print(f"Error: JVS data directory not found: {data_dir}")
        print("Please download JVS parallel100 data and place it in data/jvs002/")
        return
    
    # コーパス選定結果の読み込み
    print("Loading corpus selection results...")
    corpus_selection = load_corpus_selection(results_dir)
    
    # テキストファイルの読み込み
    print("Loading text files...")
    texts = load_texts(data_dir)
    print(f"Loaded {len(texts)} texts")
    
    # 4条件それぞれのデータリストを作成
    conditions = {
        "E1_80sent": "train_80sent",
        "E2_4sent_37phonemes": "train_4sent_37phonemes",
        "E3_4sent_random": "train_4sent_random",
        "E4_10sent_top": "train_10sent_top",
        "test_set": "test"
    }
    
    for condition_id, condition_name in conditions.items():
        if condition_id not in corpus_selection:
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Processing {corpus_selection[condition_id]['name']} ({condition_id})")
        print(f"{'=' * 60}")
        
        sentence_ids = corpus_selection[condition_id]["sentence_ids"]
        print(f"Number of sentences: {len(sentence_ids)}")
        
        # 音声ファイルの検索
        print("Finding audio files...")
        audio_files = find_audio_files(data_dir, sentence_ids)
        print(f"Found {len(audio_files)} audio files")
        
        # データリストの作成
        print("Creating data list...")
        data_list = create_espnet_data_list(sentence_ids, audio_files, texts)
        print(f"Created {len(data_list)} data entries")
        
        # データディレクトリの作成
        condition_data_dir = output_data_dir / condition_name
        condition_data_dir.mkdir(parents=True, exist_ok=True)
        
        # データリストの保存
        output_file = condition_data_dir / "data.list"
        save_data_list(data_list, output_file)
        print(f"Saved data list to: {output_file}")
        
        # 音声ファイル情報の確認（最初の数ファイルのみ）
        if audio_files:
            print("\nAudio file information (first 3 files):")
            for i, (sentence_id, audio_path) in enumerate(list(audio_files.items())[:3]):
                info = check_audio_file(audio_path)
                if info:
                    print(f"  {sentence_id}:")
                    print(f"    Sample rate: {info['samplerate']} Hz")
                    print(f"    Channels: {info['channels']}")
                    print(f"    Duration: {info['duration']:.2f} sec")
    
    print("\n" + "=" * 60)
    print("データ前処理完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
