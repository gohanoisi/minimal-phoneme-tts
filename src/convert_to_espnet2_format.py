#!/usr/bin/env python3
"""
ESPnet2形式（Kaldi形式）へのデータ変換スクリプト
JSONL形式（data.list）からESPnet2のKaldi形式に変換する。
"""

import json
from pathlib import Path
from typing import Dict, List


def load_data_list(data_list_path: Path) -> List[Dict]:
    """
    data.list（JSONL形式）を読み込む
    
    Args:
        data_list_path: data.listファイルのパス
        
    Returns:
        データエントリのリスト
    """
    entries = []
    with open(data_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def create_wav_scp(entries: List[Dict], output_path: Path) -> None:
    """
    wav.scpファイルを作成
    形式: <utt_id> <wav_path>
    
    Args:
        entries: データエントリのリスト
        output_path: 出力ファイルのパス
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            utt_id = entry['utt_id']
            wav_path = entry['audio']
            # 絶対パスを使用
            wav_path = str(Path(wav_path).resolve())
            f.write(f"{utt_id} {wav_path}\n")


def create_text(entries: List[Dict], output_path: Path) -> None:
    """
    textファイルを作成
    形式: <utt_id> <phoneme_sequence>
    
    Args:
        entries: データエントリのリスト
        output_path: 出力ファイルのパス
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            utt_id = entry['utt_id']
            phoneme = entry['phoneme']
            f.write(f"{utt_id} {phoneme}\n")


def create_utt2spk(entries: List[Dict], output_path: Path) -> None:
    """
    utt2spkファイルを作成
    形式: <utt_id> <speaker_id>
    
    Args:
        entries: データエントリのリスト
        output_path: 出力ファイルのパス
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            utt_id = entry['utt_id']
            speaker = entry['speaker']
            f.write(f"{utt_id} {speaker}\n")


def create_spk2utt(entries: List[Dict], output_path: Path) -> None:
    """
    spk2uttファイルを作成
    形式: <speaker_id> <utt_id1> <utt_id2> ...
    
    Args:
        entries: データエントリのリスト
        output_path: 出力ファイルのパス
    """
    # 話者ごとに発話IDをグループ化
    speaker_to_utts: Dict[str, List[str]] = {}
    for entry in entries:
        speaker = entry['speaker']
        utt_id = entry['utt_id']
        if speaker not in speaker_to_utts:
            speaker_to_utts[speaker] = []
        speaker_to_utts[speaker].append(utt_id)
    
    # ソートして出力
    with open(output_path, 'w', encoding='utf-8') as f:
        for speaker in sorted(speaker_to_utts.keys()):
            utt_ids = sorted(speaker_to_utts[speaker])
            f.write(f"{speaker} {' '.join(utt_ids)}\n")


def convert_data_list_to_espnet2_format(
    data_list_path: Path,
    output_data_dir: Path
) -> None:
    """
    data.listをESPnet2形式（Kaldi形式）に変換
    
    Args:
        data_list_path: 入力data.listファイルのパス
        output_data_dir: 出力データディレクトリのパス
    """
    # 出力ディレクトリを作成
    output_data_dir.mkdir(parents=True, exist_ok=True)
    
    # data.listを読み込み
    print(f"Loading data.list from: {data_list_path}")
    entries = load_data_list(data_list_path)
    print(f"  Loaded {len(entries)} entries")
    
    # ESPnet2形式のファイルを生成
    wav_scp_path = output_data_dir / "wav.scp"
    text_path = output_data_dir / "text"
    utt2spk_path = output_data_dir / "utt2spk"
    spk2utt_path = output_data_dir / "spk2utt"
    
    print(f"Creating ESPnet2 format files in: {output_data_dir}")
    create_wav_scp(entries, wav_scp_path)
    create_text(entries, text_path)
    create_utt2spk(entries, utt2spk_path)
    create_spk2utt(entries, spk2utt_path)
    
    print(f"  Created: wav.scp ({len(entries)} entries)")
    print(f"  Created: text ({len(entries)} entries)")
    print(f"  Created: utt2spk ({len(entries)} entries)")
    
    # 話者数を確認
    speakers = set(entry['speaker'] for entry in entries)
    print(f"  Created: spk2utt ({len(speakers)} speakers)")
    print(f"Conversion completed!")


def main():
    """メイン処理"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # 変換対象のデータセット
    datasets = [
        ("train_80sent", "train_80sent"),
        ("train_4sent_37phonemes", "train_4sent_37phonemes"),
        ("train_4sent_random", "train_4sent_random"),
        ("train_10sent_top", "train_10sent_top"),
        ("test", "test"),
    ]
    
    print("=" * 60)
    print("ESPnet2形式へのデータ変換")
    print("=" * 60)
    
    for input_dirname, output_dirname in datasets:
        input_data_dir = data_dir / input_dirname
        data_list_path = input_data_dir / "data.list"
        
        if not data_list_path.exists():
            print(f"\nWarning: {data_list_path} not found. Skipping...")
            continue
        
        output_data_dir = data_dir / output_dirname
        
        print(f"\n{'-' * 60}")
        print(f"Converting: {input_dirname}")
        print(f"{'-' * 60}")
        
        convert_data_list_to_espnet2_format(data_list_path, output_data_dir)
    
    print("\n" + "=" * 60)
    print("All conversions completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
