#!/usr/bin/env python3
"""
Fine-tuningスクリプト
各条件のコーパスで事前学習済み日本語TTSモデルをfine-tuningする。
"""

import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed: int = 42):
    """
    乱数シードを固定する
    
    Args:
        seed: 乱数シード
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def load_data_list(data_list_path: Path) -> list:
    """
    データリストを読み込む
    
    Args:
        data_list_path: データリストファイルのパス
        
    Returns:
        データリスト
    """
    data_list = []
    with open(data_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))
    return data_list


def check_gpu():
    """
    GPUの利用可能性を確認する
    
    Returns:
        GPUが利用可能かどうか
    """
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    else:
        print("GPU not available, using CPU")
        return False


def run_espnet2_tts_train(
    condition_name: str,
    train_data_dir: Path,
    valid_data_dir: Path,
    output_dir: Path,
    config_file: Path,
    pretrained_model: Optional[str] = None,
    collect_stats: bool = False,
    seed: int = 42
) -> int:
    """
    ESPnet2のtts_trainを実行する
    
    Args:
        condition_name: 条件名
        train_data_dir: 学習データディレクトリ（Kaldi形式）
        valid_data_dir: 検証データディレクトリ（Kaldi形式）
        output_dir: 出力ディレクトリ
        config_file: 設定ファイル（YAML）のパス
        pretrained_model: 事前学習モデルのパス（Fine-tuningの場合）
        collect_stats: 統計情報収集モードの場合True
        seed: 乱数シード
        
    Returns:
        終了コード（0: 成功、非0: 失敗）
    """
    # 出力ディレクトリの作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ESPnet2のtts_trainコマンドを構築
    cmd = [
        sys.executable, "-m", "espnet2.bin.tts_train",
        "--config", str(config_file),
        "--train_data_path_and_name_and_type", f"{train_data_dir}/wav.scp,speech,sound",
        "--train_data_path_and_name_and_type", f"{train_data_dir}/text,text,text",
        "--valid_data_path_and_name_and_type", f"{valid_data_dir}/wav.scp,speech,sound",
        "--valid_data_path_and_name_and_type", f"{valid_data_dir}/text,text,text",
        "--output_dir", str(output_dir),
        "--ngpu", "1" if torch.cuda.is_available() else "0",
        "--seed", str(seed),
    ]
    
    # 統計情報収集モードの場合
    if collect_stats:
        cmd.append("--collect_stats")
        cmd.append("true")
        print(f"[Step 1] Collecting statistics for {condition_name}...")
    else:
        # Fine-tuningの場合、事前学習モデルを指定
        if pretrained_model:
            # 事前学習モデルのパス形式: model_path:model_name:model_name
            # 例: ./downloads/kan-bayashi_jsut_tacotron2_accent_with_pause/.../train.loss.ave_5best.pth:tts:tts
            cmd.append("--init_param")
            cmd.append(f"{pretrained_model}:tts:tts")
        print(f"[Step 2] Fine-tuning {condition_name}...")
    
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    # コマンドを実行
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        return result.returncode
    except Exception as e:
        print(f"Error running ESPnet2 training: {e}")
        return 1


def train_condition(
    condition_name: str,
    condition_dir_name: str,
    train_data_dir: Path,
    valid_data_dir: Path,
    output_dir: Path,
    config_file: Path,
    pretrained_model: Optional[str] = None,
    logs_dir: Path = None,
    seed: int = 42
) -> Dict:
    """
    1つの条件に対して統計情報収集とFine-tuningを実行する
    
    Args:
        condition_name: 条件名（表示用）
        condition_dir_name: 条件ディレクトリ名（例: "train_80sent"）
        train_data_dir: 学習データディレクトリ（Kaldi形式）
        valid_data_dir: 検証データディレクトリ（Kaldi形式）
        output_dir: 出力ディレクトリ
        config_file: 設定ファイル（YAML）のパス
        pretrained_model: 事前学習モデルのパス
        logs_dir: ログ保存ディレクトリ
        seed: 乱数シード
        
    Returns:
        学習結果の辞書（開始時刻、終了時刻、総学習時間、成功フラグなど）
    """
    print("=" * 60)
    print(f"Training condition: {condition_name}")
    print("=" * 60)
    
    # 乱数シード固定
    set_seed(seed)
    
    # GPU確認
    check_gpu()
    
    # 学習時間記録用
    start_time = datetime.now()
    result = {
        "condition_name": condition_name,
        "condition_dir_name": condition_dir_name,
        "start_time": start_time.isoformat(),
        "end_time": None,
        "total_time_seconds": None,
        "stats_collection_success": False,
        "fine_tuning_success": False,
    }
    
    # Step 1: 統計情報収集
    stats_output_dir = output_dir / f"{condition_dir_name}" / "stats"
    stats_start = time.time()
    stats_code = run_espnet2_tts_train(
        condition_name=f"{condition_name} (Stats Collection)",
        train_data_dir=train_data_dir,
        valid_data_dir=valid_data_dir,
        output_dir=stats_output_dir,
        config_file=config_file,
        collect_stats=True,
        seed=seed
    )
    stats_end = time.time()
    result["stats_collection_success"] = (stats_code == 0)
    result["stats_collection_time_seconds"] = stats_end - stats_start
    
    if stats_code != 0:
        print(f"Warning: Statistics collection failed for {condition_name} (exit code: {stats_code})")
        # 統計情報収集に失敗してもFine-tuningは続行する（ESPnet2が自動で統計情報を収集する可能性があるため）
    
    # Step 2: Fine-tuning
    ft_output_dir = output_dir / f"{condition_dir_name}"
    ft_start = time.time()
    ft_code = run_espnet2_tts_train(
        condition_name=f"{condition_name} (Fine-tuning)",
        train_data_dir=train_data_dir,
        valid_data_dir=valid_data_dir,
        output_dir=ft_output_dir,
        config_file=config_file,
        pretrained_model=pretrained_model,
        collect_stats=False,
        seed=seed
    )
    ft_end = time.time()
    result["fine_tuning_success"] = (ft_code == 0)
    result["fine_tuning_time_seconds"] = ft_end - ft_start
    
    # 学習時間の記録
    end_time = datetime.now()
    result["end_time"] = end_time.isoformat()
    result["total_time_seconds"] = (end_time - start_time).total_seconds()
    
    # ログディレクトリに結果を保存
    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"training_log_{condition_dir_name}_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nTraining log saved to: {log_file}")
    
    # 結果の表示
    print("\n" + "=" * 60)
    print(f"Training completed for {condition_name}")
    print("=" * 60)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {result['total_time_seconds']:.2f} seconds ({result['total_time_seconds']/60:.2f} minutes)")
    print(f"Stats collection: {'Success' if result['stats_collection_success'] else 'Failed'}")
    print(f"Fine-tuning: {'Success' if result['fine_tuning_success'] else 'Failed'}")
    print("=" * 60 + "\n")
    
    return result


def train_all_conditions(
    project_root: Path,
    pretrained_model: Optional[str] = None,
    config_file: Optional[Path] = None,
    logs_dir: Optional[Path] = None,
    seed: int = 42
):
    """
    全条件をループ処理してFine-tuningを実行する
    
    Args:
        project_root: プロジェクトルートディレクトリ
        pretrained_model: 事前学習モデルのパス
        config_file: 設定ファイル（YAML）のパス
        logs_dir: ログ保存ディレクトリ
        seed: 乱数シード
    """
    # デフォルトパスの設定
    if config_file is None:
        config_file = project_root / "configs" / "finetune_tacotron2.yaml"
    if logs_dir is None:
        logs_dir = project_root / "logs"
    if pretrained_model is None:
        # デフォルトの事前学習モデルパス（ESPnet model zooのダウンロード後の実際のパス構造）
        # 実際のパス: downloads/{hash}/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train.loss.ave_5best.pth
        default_model_path = project_root / "downloads" / "0afe7c220cac7d9893eea4ff1e4ca64e" / "exp" / "tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause" / "train.loss.ave_5best.pth"
        if default_model_path.exists():
            pretrained_model = str(default_model_path)
        else:
            # パスが存在しない場合、動的に検索
            downloads_dir = project_root / "downloads"
            model_file = None
            if downloads_dir.exists():
                for hash_dir in downloads_dir.iterdir():
                    if hash_dir.is_dir():
                        candidate = hash_dir / "exp" / "tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause" / "train.loss.ave_5best.pth"
                        if candidate.exists():
                            model_file = candidate
                            break
            if model_file:
                pretrained_model = str(model_file)
            else:
                print("Warning: Pretrained model not found. Please specify --pretrained_model explicitly.")
    
    # データディレクトリと条件名のマッピング
    conditions = [
        ("train_80sent", "E1: 80文コーパス"),
        ("train_4sent_37phonemes", "E2: 37音素4文"),
        ("train_4sent_random", "E3: 低カバレッジ4文"),
        ("train_10sent_top", "E4: 上位10文"),
    ]
    
    # 出力ディレクトリ
    output_dir = project_root / "exp"
    
    # 検証データディレクトリ（テストセット）
    valid_data_dir = project_root / "data" / "test"
    
    # 全条件をループ処理
    all_results = []
    for condition_dir_name, condition_name in conditions:
        train_data_dir = project_root / "data" / condition_dir_name
        
        # データディレクトリの存在確認
        if not train_data_dir.exists():
            print(f"Warning: Training data directory not found: {train_data_dir}")
            continue
        
        # 各条件の学習を実行
        result = train_condition(
            condition_name=condition_name,
            condition_dir_name=condition_dir_name,
            train_data_dir=train_data_dir,
            valid_data_dir=valid_data_dir,
            output_dir=output_dir,
            config_file=config_file,
            pretrained_model=pretrained_model,
            logs_dir=logs_dir,
            seed=seed
        )
        all_results.append(result)
    
    # 全結果をログファイルに保存
    if logs_dir and all_results:
        summary_file = logs_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nTraining summary saved to: {summary_file}")


def train_espnet2(
    condition_name: str,
    data_list_path: Path,
    output_dir: Path,
    pretrained_model: Optional[str] = None,
    config_file: Optional[Path] = None,
    batch_size: int = 4,
    max_steps: int = 5000,
    learning_rate: float = 1e-4,
    seed: int = 42
):
    """
    ESPnet2を使用してfine-tuningを実行する（旧インターフェース、後方互換性のため保持）
    
    注: この関数は後方互換性のため保持されていますが、
    新しい実装では train_all_conditions() または train_condition() を使用してください。
    """
    print("=" * 60)
    print(f"Fine-tuning: {condition_name}")
    print("=" * 60)
    print("\nNote: This function is kept for backward compatibility.")
    print("Please use train_all_conditions() or train_condition() for new implementations.")
    print("=" * 60)


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuning script for TTS")
    parser.add_argument(
        "--all_conditions",
        action="store_true",
        help="Train all conditions in a loop"
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        choices=["train_80sent", "train_4sent_37phonemes", "train_4sent_random", "train_10sent_top"],
        help="Single condition to train (only used if --all_conditions is not set)"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default=None,
        help="Directory to save training logs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # パス設定
    project_root = Path(__file__).parent.parent
    
    if args.all_conditions:
        # 全条件をループ処理
        train_all_conditions(
            project_root=project_root,
            pretrained_model=args.pretrained_model,
            config_file=Path(args.config) if args.config else None,
            logs_dir=Path(args.logs_dir) if args.logs_dir else None,
            seed=args.seed
        )
    else:
        # 単一条件の学習（旧インターフェース）
        if args.condition is None:
            parser.error("--condition is required when --all_conditions is not set")
        
        # 条件名のマッピング
        condition_map = {
            "train_80sent": "E1: 80文コーパス",
            "train_4sent_37phonemes": "E2: 37音素4文",
            "train_4sent_random": "E3: 低カバレッジ4文",
            "train_10sent_top": "E4: 上位10文",
        }
        
        condition_name = condition_map.get(args.condition, args.condition)
        train_data_dir = project_root / "data" / args.condition
        valid_data_dir = project_root / "data" / "test"
        output_dir = project_root / "exp"
        config_file = Path(args.config) if args.config else (project_root / "configs" / "finetune_tacotron2.yaml")
        logs_dir = Path(args.logs_dir) if args.logs_dir else (project_root / "logs")
        
        train_condition(
            condition_name=condition_name,
            condition_dir_name=args.condition,
            train_data_dir=train_data_dir,
            valid_data_dir=valid_data_dir,
            output_dir=output_dir,
            config_file=config_file,
            pretrained_model=args.pretrained_model,
            logs_dir=logs_dir,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
