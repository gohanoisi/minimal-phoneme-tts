#!/usr/bin/env python3
"""
Fine-tuningスクリプト
各条件のコーパスで事前学習済み日本語TTSモデルをfine-tuningする。
"""

import json
import os
import random
import numpy as np
from pathlib import Path
from typing import Dict, Optional
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
    ESPnet2を使用してfine-tuningを実行する
    
    Args:
        condition_name: 条件名（例: "E1_80sent"）
        data_list_path: データリストファイルのパス
        output_dir: 出力ディレクトリ
        pretrained_model: 事前学習モデルのパス
        config_file: 設定ファイルのパス
        batch_size: バッチサイズ
        max_steps: 最大ステップ数
        learning_rate: 学習率
        seed: 乱数シード
    """
    print("=" * 60)
    print(f"Fine-tuning: {condition_name}")
    print("=" * 60)
    
    # 乱数シード固定
    set_seed(seed)
    
    # GPU確認
    use_gpu = check_gpu()
    
    # データリストの読み込み
    print(f"Loading data list from: {data_list_path}")
    data_list = load_data_list(data_list_path)
    print(f"Loaded {len(data_list)} samples")
    
    # 出力ディレクトリの作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ESPnet2のコマンドを構築
    # 注: 実際のESPnet2の学習コマンドは、espnet2のレシピに従う必要があります
    # ここでは基本的な構造のみを示します
    
    print("\nNote: This is a placeholder script.")
    print("Actual ESPnet2 training requires:")
    print("1. ESPnet2 installation")
    print("2. Configuration YAML file")
    print("3. Pretrained model download")
    print("4. Running espnet2/bin/train.py with proper arguments")
    
    # TODO: ESPnet2の実際の学習コマンドを実装
    # 例:
    # cmd = [
    #     "python", "-m", "espnet2.bin.train",
    #     "--config", str(config_file),
    #     "--train_data_path_and_name_and_type", f"{data_list_path},data,json",
    #     "--output_dir", str(output_dir),
    #     ...
    # ]
    
    print(f"\nOutput directory: {output_dir}")
    print("Training configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max steps: {max_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Seed: {seed}")


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuning script for TTS")
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["E1_80sent", "E2_4sent_37phonemes", "E3_4sent_random", "E4_10sent_top"],
        help="Experiment condition"
    )
    parser.add_argument(
        "--data_list",
        type=str,
        required=True,
        help="Path to data list file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints"
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
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
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
    data_list_path = Path(args.data_list)
    output_dir = Path(args.output_dir)
    
    # Fine-tuning実行
    train_espnet2(
        condition_name=args.condition,
        data_list_path=data_list_path,
        output_dir=output_dir,
        pretrained_model=args.pretrained_model,
        config_file=Path(args.config) if args.config else None,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
