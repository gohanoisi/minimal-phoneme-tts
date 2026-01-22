#!/usr/bin/env python3
"""
音声合成スクリプト
各条件でfine-tuningしたモデルを使用し、テストセットから音声を合成する。
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse

import numpy as np
import soundfile as sf
import torch

# ESPnet2のインポート
try:
    from espnet2.bin.tts_inference import Text2Speech
except ImportError:
    print("Error: ESPnet2 is not installed. Please install it first.")
    sys.exit(1)


def load_test_texts(test_data_list_file: Path) -> Dict[str, str]:
    """
    テストセットのデータリストファイルを読み込む（JSONL形式）
    
    Args:
        test_data_list_file: テストセットのdata.listファイルのパス
        
    Returns:
        文ID -> テキストの辞書
    """
    texts = {}
    
    if not test_data_list_file.exists():
        raise FileNotFoundError(f"Test data list file not found: {test_data_list_file}")
    
    with open(test_data_list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # JSONL形式: {"utt_id": "...", "text": "...", ...}
            try:
                data = json.loads(line)
                utt_id = data.get("utt_id")
                text = data.get("text")
                if utt_id and text:
                    texts[utt_id] = text
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {line[:50]}... Error: {e}")
                continue
    
    return texts


def synthesize_condition(
    condition_name: str,
    condition_dir_name: str,
    checkpoint_dir: Path,
    test_data_list_file: Path,
    output_dir: Path,
    logs_dir: Path,
    project_root: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    vocoder_tag: Optional[str] = None
) -> Dict:
    """
    1つの条件に対して音声合成を実行する
    
    Args:
        condition_name: 条件名（表示用）
        condition_dir_name: 条件ディレクトリ名（例: "train_80sent"）
        checkpoint_dir: チェックポイントディレクトリ（exp/<condition>/）
        test_data_list_file: テストセットのdata.listファイルのパス
        output_dir: 出力ディレクトリ（outputs/audio/）
        logs_dir: ログ保存ディレクトリ
        device: デバイス（cuda/cpu）
        seed: 乱数シード
        vocoder_tag: ボコーダータグ（例: "jsut.parallel_wavegan.v1"）
        
    Returns:
        合成結果の辞書（開始時刻、終了時刻、総合成時間、成功フラグなど）
    """
    print("=" * 60)
    print(f"Synthesizing condition: {condition_name}")
    print("=" * 60)
    
    # 開始時刻
    start_time = datetime.now()
    result = {
        "condition_name": condition_name,
        "condition_dir_name": condition_dir_name,
        "start_time": start_time.isoformat(),
        "end_time": None,
        "total_time_seconds": None,
        "num_sentences": 0,
        "num_success": 0,
        "num_failed": 0,
        "synthesis_times": [],
        "errors": []
    }
    
    # チェックポイントと設定ファイルのパス
    # チェックポイントファイルの優先順位: 10epoch.pth > latest.pth > checkpoint.pth
    if (checkpoint_dir / "10epoch.pth").exists():
        checkpoint_file = checkpoint_dir / "10epoch.pth"
        print(f"Using 10epoch.pth checkpoint")
    elif (checkpoint_dir / "latest.pth").exists():
        checkpoint_file = checkpoint_dir / "latest.pth"
        print(f"Using latest.pth checkpoint")
    else:
        checkpoint_file = checkpoint_dir / "checkpoint.pth"
        print(f"Using checkpoint.pth (fallback)")
    
    config_file = checkpoint_dir / "config.yaml"
    
    if not checkpoint_file.exists():
        error_msg = f"Checkpoint file not found: {checkpoint_file}"
        print(f"Error: {error_msg}")
        result["errors"].append(error_msg)
        result["end_time"] = datetime.now().isoformat()
        return result
    
    if not config_file.exists():
        error_msg = f"Config file not found: {config_file}"
        print(f"Error: {error_msg}")
        result["errors"].append(error_msg)
        result["end_time"] = datetime.now().isoformat()
        return result
    
    # テストセットのテキストを読み込む
    print(f"Loading test texts from: {test_data_list_file}")
    test_texts = load_test_texts(test_data_list_file)
    result["num_sentences"] = len(test_texts)
    print(f"Number of test sentences: {len(test_texts)}")
    
    # Text2Speechオブジェクトの初期化
    print(f"Loading model from: {checkpoint_file}")
    print(f"Using device: {device}")
    
    # ボコーダーの設定（JSUT用のParallel WaveGANボコーダー）
    vocoder_config = None
    vocoder_file = None
    if vocoder_tag:
        print(f"Loading vocoder: {vocoder_tag}")
        try:
            # parallel_waveganパッケージを使用してボコーダーをダウンロード
            from parallel_wavegan.utils import download_pretrained_model
            from pathlib import Path as PPath
            import gdown
            
            # vocoder_tagからparallel_wavegan/プレフィックスを削除
            vocoder_model_tag = vocoder_tag.replace("parallel_wavegan/", "") if vocoder_tag.startswith("parallel_wavegan/") else vocoder_tag
            
            # JSUT用のParallel WaveGANボコーダーのGoogle Drive URL
            vocoder_urls = {
                "jsut.parallel_wavegan.v1": "https://drive.google.com/open?id=1OwrUQzAmvjj1x9cDhnZPp6dqtsEqGEJM"
            }
            
            if vocoder_model_tag in vocoder_urls:
                # Google Driveから直接ダウンロード
                url = vocoder_urls[vocoder_model_tag]
                print(f"Downloading vocoder from Google Drive: {url}")
                # ダウンロード先ディレクトリ
                # project_rootを取得（関数の引数から）
                vocoder_dir = project_root / "downloads" / "vocoders" / vocoder_model_tag
                vocoder_dir.mkdir(parents=True, exist_ok=True)
                
                # ファイルIDを抽出
                file_id = url.split("id=")[-1] if "id=" in url else None
                if file_id:
                    output_file = vocoder_dir / f"{vocoder_model_tag}.tar.gz"
                    if not output_file.exists():
                        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(output_file), quiet=False)
                    
                    # tar.gzを展開
                    import tarfile
                    if output_file.exists():
                        with tarfile.open(output_file, "r:gz") as tar:
                            tar.extractall(vocoder_dir)
                        
                        # 展開されたファイルを探す
                        pkl_files = list(vocoder_dir.glob("**/*.pkl"))
                        config_files = list(vocoder_dir.glob("**/config.yml"))
                        
                        if pkl_files and config_files:
                            vocoder_file = str(pkl_files[0])
                            vocoder_config = str(config_files[0])
                            print(f"Vocoder downloaded: {vocoder_file}")
                        else:
                            raise FileNotFoundError(f"Vocoder files not found in {vocoder_dir}")
                    else:
                        raise FileNotFoundError(f"Failed to download vocoder: {output_file}")
                else:
                    raise ValueError(f"Invalid Google Drive URL: {url}")
            else:
                # 通常のダウンロード方法を試す
                vocoder_file = download_pretrained_model(vocoder_model_tag)
                vocoder_config = str(PPath(vocoder_file).parent / "config.yml")
                print(f"Vocoder downloaded: {vocoder_file}")
        except ImportError as e:
            print(f"Warning: Required package not installed: {e}")
            print("To use Parallel WaveGAN vocoder, install: pip install -U parallel_wavegan gdown")
            print("Using default Griffin-Lim vocoder.")
            vocoder_config = None
            vocoder_file = None
        except Exception as e:
            print(f"Warning: Failed to download vocoder: {e}")
            print("Using default Griffin-Lim vocoder.")
            vocoder_config = None
            vocoder_file = None
    
    try:
        text2speech = Text2Speech(
            train_config=str(config_file),
            model_file=str(checkpoint_file),
            vocoder_config=str(vocoder_config) if vocoder_config else None,
            vocoder_file=str(vocoder_file) if vocoder_file else None,
            device=device,
            seed=seed,
            always_fix_seed=True,
            prefer_normalized_feats=False  # デノーマライズされた特徴量を使用
        )
        print("Model loaded successfully")
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        print(f"Error: {error_msg}")
        result["errors"].append(error_msg)
        result["end_time"] = datetime.now().isoformat()
        return result
    
    # 出力ディレクトリの作成
    condition_output_dir = output_dir / condition_dir_name
    condition_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {condition_output_dir}")
    
    # 各文を合成
    print("\nStarting synthesis...")
    for i, (utt_id, text) in enumerate(test_texts.items(), 1):
        print(f"[{i}/{len(test_texts)}] Synthesizing: {utt_id}")
        
        try:
            # 音声合成の実行（テキストを渡す、ESPnet2が内部で音素変換を行う）
            synth_start = time.time()
            output_dict = text2speech(text)
            synth_end = time.time()
            
            # デバッグ: メルスペクトログラムの値を確認
            try:
                feat_gen = output_dict.get("feat_gen")
                feat_gen_denorm = output_dict.get("feat_gen_denorm")
                
                if feat_gen is not None:
                    if isinstance(feat_gen, torch.Tensor):
                        feat_gen_np = feat_gen.detach().cpu().numpy()
                    else:
                        feat_gen_np = np.array(feat_gen)
                    
                    print(f"  [Debug] Normalized mel spectrogram: shape={feat_gen_np.shape}, min={feat_gen_np.min():.4f}, max={feat_gen_np.max():.4f}, mean={feat_gen_np.mean():.4f}, std={feat_gen_np.std():.4f}")
                
                if feat_gen_denorm is not None:
                    if isinstance(feat_gen_denorm, torch.Tensor):
                        feat_gen_denorm_np = feat_gen_denorm.detach().cpu().numpy()
                    else:
                        feat_gen_denorm_np = np.array(feat_gen_denorm)
                    
                    print(f"  [Debug] Denormalized mel spectrogram: shape={feat_gen_denorm_np.shape}, min={feat_gen_denorm_np.min():.4f}, max={feat_gen_denorm_np.max():.4f}, mean={feat_gen_denorm_np.mean():.4f}, std={feat_gen_denorm_np.std():.4f}")
                else:
                    print(f"  [Warning] feat_gen_denorm is None! This may be the problem.")
            except Exception as e:
                print(f"  [Debug] Failed to check mel spectrogram: {e}")
            
            # 音声データの取得
            if "wav" in output_dict:
                wav = output_dict["wav"]
                # numpy配列に変換
                if isinstance(wav, torch.Tensor):
                    wav = wav.cpu().numpy()
                
                # サンプリングレートの取得
                fs = text2speech.fs
                
                # 音声ファイルの保存
                output_wav_file = condition_output_dir / f"{utt_id}.wav"
                sf.write(str(output_wav_file), wav, fs, "PCM_16")
                
                synthesis_time = synth_end - synth_start
                result["synthesis_times"].append(synthesis_time)
                result["num_success"] += 1
                
                print(f"  ✓ Saved: {output_wav_file} ({synthesis_time:.2f}s)")
            else:
                error_msg = f"No wav output for {utt_id}"
                print(f"  ✗ {error_msg}")
                result["errors"].append(error_msg)
                result["num_failed"] += 1
                
        except Exception as e:
            error_msg = f"Failed to synthesize {utt_id}: {e}"
            print(f"  ✗ {error_msg}")
            result["errors"].append(error_msg)
            result["num_failed"] += 1
    
    # 終了時刻
    end_time = datetime.now()
    result["end_time"] = end_time.isoformat()
    result["total_time_seconds"] = (end_time - start_time).total_seconds()
    
    # 結果の表示
    print("\n" + "=" * 60)
    print(f"Synthesis completed for {condition_name}")
    print("=" * 60)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {result['total_time_seconds']:.2f} seconds ({result['total_time_seconds']/60:.2f} minutes)")
    print(f"Total sentences: {result['num_sentences']}")
    print(f"Success: {result['num_success']}")
    print(f"Failed: {result['num_failed']}")
    if result["synthesis_times"]:
        avg_time = sum(result["synthesis_times"]) / len(result["synthesis_times"])
        print(f"Average synthesis time per sentence: {avg_time:.2f} seconds")
    print("=" * 60 + "\n")
    
    # ログファイルに結果を保存
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"synthesis_{condition_dir_name}_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Synthesis log saved to: {log_file}")
    
    return result


def get_default_pretrained_paths(project_root: Path) -> tuple[Path, Path]:
    """事前学習モデル（JSUT Tacotron2）のデフォルトパスを返す。"""
    base = (
        project_root
        / "downloads"
        / "0afe7c220cac7d9893eea4ff1e4ca64e"
        / "exp"
        / "tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause"
    )
    return base / "train.loss.ave_5best.pth", base / "config.yaml"


def load_pretrained_text2speech(
    project_root: Path,
    vocoder_tag: Optional[str],
    device: str,
    seed: int,
) -> Text2Speech:
    """事前学習モデル + ボコーダーでText2Speechを構築する。"""
    pretrained_model, pretrained_config = get_default_pretrained_paths(project_root)
    if not pretrained_model.exists() or not pretrained_config.exists():
        raise FileNotFoundError(
            f"Pretrained model/config not found: {pretrained_model}, {pretrained_config}"
        )

    # 既存のvocoderダウンロードロジック（synthesize_condition内）を再利用したいが、
    # ここでは既にDL済みのパスを前提にする（無ければsynthesize_condition側と同様にDLされる）。
    # 安全のため、synthesize_conditionのvocoderダウンロードをそのまま使いたい場合は後で共通化する。
    vocoder_config = None
    vocoder_file = None
    if vocoder_tag:
        # synthesize_conditionと同じ扱い: jsut.parallel_wavegan.v1 の展開先を探す
        vocoder_model_tag = (
            vocoder_tag.replace("parallel_wavegan/", "")
            if vocoder_tag.startswith("parallel_wavegan/")
            else vocoder_tag
        )
        vocoder_dir = project_root / "downloads" / "vocoders" / vocoder_model_tag / vocoder_model_tag
        cfg = vocoder_dir / "config.yml"
        pkl = list(vocoder_dir.glob("*.pkl"))
        if cfg.exists() and pkl:
            vocoder_config = cfg
            vocoder_file = pkl[0]

    return Text2Speech(
        train_config=str(pretrained_config),
        model_file=str(pretrained_model),
        vocoder_config=str(vocoder_config) if vocoder_config else None,
        vocoder_file=str(vocoder_file) if vocoder_file else None,
        device=device,
        seed=seed,
        always_fix_seed=True,
        prefer_normalized_feats=False,
    )


def synthesize_pretrained(
    project_root: Path,
    test_data_list_file: Path,
    output_dir: Path,
    logs_dir: Path,
    device: str,
    seed: int,
    vocoder_tag: Optional[str],
    utt_id: Optional[str] = None,
) -> Dict:
    """事前学習モデルでテストセットを合成する（必要なら1文だけ）。"""
    print("=" * 60)
    print("Synthesizing baseline (pretrained) model")
    print("=" * 60)

    start_time = datetime.now()
    result = {
        "mode": "pretrained",
        "start_time": start_time.isoformat(),
        "end_time": None,
        "total_time_seconds": None,
        "num_sentences": 0,
        "num_success": 0,
        "num_failed": 0,
        "synthesis_times": [],
        "errors": [],
        "utt_id": utt_id,
    }

    test_texts = load_test_texts(test_data_list_file)
    if utt_id:
        if utt_id not in test_texts:
            raise KeyError(f"utt_id not found in test set: {utt_id}")
        test_texts = {utt_id: test_texts[utt_id]}

    result["num_sentences"] = len(test_texts)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    t2s = load_pretrained_text2speech(
        project_root=project_root, vocoder_tag=vocoder_tag, device=device, seed=seed
    )

    for i, (uid, text) in enumerate(test_texts.items(), 1):
        print(f"[{i}/{len(test_texts)}] Synthesizing (baseline): {uid}")
        try:
            t0 = time.time()
            out = t2s(text)
            t1 = time.time()
            if "wav" not in out:
                raise RuntimeError("No wav in output_dict")
            wav = out["wav"]
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            out_path = output_dir / f"{uid}.wav"
            sf.write(str(out_path), wav, t2s.fs, "PCM_16")
            result["num_success"] += 1
            result["synthesis_times"].append(t1 - t0)
            print(
                f"  ✓ Saved: {out_path} (amp std={wav.std():.4f}, min={wav.min():.4f}, max={wav.max():.4f})"
            )
        except Exception as e:
            msg = f"Failed to synthesize {uid}: {e}"
            print(f"  ✗ {msg}")
            result["num_failed"] += 1
            result["errors"].append(msg)

    end_time = datetime.now()
    result["end_time"] = end_time.isoformat()
    result["total_time_seconds"] = (end_time - start_time).total_seconds()
    log_file = logs_dir / f"synthesis_baseline_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Baseline synthesis log saved to: {log_file}")
    return result


def synthesize_all_conditions(
    project_root: Path,
    base_dir: Path = None,
    output_dir: Path = None,
    logs_dir: Optional[Path] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    vocoder_tag: Optional[str] = None
):
    """
    全条件をループ処理して音声合成を実行する
    
    Args:
        project_root: プロジェクトルートディレクトリ
        base_dir: チェックポイントのベースディレクトリ（デフォルト: project_root / "exp"）
        output_dir: 出力ディレクトリ（デフォルト: project_root / "outputs" / "audio"）
        logs_dir: ログ保存ディレクトリ
        device: デバイス（cuda/cpu）
        seed: 乱数シード
    """
    # デフォルトパスの設定
    if base_dir is None:
        base_dir = project_root / "exp"
    if output_dir is None:
        output_dir = project_root / "outputs" / "audio"
    if logs_dir is None:
        logs_dir = project_root / "logs"
    
    # テストセットのデータリストファイル
    test_data_list_file = project_root / "data" / "test" / "data.list"
    
    # 条件名のマッピング
    conditions = [
        ("train_80sent", "E1: 80文コーパス"),
        ("train_4sent_37phonemes", "E2: 37音素4文"),
        ("train_4sent_random", "E3: 低カバレッジ4文"),
        ("train_10sent_top", "E4: 上位10文"),
    ]
    
    # 全条件をループ処理
    all_results = []
    for condition_dir_name, condition_name in conditions:
        checkpoint_dir = base_dir / condition_dir_name
        
        # チェックポイントディレクトリの存在確認
        if not checkpoint_dir.exists():
            print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
            continue
        
        # 各条件の音声合成を実行
        result = synthesize_condition(
            condition_name=condition_name,
            condition_dir_name=condition_dir_name,
            checkpoint_dir=checkpoint_dir,
            test_data_list_file=test_data_list_file,
            output_dir=output_dir,
            logs_dir=logs_dir,
            project_root=project_root,
            device=device,
            seed=seed,
            vocoder_tag=vocoder_tag
        )
        all_results.append(result)
    
    # 全結果をログファイルに保存
    if logs_dir and all_results:
        summary_file = logs_dir / f"synthesis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSynthesis summary saved to: {summary_file}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="音声合成スクリプト")
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        choices=["train_80sent", "train_4sent_37phonemes", "train_4sent_random", "train_10sent_top"],
        help="Single condition to synthesize (if not specified, all conditions will be processed)"
    )
    parser.add_argument(
        "--all_conditions",
        action="store_true",
        help="Synthesize all conditions in a loop"
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        help="Use pretrained baseline model instead of fine-tuned checkpoints"
    )
    parser.add_argument(
        "--utt_id",
        type=str,
        default=None,
        help="If set, synthesize only this utt_id (baseline mode)"
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default=None,
        help="Directory to save synthesis logs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (cuda/cpu). Default: auto-detect"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--vocoder_tag",
        type=str,
        default="jsut.parallel_wavegan.v1",
        help="Vocoder tag (e.g., 'jsut.parallel_wavegan.v1'). Use 'parallel_wavegan/' prefix for Parallel WaveGAN vocoders."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base directory for checkpoints (default: exp)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for audio files (default: outputs/audio)"
    )
    
    args = parser.parse_args()
    
    # パス設定
    project_root = Path(__file__).parent.parent
    
    # デバイスの自動検出
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if args.use_pretrained:
        # ベースライン（事前学習）モデルで合成
        test_data_list_file = project_root / "data" / "test" / "data.list"
        out_dir = project_root / "outputs" / "audio" / "baseline"
        logs_dir = Path(args.logs_dir) if args.logs_dir else (project_root / "logs")
        synthesize_pretrained(
            project_root=project_root,
            test_data_list_file=test_data_list_file,
            output_dir=out_dir,
            logs_dir=logs_dir,
            device=device,
            seed=args.seed,
            vocoder_tag=args.vocoder_tag,
            utt_id=args.utt_id,
        )
        return

    # ベースディレクトリと出力ディレクトリの設定
    base_dir = Path(args.base_dir) if args.base_dir else (project_root / "exp")
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "outputs" / "audio")
    
    if args.all_conditions or args.condition is None:
        # 全条件をループ処理
        synthesize_all_conditions(
            project_root=project_root,
            base_dir=base_dir,
            output_dir=output_dir,
            logs_dir=Path(args.logs_dir) if args.logs_dir else None,
            device=device,
            seed=args.seed,
            vocoder_tag=args.vocoder_tag
        )
    else:
        # 単一条件の音声合成
        condition_map = {
            "train_80sent": "E1: 80文コーパス",
            "train_4sent_37phonemes": "E2: 37音素4文",
            "train_4sent_random": "E3: 低カバレッジ4文",
            "train_10sent_top": "E4: 上位10文",
        }
        
        condition_name = condition_map.get(args.condition, args.condition)
        checkpoint_dir = base_dir / args.condition
        test_data_list_file = project_root / "data" / "test" / "data.list"
        logs_dir = Path(args.logs_dir) if args.logs_dir else (project_root / "logs")
        
        synthesize_condition(
            condition_name=condition_name,
            condition_dir_name=args.condition,
            checkpoint_dir=checkpoint_dir,
            test_data_list_file=test_data_list_file,
            output_dir=output_dir,
            logs_dir=logs_dir,
            project_root=project_root,
            device=device,
            seed=args.seed,
            vocoder_tag=args.vocoder_tag
        )


if __name__ == "__main__":
    main()
