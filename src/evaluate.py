#!/usr/bin/env python3
"""
単一条件の評価スクリプト
MCD（Mel-Cepstral Distortion）とlog-F0 RMSEを計算する
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pysptk
import pyworld as pw
import soundfile as sf
from fastdtw import fastdtw
from scipy import spatial


def _get_basename(path: str) -> str:
    """ファイルパスからベース名を取得"""
    return Path(path).stem


def _get_best_mcep_params(fs: int) -> Tuple[int, float]:
    """サンプリングレートに応じた最適なmcepパラメータを取得"""
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 25, 0.41
    else:
        return 25, 0.41


def extract_mcep(
    x: np.ndarray,
    fs: int,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
) -> np.ndarray:
    """SPTKベースのmel-cepstrumを抽出"""
    # パディング
    n_pad = n_fft - (len(x) - n_fft) % n_shift
    x = np.pad(x, (0, n_pad), "reflect")
    
    # フレーム数
    n_frame = (len(x) - n_fft) // n_shift + 1
    
    # ウィンドウ関数
    win = pysptk.sptk.hamming(n_fft)
    
    # mcepパラメータの確認
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    
    # mel-cepstrum計算
    mcep = [
        pysptk.mcep(
            x[n_shift * i : n_shift * i + n_fft] * win,
            mcep_dim,
            mcep_alpha,
            eps=1e-6,
            etype=1,
        )
        for i in range(n_frame)
    ]
    
    return np.stack(mcep)


def extract_f0(
    x: np.ndarray,
    fs: int,
    f0min: int = 40,
    f0max: int = 800,
) -> np.ndarray:
    """WorldベースのF0を抽出"""
    x = x.astype(np.float64)
    f0, time_axis = pw.harvest(
        x,
        fs,
        f0_floor=f0min,
        f0_ceil=f0max,
        frame_period=256 / fs * 1000,  # n_shift=256を想定
    )
    return f0


def calculate_mcd(
    ref_audio_path: Path,
    synth_audio_path: Path,
    fs: int = 24000,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
) -> float:
    """
    MCD（Mel-Cepstral Distortion）を計算
    
    Args:
        ref_audio_path: 参照音声ファイルパス
        synth_audio_path: 合成音声ファイルパス
        fs: サンプリングレート
        n_fft: FFT長
        n_shift: シフト長
        mcep_dim: mel-cepstrum次元数
        mcep_alpha: all-pass filter係数
        
    Returns:
        MCD値（dB）
    """
    # 音声読み込み
    ref_audio, _ = librosa.load(str(ref_audio_path), sr=fs)
    synth_audio, _ = librosa.load(str(synth_audio_path), sr=fs)
    
    # mcepパラメータの確認
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    
    # mel-cepstrum抽出
    ref_mcep = extract_mcep(ref_audio, fs, n_fft, n_shift, mcep_dim, mcep_alpha)
    synth_mcep = extract_mcep(synth_audio, fs, n_fft, n_shift, mcep_dim, mcep_alpha)
    
    # DTWでアライメント
    distance, path = fastdtw(
        ref_mcep[:, 1:],  # c0を除く
        synth_mcep[:, 1:],
        dist=spatial.distance.euclidean,
    )
    
    # MCD計算
    mcd_sum = 0.0
    for ref_idx, synth_idx in path:
        diff = ref_mcep[ref_idx, 1:] - synth_mcep[synth_idx, 1:]
        mcd_sum += np.sqrt(np.sum(diff ** 2))
    
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0 * mcd_sum / len(path))
    
    return mcd


def calculate_log_f0_rmse(
    ref_audio_path: Path,
    synth_audio_path: Path,
    fs: int = 24000,
    f0min: int = 40,
    f0max: int = 800,
    n_fft: int = 512,
    n_shift: int = 256,
) -> float:
    """
    log-F0 RMSEを計算（DTWアライメント使用）
    
    Args:
        ref_audio_path: 参照音声ファイルパス
        synth_audio_path: 合成音声ファイルパス
        fs: サンプリングレート
        f0min: 最小F0値
        f0max: 最大F0値
        n_fft: FFT長（DTWアライメント用）
        n_shift: シフト長（DTWアライメント用）
        
    Returns:
        log-F0 RMSE値
    """
    # 音声読み込み
    ref_audio, _ = librosa.load(str(ref_audio_path), sr=fs)
    synth_audio, _ = librosa.load(str(synth_audio_path), sr=fs)
    
    # F0抽出
    ref_f0 = extract_f0(ref_audio, fs, f0min, f0max)
    synth_f0 = extract_f0(synth_audio, fs, f0min, f0max)
    
    # 有声音フレームのみを抽出（DTWアライメント前にフィルタリング）
    ref_voiced = ref_f0 > 0
    synth_voiced = synth_f0 > 0
    
    # 有声音フレームが少なすぎる場合はinfを返す
    if np.sum(ref_voiced) < 10 or np.sum(synth_voiced) < 10:
        return float('inf')
    
    # 有声音フレームのF0のみを使用
    ref_f0_voiced = ref_f0[ref_voiced]
    synth_f0_voiced = synth_f0[synth_voiced]
    
    # 時間軸のアライメントのために、mel-cepstrumを使用してDTWを計算
    # （F0の長さが異なる可能性があるため）
    ref_mcep = extract_mcep(ref_audio, fs, n_fft, n_shift, mcep_dim=25, mcep_alpha=0.41)
    synth_mcep = extract_mcep(synth_audio, fs, n_fft, n_shift, mcep_dim=25, mcep_alpha=0.41)
    
    # DTWでアライメント
    try:
        distance, path = fastdtw(
            ref_mcep[:, 1:],  # c0を除く
            synth_mcep[:, 1:],
            dist=spatial.distance.euclidean,
        )
    except Exception as e:
        logging.warning(f"DTW failed for {ref_audio_path.name}: {e}")
        return float('inf')
    
    # F0フレームとmel-cepstrumフレームの対応関係を計算
    # F0のフレーム数とmel-cepstrumのフレーム数が異なる可能性があるため、
    # フレームインデックスを正規化
    ref_f0_frames = len(ref_f0)
    synth_f0_frames = len(synth_f0)
    ref_mcep_frames = len(ref_mcep)
    synth_mcep_frames = len(synth_mcep)
    
    # DTWパスに基づいてF0をアライメント
    aligned_ref_f0 = []
    aligned_synth_f0 = []
    
    for ref_mcep_idx, synth_mcep_idx in path:
        # mel-cepstrumフレームインデックスをF0フレームインデックスに変換
        ref_f0_idx = int(ref_mcep_idx * ref_f0_frames / ref_mcep_frames)
        synth_f0_idx = int(synth_mcep_idx * synth_f0_frames / synth_mcep_frames)
        
        # 範囲チェック
        if ref_f0_idx >= ref_f0_frames:
            ref_f0_idx = ref_f0_frames - 1
        if synth_f0_idx >= synth_f0_frames:
            synth_f0_idx = synth_f0_frames - 1
        
        # 有声音フレームのみを使用
        if ref_f0[ref_f0_idx] > 0 and synth_f0[synth_f0_idx] > 0:
            aligned_ref_f0.append(ref_f0[ref_f0_idx])
            aligned_synth_f0.append(synth_f0[synth_f0_idx])
    
    # アライメント後の有声音フレームが少なすぎる場合はinfを返す
    if len(aligned_ref_f0) < 10:
        return float('inf')
    
    # log-F0計算
    aligned_ref_f0 = np.array(aligned_ref_f0)
    aligned_synth_f0 = np.array(aligned_synth_f0)
    
    ref_log_f0 = np.log(aligned_ref_f0 + 1e-8)
    synth_log_f0 = np.log(aligned_synth_f0 + 1e-8)
    
    # RMSE計算
    rmse = np.sqrt(np.mean((ref_log_f0 - synth_log_f0) ** 2))
    
    return rmse


def evaluate_condition(
    synth_audio_dir: Path,
    reference_audio_dir: Path,
    output_json: Path,
    fs: int = 24000,
) -> Dict[str, float]:
    """
    単一条件の評価を実行
    
    Args:
        synth_audio_dir: 合成音声ディレクトリ
        reference_audio_dir: 参照音声ディレクトリ
        output_json: 出力JSONファイルパス
        fs: サンプリングレート
        
    Returns:
        評価結果の辞書（{"mcd": float, "log_f0_rmse": float}）
    """
    # 音声ファイルのリストを取得
    synth_files = sorted(synth_audio_dir.glob("*.wav"))
    ref_files = sorted(reference_audio_dir.glob("*.wav"))
    
    if len(synth_files) == 0:
        raise ValueError(f"No synthetic audio files found in {synth_audio_dir}")
    if len(ref_files) == 0:
        raise ValueError(f"No reference audio files found in {reference_audio_dir}")
    
    # ファイル名でマッチング
    synth_dict = {_get_basename(f): f for f in synth_files}
    ref_dict = {_get_basename(f): f for f in ref_files}
    
    # 共通のファイル名を取得
    common_basenames = set(synth_dict.keys()) & set(ref_dict.keys())
    
    if len(common_basenames) == 0:
        raise ValueError("No matching audio files found between synthetic and reference directories")
    
    # 各ファイルの評価
    mcd_values = []
    log_f0_rmse_values = []
    file_results = {}
    
    for basename in sorted(common_basenames):
        synth_path = synth_dict[basename]
        ref_path = ref_dict[basename]
        
        try:
            # MCD計算
            mcd = calculate_mcd(ref_path, synth_path, fs=fs)
            mcd_values.append(mcd)
            
            # log-F0 RMSE計算
            log_f0_rmse = calculate_log_f0_rmse(ref_path, synth_path, fs=fs)
            if not np.isinf(log_f0_rmse):
                log_f0_rmse_values.append(log_f0_rmse)
            
            file_results[basename] = {
                "mcd": float(mcd),
                "log_f0_rmse": float(log_f0_rmse) if not np.isinf(log_f0_rmse) else None,
            }
        except Exception as e:
            logging.warning(f"Failed to evaluate {basename}: {e}")
            continue
    
    # 平均値計算
    avg_mcd = np.mean(mcd_values) if mcd_values else float('inf')
    avg_log_f0_rmse = np.mean(log_f0_rmse_values) if log_f0_rmse_values else float('inf')
    
    # 結果をまとめる
    results = {
        "mcd": float(avg_mcd),
        "log_f0_rmse": float(avg_log_f0_rmse),
        "num_files": len(common_basenames),
        "file_results": file_results,
    }
    
    # JSONファイルに保存
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="Evaluate single condition")
    parser.add_argument(
        "--synth_audio_dir",
        type=str,
        required=True,
        help="Directory containing synthetic audio files"
    )
    parser.add_argument(
        "--reference_audio_dir",
        type=str,
        required=True,
        help="Directory containing reference audio files"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=24000,
        help="Sampling rate (default: 24000)"
    )
    
    args = parser.parse_args()
    
    # 評価実行
    results = evaluate_condition(
        synth_audio_dir=Path(args.synth_audio_dir),
        reference_audio_dir=Path(args.reference_audio_dir),
        output_json=Path(args.output_json),
        fs=args.fs,
    )
    
    # 結果表示
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"MCD: {results['mcd']:.4f} dB")
    print(f"log-F0 RMSE: {results['log_f0_rmse']:.4f}")
    print(f"Number of files: {results['num_files']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
