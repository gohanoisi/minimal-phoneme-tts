# 開発計画（development_plan）

## 現状（2026-01-22 更新）

### 完了した作業

1. **データ形式修正**
   - `src/convert_to_espnet2_format.py` を修正し、Kaldi形式の `text` を **raw text（日本語文）** で出力するように変更
   - 全条件の `data/*/text` を raw text 形式に再生成完了

2. **ベースライン合成確認**
   - 事前学習モデルでテストセット18文を合成（`outputs/audio/baseline/`）
   - 振幅統計: std≈0.05（正常範囲）を確認

3. **1epochテスト学習（全4条件）**
   - `exp_text/` 以下で各条件を1epoch学習
   - 各条件で `VOICEACTRESS100_010.wav` を合成して振幅確認
   - 結果:
     - `train_80sent`: std=0.0179
     - `train_4sent_37phonemes`: std=0.0370
     - `train_4sent_random`: std=0.0325
     - `train_10sent_top`: std=0.0314
   - 以前の「一定音」状態（std≈0.003）より大幅改善を確認

### 重要な発見（合成失敗の原因）

- **根本原因**: `src/convert_to_espnet2_format.py` が Kaldi形式の `text` を **音素列**で出力していた
- **不整合**: `configs/finetune_tacotron2.yaml` は `g2p: pyopenjtalk_accent_with_pause` を使って **raw text→音素化**する前提
- **結果**: fine-tuning後のモデルが「一定音」になりやすい（読み上げが成立しない）
- **解決**: Kaldi `text` を raw text に修正し、`exp_text/` で再学習→合成振幅の改善を確認

### 音声ファイルの違いについて

`outputs/audio/exp_text/` 内の2つのファイルの違い:

1. **`exp_text_1epoch.wav`** (2026-01-20 18:40生成)
   - 学習条件: `train_80sent`（80文）で1epoch学習
   - テキスト: 「スマートフォンから、フィーチャーフォンまで、マルチデバイスに対応。」
   - 振幅: std=0.0328, min=-0.1744, max=0.1708

2. **`train_4sent_37phonemes/VOICEACTRESS100_010.wav`** (2026-01-20 20:08生成)
   - 学習条件: `train_4sent_37phonemes`（4文、37音素）で1epoch学習
   - テキスト: 同じ「スマートフォンから、フィーチャーフォンまで、マルチデバイスに対応。」
   - 振幅: std=0.0370, min=-0.2376, max=0.1642

**結論**: 両方とも1epochで学習したものですが、**学習データが異なります**（80文 vs 4文）。同じテキストでも、学習データの違いにより音声特性が異なります。

### 現在の課題

- **10epoch学習が完了していない**
  - `scripts/train_all_10epoch.sh` を実行したが、1epoch終了後にWindows/WSLのシンボリックリンク禁止による `PermissionError` で停止
  - 実際には各条件で1epochぶんのみ学習が完了
  - **解決**: `src/train.py`にsymlinkエラー修正（WSL対応のfallback実装）を追加済み

### 完了した作業（2026-01-20 夜間）

1. **symlinkエラー修正**
   - `src/train.py`にWSL環境対応のfallback実装を追加
   - ESPnet2の実行後に`latest.pth`のsymlink作成を試み、失敗時はコピーで代替

2. **Phase 6評価スクリプト実装**
   - `src/evaluate.py`: 単一条件の評価（MCD、log-F0 RMSE計算）
   - `src/evaluate_all_conditions.py`: 4条件一括評価とCSV出力
   - ESPnet2の評価ツールを参考に実装（pysptk、pyworld使用）

### 完了した作業（2026-01-21〜2026-01-22）

1. **10epoch学習の完了**
   - 全4条件で10epoch学習完了（2026-01-21 02:03-03:46）
   - symlinkエラー問題の解決（ESPnet2のtrainer.py修正）
   - 学習スクリプトに--resumeオプション自動追加機能を実装

2. **音声合成失敗問題の解決**
   - 原因: synthesize.pyがcheckpoint.pthを読み込んでいた（10epoch.pthが正しい）
   - 修正: チェックポイントファイルの優先順位を変更（10epoch.pth > latest.pth > checkpoint.pth）
   - 全72ファイル音声合成完了（2026-01-21 07:58-2026-01-22 02:13）

3. **重要な修正内容**
   - `espnet/espnet2/train/trainer.py`: symlinkエラー時にコピーで代替する処理を追加
   - `src/synthesize.py`: チェックポイントファイルの優先順位を修正
   - `scripts/train_all_10epoch.sh`: --resumeオプション自動追加、音声合成自動実行を追加

### 完了した作業（2026-01-22 更新）

1. **Phase 6（客観評価）の実行完了**
   - `python src/evaluate_all_conditions.py` を実行完了
   - 評価結果を`results/evaluation_results.json`と`results/evaluation_summary.csv`に保存完了
   - MCD、log-F0 RMSEの計算と比較完了
   - 評価結果:
     - train_80sent: MCD=5.0490 dB, log-F0 RMSE=0.2573
     - train_4sent_37phonemes: MCD=5.2273 dB, log-F0 RMSE=0.3364
     - train_4sent_random: MCD=5.2660 dB, log-F0 RMSE=0.3296
     - train_10sent_top: MCD=5.1021 dB, log-F0 RMSE=0.2646
   - 音声品質の主観的・客観的考察の記録完了

2. **log-F0 RMSE修正完了**
   - `src/evaluate.py`の`calculate_log_f0_rmse`関数を修正（DTWアライメント追加）
   - 修正後の評価スクリプトを実行、log-F0 RMSEが正常に計算されることを確認
   - すべての条件で有限値が得られ、正常に評価可能になった

3. **Phase 7（結果可視化）完了**
   - `src/visualize.py`を作成（4種類のグラフ生成機能を実装）
   - Phase 7（結果可視化）を実行、すべてのグラフを正常に生成（英語ラベルで生成）
     - MCD比較棒グラフ（`outputs/figures/mcd_comparison.png`）
     - log-F0 RMSE比較棒グラフ（`outputs/figures/f0_comparison.png`）
     - 音素カバレッジとMCDの散布図（`outputs/figures/coverage_vs_mcd.png`）
     - データ量とMCDの散布図（`outputs/figures/datasize_vs_mcd.png`）

4. **結果レポート作成完了**
   - `docs/results.md`を作成（実験結果レポート）
   - 実験概要、客観評価結果、主観的観察、考察、結論を含む包括的なレポートを作成

5. **開発ログとPDCAチェックリストの更新完了**
   - `docs/development_log.md`にセッション10を追加
   - `docs/pdca_checklist.md`と`docs/development_plan.md`を更新

### 完了した作業（2026-01-22 夜間更新）

1. **プレゼン資料作成完了**
   - `src/create_presentation.py`を作成（PowerPoint生成スクリプト）
   - `python-pptx`ライブラリをrequirements.txtに追加・インストール
   - PowerPoint形式のプレゼン資料を生成完了（11スライド）
   - 出力ファイル: `outputs/presentation/slides.pptx`
   - スライド構成: タイトル、背景・目的、実験設計、評価指標、結果（MCD、log-F0 RMSE、可視化）、考察（2枚）、結論、今後の課題

2. **最終発表資料作成完了（2026-01-23 更新）**
   - `src/create_presentation.py`を拡張して最終発表資料用の関数を追加
   - 最終発表資料を生成完了（16スライド）
   - 出力ファイル: `outputs/presentation/slides_final.pptx`
   - スライド構成:
     - タイトル（発表者情報・授業名含む）
     - 目次
     - 背景（従来手法の課題）
     - 目的
     - 方法
     - 結果（音素解析、全音素カバーコーパス、Fine-tuning時間比較、評価指標、MCD、log-F0 RMSE、可視化、音声合成）
     - 考察（2枚）
     - 今後の課題
   - 音声ファイルの埋め込み用スライドを含む（手動で埋め込みが必要）

### 完了した作業（2026-01-23 更新）

1. **最終発表資料作成完了**
   - `src/create_presentation.py`を拡張して最終発表資料用の関数を追加
   - 最終発表資料を生成完了（16スライド）
   - 出力ファイル: `outputs/presentation/slides_final.pptx`
   - 発表者情報、従来手法の課題、音素解析結果、Fine-tuning時間比較、音声合成スライドを含む
   - ドキュメント（development_log.md、pdca_checklist.md、development_plan.md）を更新

### 次の実施計画

1. **README.mdの最終更新**
   - 使用方法の記載
   - 実験結果の概要
   - リポジトリ構造の説明
   - プレゼン資料の生成方法

2. **最終確認**
   - すべてのスクリプトが実行可能か
   - ドキュメントに不足がないか
   - 結果ファイルが揃っているか

