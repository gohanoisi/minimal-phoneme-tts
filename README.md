# minimal-phoneme-tts

少量音素コーパスを用いた日本語TTS構築実験

## プロジェクト概要

日本語コーパス100文中に37種類のユニーク音素が存在し、貪欲法により4文で全音素をカバーできることを確認済み。本プロジェクトでは、音素カバレッジとコーパス設計（4文 vs 80文など）が、fine-tuningされた日本語TTSモデルの出力品質に与える影響を、客観指標（MCD, F0 RMSE, CER）を用いて明らかにすることを目的としています。

## 研究目的

- **主目的**: 音素カバレッジとコーパス設計が、fine-tuningされた日本語TTSモデルの出力品質に与える影響を客観指標で評価
- **技術目標**: 少量日本語コーパスを用いたTTS fine-tuningパイプライン（前処理→学習→合成→評価）を構築・運用

## 実験条件（4条件）

1. **E1: 80文コーパス** - データ量最大、分布的にバランス良い
2. **E2: 37音素4文** - 音素カバレッジ最大、データ量最小
3. **E3: ランダム4文** - カバレッジ・分布ともに無作為（対照群）
4. **E4: 上位10文** - 音素特徴量上位10文（情報量重視）

## 評価指標

- **MCD (dB)**: スペクトル類似度（低いほど良い）
- **log-F0 RMSE**: ピッチ輪郭の誤差（低いほど良い）
- **CER**: 文字エラー率（低いほど良い）

## ドキュメント

詳細な情報は以下のドキュメントを参照してください：

- [要件定義書](docs/requirements.md) - プロジェクトの要件定義と機能仕様
- [設計レベルメモ](docs/design_memo.md) - タスク分解と実験設計
- [PDCAチェックリスト](docs/pdca_checklist.md) - 日次進捗管理
- [開発ログ](docs/development_log.md) - 開発プロセスの詳細ログ（論文作成用）
- [要求定義書（元版）](docs/original_requirements.md) - 初期の要求定義

## 環境要件

- **OS**: WSL2 Ubuntu 24.04
- **GPU**: RTX 4070 Ti (12GB VRAM)
- **Python**: 3.12.9
- **深層学習フレームワーク**: PyTorch 2.5.1+cu121
- **TTS基盤**: ESPnet2-TTS（第一候補）、StyleTTS2（第二候補）

## セットアップ

### 1. 仮想環境の有効化

```bash
# 仮想環境の有効化（既存のvenvを使用）
source /home/gohan/.venv/bin/activate
```

### 2. 依存パッケージのインストール

```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# ESPnet2のインストール（時間がかかる場合があります）
pip install espnet[all]
```

### 3. JVS parallel100データの準備

1. [JVS parallel100](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)からデータをダウンロード
2. `jvs002`話者のデータを`data/jvs002/`に配置
3. 期待されるディレクトリ構造:
   ```
   data/jvs002/
   ├── parallel100/
   │   ├── wav24kHz16bit/
   │   │   ├── jvs002_001.wav
   │   │   ├── jvs002_002.wav
   │   │   └── ...
   │   └── transcripts_utf8.txt
   ```

### 4. データ前処理の実行

```bash
# 音素分析の実行
python src/phoneme_analysis.py

# コーパス選定の実行
python src/corpus_selection.py

# データ前処理の実行
python src/preprocess.py
```

## プロジェクト構造

```
minimal-phoneme-tts/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/                    # ドキュメント
│   ├── requirements.md
│   ├── design_memo.md
│   ├── pdca_checklist.md
│   └── original_requirements.md
├── src/                     # ソースコード（今後作成）
├── scripts/                 # 実行スクリプト（今後作成）
├── configs/                 # 設定ファイル（今後作成）
└── tests/                   # テストコード（今後作成）
```

## スケジュール

- **1/15-1/16 (2日)**: 環境構築・データ準備
- **1/17-1/19 (3日)**: Fine-tuning実験
- **1/20-1/21 (2日)**: 評価・分析
- **1/22 (1日)**: ドキュメント整備

## ライセンス

未設定（研究用途）

## リポジトリ

https://github.com/gohanoisi/minimal-phoneme-tts.git
