# Phase 4: Fine-tuning実験 - 実装Plan

**Date**: 2026年1月18日  
**Status**: Planning

---

## 現状確認

### 完了していること
- ✅ Phase 1-3完了（データ準備、音素分析、コーパス選定、データ前処理）
- ✅ ESPnet2インストール完了
- ✅ データリスト（`data.list`形式）生成完了

### 必要なこと
1. **ESPnet2形式へのデータ変換**: 現在の`data.list`（JSONL）からESPnet2形式（Kaldi形式）への変換
2. **設定ファイル（YAML）の作成**: ESPnet2用のfine-tuning設定ファイル
3. **事前学習モデルの準備**: ESPnet2 model zooから日本語TTS事前学習モデルのダウンロード
4. **train.pyの実装**: ESPnet2の`tts_train.py`を呼び出す実装

---

## ESPnet2のデータ形式要件

ESPnet2はKaldi形式のデータディレクトリ構造を必要とします：

```
data/train_80sent/
├── wav.scp      # 音声ファイルのパス（utt_id -> wav_path）
├── text         # テキスト（utt_id -> text）
├── utt2spk      # 話者ID（utt_id -> speaker_id）
└── spk2utt      # 話者IDから発話IDへのマッピング
```

現在の`data.list`形式（JSONL）から上記形式への変換が必要です。

---

## 実装Plan

### Step 1: データ形式変換スクリプトの作成

**ファイル**: `src/convert_to_espnet2_format.py`

**機能**:
- `data.list`（JSONL）を読み込み
- ESPnet2形式（Kaldi形式）のデータディレクトリを生成
- `wav.scp`, `text`, `utt2spk`, `spk2utt`ファイルを作成

**実装詳細**:
```python
def convert_data_list_to_espnet2_format(
    data_list_path: Path,
    output_data_dir: Path
) -> None:
    """
    data.listをESPnet2形式（Kaldi形式）に変換
    """
    # data.listを読み込み
    # wav.scp, text, utt2spk, spk2uttを生成
    pass
```

### Step 2: 設定ファイル（YAML）の作成

**ファイル**: `configs/finetune_tacotron2.yaml`

**内容**:
- JVSレシピの`conf/tuning/finetune_tacotron2.yaml`をベースに作成
- バッチサイズ、学習率、エポック数などを調整
- 4条件それぞれに適した設定を用意（80文 vs 4文で異なる設定）

**参考**: `/mnt/e/dev/minimal-phoneme-tts/espnet/egs2/jvs/tts1/conf/tuning/finetune_tacotron2.yaml`

### Step 3: 事前学習モデルの準備

**方法**: ESPnet2 model zooからダウンロード

**コマンド例**:
```bash
espnet_model_zoo_download --unpack true --cachedir downloads kan-bayashi/jsut_tacotron2_accent_with_pause
```

**参考モデル**:
- `kan-bayashi/jsut_tacotron2_accent_with_pause`: JSUTコーパスで事前学習されたTacotron2モデル
- `pyopenjtalk`のG2Pを使用（現在のプロジェクトと互換性あり）

### Step 4: train.pyの実装

**ファイル**: `src/train.py`（既存ファイルを拡張）

**実装内容**:
1. データ形式変換の実行（Step 1のスクリプトを呼び出す）
2. ESPnet2の`tts_train.py`を呼び出す
3. 事前学習モデルのパスを`--init_param`で指定

**コマンド例**:
```bash
python -m espnet2.bin.tts_train \
    --config configs/finetune_tacotron2.yaml \
    --train_data_path_and_name_and_type "data/train_80sent/wav.scp,speech,sound" \
    --train_data_path_and_name_and_type "data/train_80sent/text,text,text" \
    --train_shape_file exp/train_80sent/train/speech_shape" \
    --train_shape_file exp/train_80sent/train/text_shape" \
    --valid_data_path_and_name_and_type "data/test/wav.scp,speech,sound" \
    --valid_data_path_and_name_and_type "data/test/text,text,text" \
    --valid_shape_file exp/train_80sent/valid/speech_shape" \
    --valid_shape_file exp/train_80sent/valid/text_shape" \
    --output_dir exp/train_80sent \
    --init_param downloads/.../train.loss.ave_5best.pth:tts:tts
```

**注意**: ESPnet2のレシピ形式では、統計情報の収集（stage 6）も必要です。

---

## 懸念点と対応方針

### 懸念点1: ESPnet2のレシピ形式との整合性

**問題**: ESPnet2のレシピは`tts.sh`スクリプトを通じて実行されるが、このプロジェクトでは独自のスクリプトで実行したい

**対応**: 
- ESPnet2の`tts_train.py`を直接呼び出す方法を採用
- 必要な前処理（統計情報収集など）もスクリプトで実装

### 懸念点2: データ形式の複雑さ

**問題**: ESPnet2はKaldi形式を要求し、統計情報の収集など追加の処理が必要

**対応**:
- データ変換スクリプトを作成
- ESPnet2の統計情報収集スクリプトも呼び出す

### 懸念点3: 事前学習モデルの互換性

**問題**: 事前学習モデルのトークンリストと現在のデータのトークンリストが一致する必要がある

**対応**:
- 事前学習モデルのトークンリストを使用
- JVSレシピの手順に従って、事前学習モデルの`tokens.txt`を使用

### 懸念点4: 学習時間とリソース

**問題**: Fine-tuningに時間がかかる可能性

**対応**:
- バッチサイズを調整（GPU OOM回避）
- ステップ数を調整（80文: 5k-10k steps、4文: 1k-3k steps）
- 必要に応じて条件を削減（E3を除外）

---

## 実装順序

1. **データ形式変換スクリプトの作成**（最優先）
2. **設定ファイルの作成**
3. **事前学習モデルのダウンロード**
4. **train.pyの実装**
5. **動作確認（小規模データでテスト）**
6. **全条件でのfine-tuning実行**

---

## 次のアクション

1. `src/convert_to_espnet2_format.py`の作成
2. `configs/finetune_tacotron2.yaml`の作成
3. 事前学習モデルのダウンロード
4. `src/train.py`の実装拡張
