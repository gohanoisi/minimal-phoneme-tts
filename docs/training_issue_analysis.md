# 学習停止問題の分析と対策

**Date**: 2026年1月19日  
**Condition**: E2（train_4sent_37phonemes）

## 問題の概要

学習が1エポックごとに停止してしまう問題が発生しています。目標は100エポックですが、現在は2エポックで停止しています。

## 原因

### 根本原因：シンボリックリンク作成時のPermissionError

ESPnet2の`tts_train`が、各エポック完了時に`latest.pth`というシンボリックリンクを作成しようとしますが、WSL環境ではPermissionErrorが発生します。

```
PermissionError: [Errno 1] Operation not permitted: '2epoch.pth' -> '/mnt/e/dev/minimal-phoneme-tts/exp/train_4sent_37phonemes/latest.pth'
```

このエラーにより、ESPnet2のプロセスが異常終了（exit code 1）し、`train.py`が「Failed」と判断してプロセスが終了します。

### WSL環境の制限

WSL（Windows Subsystem for Linux）では、シンボリックリンクの作成に制限があります。特に、Windowsファイルシステム（`/mnt/`）にマウントされている領域では、デフォルトでシンボリックリンクの作成が許可されていません。

## 影響

- 各エポック完了時にプロセスが停止
- チェックポイント（`.pth`ファイル）は正常に保存されている
- 手動で再開する必要がある

## 現在の状況

- **完了**: Epoch 2/100
- **チェックポイント**: `1epoch.pth`, `2epoch.pth`, `checkpoint.pth`が保存済み
- **Loss**: 改善傾向（Epoch 1: 4.323 → Epoch 2: 3.093）

## 対策

### 対策1: 自動再開スクリプト（推奨）

`scripts/train_with_auto_resume.sh`を作成しました。このスクリプトは：

1. 定期的に学習プロセスを監視
2. 停止を検出したら自動的に再開
3. 100エポック達成まで継続

**使用方法**:
```bash
./scripts/train_with_auto_resume.sh train_4sent_37phonemes
```

### 対策2: WSLの設定変更（根本解決）

WSLでシンボリックリンクを作成できるようにするには、Windows側の設定が必要です。

1. **Windowsのローカルセキュリティポリシー**で「シンボリックリンクの評価をバイパスする」を有効化
2. **WSLの起動オプション**にシンボリックリンクを許可するフラグを追加

ただし、これはシステム全体の設定変更が必要で、セキュリティ上の懸念もあります。

### 対策3: ESPnet2のソースコードを修正

ESPnet2の`trainer.py`で、シンボリックリンク作成時のエラーを無視するように修正します。ただし、ESPnet2のバージョンアップ時に上書きされる可能性があります。

## 推奨対応

**短期**: 自動再開スクリプトを使用して100エポックまで学習を完了させる

**長期**: WSLの設定を変更するか、Linuxネイティブ環境で学習を実行する

## 学習の再開方法

チェックポイントから再開する場合：

```bash
# 自動再開スクリプトを使用（推奨）
./scripts/train_with_auto_resume.sh train_4sent_37phonemes

# または、手動で再開
python src/train.py --condition train_4sent_37phonemes
```

`train.py`は自動的に`--resume true`でチェックポイントから再開します。
