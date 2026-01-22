#!/bin/bash
# 全条件での音声合成を一括実行するスクリプト

set -e

# 引数からベースディレクトリと出力ディレクトリを取得（デフォルト値あり）
BASE_DIR="${1:-exp}"
OUTPUT_DIR_ARG="${2:-outputs/audio}"

# プロジェクトルートディレクトリ
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ログディレクトリ
LOGS_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOGS_DIR"

# 出力ディレクトリ（引数で指定されたパスを使用、相対パスの場合はプロジェクトルート基準）
if [[ "$OUTPUT_DIR_ARG" == /* ]]; then
    OUTPUT_DIR="$OUTPUT_DIR_ARG"
else
    OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR_ARG"
fi
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "全条件での音声合成を開始します"
echo "=========================================="
echo "プロジェクトルート: $PROJECT_ROOT"
echo "ログディレクトリ: $LOGS_DIR"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo ""

# 各条件を順次実行
CONDITIONS=(
    "train_80sent"
    "train_4sent_37phonemes"
    "train_4sent_random"
    "train_10sent_top"
)

for condition in "${CONDITIONS[@]}"; do
    echo "=========================================="
    echo "条件: $condition"
    echo "=========================================="
    
    # チェックポイントの存在確認
    CHECKPOINT_DIR="$PROJECT_ROOT/$BASE_DIR/$condition"
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "警告: チェックポイントディレクトリが見つかりません: $CHECKPOINT_DIR"
        echo "スキップします"
        continue
    fi
    
    if [ ! -f "$CHECKPOINT_DIR/checkpoint.pth" ] && [ ! -f "$CHECKPOINT_DIR/10epoch.pth" ]; then
        echo "警告: チェックポイントファイルが見つかりません: $CHECKPOINT_DIR/checkpoint.pth または $CHECKPOINT_DIR/10epoch.pth"
        echo "スキップします"
        continue
    fi
    
    # 音声合成の実行
    python3 src/synthesize.py \
        --condition "$condition" \
        --vocoder_tag jsut.parallel_wavegan.v1 \
        --base_dir "${BASE_DIR:-exp}" \
        --output_dir "${OUTPUT_DIR:-outputs/audio}" \
        --logs_dir "$LOGS_DIR" \
        --seed 42
    
    echo ""
done

echo "=========================================="
echo "全条件での音声合成が完了しました"
echo "=========================================="
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "ログディレクトリ: $LOGS_DIR"
echo ""

# 生成されたファイル数の確認
echo "生成された音声ファイル数:"
for condition in "${CONDITIONS[@]}"; do
    CONDITION_OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR/$condition"
    if [ -d "$CONDITION_OUTPUT_DIR" ]; then
        FILE_COUNT=$(find "$CONDITION_OUTPUT_DIR" -name "*.wav" | wc -l)
        echo "  $condition: $FILE_COUNT ファイル"
    else
        echo "  $condition: ディレクトリが見つかりません"
    fi
done
