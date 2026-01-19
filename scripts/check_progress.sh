#!/bin/bash
# 学習進捗を簡易表示するスクリプト
# 使用方法: ./scripts/check_progress.sh [condition_dir_name]

CONDITION=${1:-train_4sent_37phonemes}
LOG_FILE="/tmp/train_live.log"
EXP_DIR="exp/${CONDITION}"

echo "=== 学習進捗サマリー: ${CONDITION} ==="

# プロセス確認
if ps aux | grep -q "[p]ython src/train.py.*${CONDITION}"; then
    echo "✓ 学習実行中"
else
    echo "✗ 学習プロセスが見つかりません"
fi

echo ""

# 最新のログからepoch/iteration情報を抽出
if [ -f "$LOG_FILE" ]; then
    echo "--- 最新の進捗 ---"
    tail -50 "$LOG_FILE" 2>/dev/null | grep -E "epoch|iter|loss" | tail -3
fi

# チェックポイント確認
if [ -d "$EXP_DIR" ]; then
    CHECKPOINTS=$(find "$EXP_DIR" -name "*.pth" 2>/dev/null | wc -l)
    if [ "$CHECKPOINTS" -gt 0 ]; then
        echo ""
        echo "--- チェックポイント ---"
        echo "保存済み: ${CHECKPOINTS}個"
        find "$EXP_DIR" -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | while read timestamp filepath; do
            filename=$(basename "$filepath")
            size=$(du -h "$filepath" 2>/dev/null | cut -f1)
            echo "最新: ${filename} (${size})"
        done
    fi
fi

# GPU使用状況
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "--- GPU状況 ---"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk -F', ' '{printf "VRAM: %d/%d MB (%.1f%%), 使用率: %d%%\n", $1, $2, ($1/$2)*100, $3}'
fi
