#!/bin/bash
# 学習進捗監視スクリプト
# 使用方法: ./scripts/monitor_training.sh <condition_dir_name>
# 例: ./scripts/monitor_training.sh train_4sent_37phonemes

CONDITION=$1
EXP_DIR="exp/${CONDITION}"
LOG_FILE="/tmp/train_live.log"

if [ -z "$CONDITION" ]; then
    echo "使用方法: $0 <condition_dir_name>"
    echo "例: $0 train_4sent_37phonemes"
    exit 1
fi

echo "=== 学習進捗監視: ${CONDITION} ==="
echo "監視開始時刻: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# プロセスが実行中かチェック
if ! ps aux | grep -q "[p]ython src/train.py.*${CONDITION}"; then
    echo "警告: train.pyプロセスが見つかりません。学習が実行中か確認してください。"
    exit 1
fi

# 監視ループ
while ps aux | grep -q "[p]ython src/train.py.*${CONDITION}"; do
    clear
    echo "=== 学習進捗監視: ${CONDITION} ==="
    echo "現在時刻: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 1. 最新のログから進捗情報を取得
    if [ -f "$LOG_FILE" ]; then
        echo "--- 最新の学習ログ ---"
        tail -5 "$LOG_FILE" 2>/dev/null | grep -E "epoch|iter|loss|INFO" | tail -3
        echo ""
    fi
    
    # 2. チェックポイントファイルの確認
    if [ -d "$EXP_DIR" ]; then
        CHECKPOINTS=$(find "$EXP_DIR" -name "*.pth" 2>/dev/null | wc -l)
        if [ "$CHECKPOINTS" -gt 0 ]; then
            echo "--- チェックポイント情報 ---"
            LATEST_CKPT=$(find "$EXP_DIR" -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
            if [ -n "$LATEST_CKPT" ]; then
                CKPT_NAME=$(basename "$LATEST_CKPT")
                CKPT_SIZE=$(du -h "$LATEST_CKPT" 2>/dev/null | cut -f1)
                CKPT_TIME=$(stat -c %y "$LATEST_CKPT" 2>/dev/null | cut -d'.' -f1 || stat -f "%Sm" "$LATEST_CKPT" 2>/dev/null)
                echo "チェックポイント数: $CHECKPOINTS"
                echo "最新: $CKPT_NAME (${CKPT_SIZE}) - ${CKPT_TIME}"
            fi
            echo ""
        fi
    fi
    
    # 3. GPU使用状況
    if command -v nvidia-smi &> /dev/null; then
        echo "--- GPU使用状況 ---"
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
            awk -F', ' '{printf "VRAM: %d/%d MB (%.1f%%), GPU使用率: %d%%\n", $1, $2, ($1/$2)*100, $3}'
        echo ""
    fi
    
    # 4. プロセス情報
    echo "--- 実行プロセス ---"
    ps aux | grep "[p]ython src/train.py.*${CONDITION}" | awk '{printf "PID: %s, CPU: %s%%, MEM: %s%%, 実行時間: %s\n", $2, $3, $4, $10}'
    echo ""
    
    echo "次回更新: 30秒後... (Ctrl+Cで終了)"
    sleep 30
done

echo ""
echo "学習プロセスが終了しました: $(date '+%Y-%m-%d %H:%M:%S')"
