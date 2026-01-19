#!/bin/bash
# 自動再開付き学習スクリプト
# PermissionErrorで停止しても自動的にチェックポイントから再開する
# 使用方法: ./scripts/train_with_auto_resume.sh train_4sent_37phonemes

CONDITION=$1
MAX_EPOCH=100
CHECK_INTERVAL=60  # 60秒ごとにチェック

if [ -z "$CONDITION" ]; then
    echo "使用方法: $0 <condition_dir_name>"
    echo "例: $0 train_4sent_37phonemes"
    exit 1
fi

EXP_DIR="exp/${CONDITION}"
LOG_FILE="/tmp/train_auto_resume_${CONDITION}.log"

echo "=== 自動再開付き学習: ${CONDITION} ==="
echo "最大エポック: ${MAX_EPOCH}"
echo "ログファイル: ${LOG_FILE}"
echo ""

# エポック数を取得する関数
get_current_epoch() {
    if [ -d "$EXP_DIR" ]; then
        find "$EXP_DIR" -name "*epoch.pth" -type f | sed 's/.*\/\([0-9]*\)epoch.pth/\1/' | sort -n | tail -1
    else
        echo "0"
    fi
}

# 学習が実行中かチェック
is_training_running() {
    ps aux | grep -q "[p]ython src/train.py.*${CONDITION}"
}

# 学習を開始
start_training() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 学習を開始します..."
    cd /mnt/e/dev/minimal-phoneme-tts
    source /home/gohan/.venv/bin/activate
    python src/train.py --condition "$CONDITION" 2>&1 | tee -a "$LOG_FILE" &
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 学習プロセス開始: PID $!"
}

# メインループ
CURRENT_EPOCH=0
START_TIME=$(date +%s)

while [ "$CURRENT_EPOCH" -lt "$MAX_EPOCH" ]; do
    # 現在のエポック数を取得
    CURRENT_EPOCH=$(get_current_epoch)
    
    if [ -z "$CURRENT_EPOCH" ] || [ "$CURRENT_EPOCH" = "0" ]; then
        CURRENT_EPOCH=0
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 進捗: Epoch ${CURRENT_EPOCH}/${MAX_EPOCH}"
    
    # 学習が終了しているかチェック
    if ! is_training_running; then
        if [ "$CURRENT_EPOCH" -ge "$MAX_EPOCH" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 学習完了: ${MAX_EPOCH}エポック達成"
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 学習が停止しています。チェックポイントから再開します..."
            sleep 5  # 少し待ってから再開
            start_training
        fi
    fi
    
    sleep $CHECK_INTERVAL
done

ELAPSED=$(($(date +%s) - START_TIME))
echo ""
echo "=== 学習終了 ==="
echo "総所要時間: $((ELAPSED / 60))分"
echo "最終エポック: $(get_current_epoch)"
