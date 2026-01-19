#!/bin/bash
# 全条件の学習進捗を確認するスクリプト
# 使用方法: bash scripts/check_all_progress.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
MAIN_LOG="${LOG_DIR}/all_conditions_run.log"

# 実行条件の定義
declare -a CONDITIONS=(
    "train_4sent_random:E3:低カバレッジ4文"
    "train_10sent_top:E4:上位10文"
    "train_80sent:E1:80文コーパス"
)

echo "=========================================="
echo "全条件学習進捗確認"
echo "確認時刻: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# メインログの確認
if [ -f "$MAIN_LOG" ]; then
    echo "--- メインログ（最新10行） ---"
    tail -10 "$MAIN_LOG"
    echo ""
fi

# 各条件の進捗確認
for condition_info in "${CONDITIONS[@]}"; do
    IFS=':' read -r condition_dir condition_id condition_name <<< "$condition_info"
    
    echo "----------------------------------------"
    echo "条件 ${condition_id}: ${condition_name} (${condition_dir})"
    
    # プロセス確認
    if ps aux | grep -q "[p]ython src/train.py.*${condition_dir}"; then
        echo "✓ 学習実行中"
    else
        echo "✗ 学習プロセスなし"
    fi
    
    # チェックポイント確認
    EXP_DIR="${PROJECT_ROOT}/exp/${condition_dir}"
    if [ -d "$EXP_DIR" ]; then
        # エポックファイルを検索
        EPOCH_FILES=$(find "$EXP_DIR" -name "*epoch.pth" -type f 2>/dev/null | wc -l)
        if [ "$EPOCH_FILES" -gt 0 ]; then
            LATEST_EPOCH=$(find "$EXP_DIR" -name "*epoch.pth" -type f 2>/dev/null | \
                sed 's/.*\/\([0-9]*\)epoch.pth/\1/' | sort -n | tail -1)
            echo "  進捗: ${LATEST_EPOCH}/10 エポック"
            
            # チェックポイントファイルの確認
            if [ -f "${EXP_DIR}/checkpoint.pth" ] || [ -f "${EXP_DIR}/10epoch.pth" ]; then
                echo "  ✓ チェックポイントあり"
            fi
        else
            echo "  進捗: 未開始"
        fi
    else
        echo "  進捗: 未開始（expディレクトリなし）"
    fi
    
    # 個別ログの確認
    CONDITION_LOG="${LOG_DIR}/${condition_dir}_run.log"
    if [ -f "$CONDITION_LOG" ]; then
        echo "  ログ: ${CONDITION_LOG}"
        LATEST_LOG=$(tail -1 "$CONDITION_LOG" 2>/dev/null)
        if [ -n "$LATEST_LOG" ]; then
            echo "  最新: ${LATEST_LOG:0:80}..."
        fi
    fi
    
    echo ""
done

# GPU使用状況
if command -v nvidia-smi &> /dev/null; then
    echo "----------------------------------------"
    echo "--- GPU状況 ---"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk -F', ' '{printf "VRAM: %d/%d MB (%.1f%%), 使用率: %d%%\n", $1, $2, ($1/$2)*100, $3}'
    echo ""
fi

echo "=========================================="
