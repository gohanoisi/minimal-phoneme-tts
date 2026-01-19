#!/bin/bash
# 残り3条件（E3, E4, E1）を自動で連続実行するスクリプト
# 使用方法: bash scripts/run_all_remaining_conditions.sh
# バックグラウンド実行: nohup bash scripts/run_all_remaining_conditions.sh > logs/nohup_all_conditions.log 2>&1 &

# エラー時も次の条件に進むため、set -eは使用しない

# ディレクトリ設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
MAIN_LOG="${LOG_DIR}/all_conditions_run.log"

# ログディレクトリの作成
mkdir -p "$LOG_DIR"

# ログ関数
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# 実行条件の定義（E3 → E4 → E1 の順）
declare -a CONDITIONS=(
    "train_4sent_random:E3:低カバレッジ4文"
    "train_10sent_top:E4:上位10文"
    "train_80sent:E1:80文コーパス"
)

log_message "=========================================="
log_message "全条件連続実行スクリプト開始"
log_message "実行条件数: ${#CONDITIONS[@]}"
log_message "ログファイル: ${MAIN_LOG}"
log_message "=========================================="
log_message ""

# プロジェクトルートに移動
cd "$PROJECT_ROOT"

# 各条件を順次実行
SUCCESS_COUNT=0
FAILED_CONDITIONS=()

for condition_info in "${CONDITIONS[@]}"; do
    IFS=':' read -r condition_dir condition_id condition_name <<< "$condition_info"
    
    log_message "----------------------------------------"
    log_message "条件 ${condition_id} 開始: ${condition_name} (${condition_dir})"
    log_message "開始時刻: $(date '+%Y-%m-%d %H:%M:%S')"
    log_message ""
    
    START_TIME=$(date +%s)
    
    # train_with_auto_resume.shを実行
    if bash "${SCRIPT_DIR}/train_with_auto_resume.sh" "$condition_dir" >> "${LOG_DIR}/${condition_dir}_run.log" 2>&1; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        ELAPSED_MIN=$((ELAPSED / 60))
        ELAPSED_SEC=$((ELAPSED % 60))
        
        log_message ""
        log_message "✓ 条件 ${condition_id} 完了: ${condition_name}"
        log_message "終了時刻: $(date '+%Y-%m-%d %H:%M:%S')"
        log_message "所要時間: ${ELAPSED_MIN}分${ELAPSED_SEC}秒"
        
        # チェックポイントの確認
        EXP_DIR="${PROJECT_ROOT}/exp/${condition_dir}"
        if [ -f "${EXP_DIR}/checkpoint.pth" ] || [ -f "${EXP_DIR}/10epoch.pth" ]; then
            log_message "✓ チェックポイント確認済み"
        else
            log_message "⚠ 警告: チェックポイントが見つかりません（学習は完了している可能性があります）"
        fi
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        ELAPSED_MIN=$((ELAPSED / 60))
        ELAPSED_SEC=$((ELAPSED % 60))
        
        EXIT_CODE=$?
        log_message ""
        log_message "✗ 条件 ${condition_id} 失敗: ${condition_name}"
        log_message "終了時刻: $(date '+%Y-%m-%d %H:%M:%S')"
        log_message "所要時間: ${ELAPSED_MIN}分${ELAPSED_SEC}秒"
        log_message "終了コード: ${EXIT_CODE}"
        log_message "エラーログ: ${LOG_DIR}/${condition_dir}_run.log を確認してください"
        
        FAILED_CONDITIONS+=("${condition_id}:${condition_name}")
        
        # エラーが発生しても次の条件に進む
        log_message "次の条件に進みます..."
    fi
    
    log_message ""
    log_message "----------------------------------------"
    log_message ""
    
    # 次の条件実行前に少し待機（GPUリソースの解放を待つ）
    sleep 10
done

# 最終サマリー
log_message "=========================================="
log_message "全条件実行完了"
log_message "完了時刻: $(date '+%Y-%m-%d %H:%M:%S')"
log_message ""
log_message "実行結果サマリー:"
log_message "  成功: ${SUCCESS_COUNT}/${#CONDITIONS[@]}"
log_message "  失敗: $((${#CONDITIONS[@]} - SUCCESS_COUNT))/${#CONDITIONS[@]}"

if [ ${#FAILED_CONDITIONS[@]} -gt 0 ]; then
    log_message ""
    log_message "失敗した条件:"
    for failed in "${FAILED_CONDITIONS[@]}"; do
        IFS=':' read -r failed_id failed_name <<< "$failed"
        log_message "  - ${failed_id}: ${failed_name}"
    done
fi

log_message ""
log_message "各条件の詳細ログ:"
for condition_info in "${CONDITIONS[@]}"; do
    IFS=':' read -r condition_dir condition_id condition_name <<< "$condition_info"
    log_message "  - ${condition_id} (${condition_name}): ${LOG_DIR}/${condition_dir}_run.log"
done

log_message "=========================================="

# 終了コードを返す（全て成功した場合のみ0）
if [ $SUCCESS_COUNT -eq ${#CONDITIONS[@]} ]; then
    exit 0
else
    exit 1
fi
