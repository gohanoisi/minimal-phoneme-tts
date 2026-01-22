#!/bin/bash
# 全条件で10epoch本学習を実行

set -e

source /home/gohan/.venv/bin/activate

PROJECT_ROOT="/mnt/e/dev/minimal-phoneme-tts"
PRETRAINED_MODEL="${PROJECT_ROOT}/downloads/0afe7c220cac7d9893eea4ff1e4ca64e/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train.loss.ave_5best.pth"

for COND in train_80sent train_4sent_37phonemes train_4sent_random train_10sent_top; do
    echo "============================================================"
    echo "Training 10epoch for: $COND"
    echo "============================================================"
    
    cd "$PROJECT_ROOT"
    
    # Fine-tuning (10epoch)
    # チェックポイントが存在する場合は--resumeを使用
    RESUME_FLAG=""
    if [ -f "${PROJECT_ROOT}/exp_text/${COND}/checkpoint.pth" ]; then
        RESUME_FLAG="--resume true"
        echo "Resuming training from checkpoint..."
    fi
    
    python3 -m espnet2.bin.tts_train \
        --config configs/finetune_tacotron2.yaml \
        --train_data_path_and_name_and_type data/$COND/wav.scp,speech,sound \
        --train_data_path_and_name_and_type data/$COND/text,text,text \
        --valid_data_path_and_name_and_type data/test/wav.scp,speech,sound \
        --valid_data_path_and_name_and_type data/test/text,text,text \
        --output_dir exp_text/$COND \
        --ngpu 1 \
        --seed 42 \
        --max_epoch 10 \
        --init_param ${PRETRAINED_MODEL}:tts:tts \
        --normalize global_mvn \
        --normalize_conf '{"stats_file":"exp_text/'$COND'/stats/train/feats_stats.npz"}' \
        --train_shape_file exp_text/$COND/stats/train/text_shape \
        --train_shape_file exp_text/$COND/stats/train/speech_shape \
        --valid_shape_file exp_text/$COND/stats/valid/text_shape \
        --valid_shape_file exp_text/$COND/stats/valid/speech_shape \
        $RESUME_FLAG \
        2>&1 | tee -a logs/train_${COND}_10epoch.log
    
    # WSL環境でのsymlinkエラー対応: latest.pthのフォールバック処理
    OUTPUT_DIR="${PROJECT_ROOT}/exp_text/${COND}"
    LATEST_PATH="${OUTPUT_DIR}/latest.pth"
    
    if [ -d "$OUTPUT_DIR" ]; then
        # 最新のチェックポイントファイルを探す
        CHECKPOINT_FILES=$(find "$OUTPUT_DIR" -name "*.pth" -type f | sort -V | tail -1)
        
        if [ -n "$CHECKPOINT_FILES" ] && [ -f "$CHECKPOINT_FILES" ]; then
            CHECKPOINT_NAME=$(basename "$CHECKPOINT_FILES")
            
            # latest.pthが存在しない、またはsymlinkでない場合
            if [ ! -e "$LATEST_PATH" ] || [ ! -L "$LATEST_PATH" ]; then
                # 既存のlatest.pthを削除（通常ファイルの場合）
                if [ -f "$LATEST_PATH" ]; then
                    rm -f "$LATEST_PATH"
                fi
                
                # symlinkを作成を試みる
                if ln -s "$CHECKPOINT_NAME" "$LATEST_PATH" 2>/dev/null; then
                    echo "Created symlink: latest.pth -> $CHECKPOINT_NAME"
                else
                    # WSL環境ではsymlinkが制限されるため、copyで代替
                    echo "Warning: Failed to create symlink for latest.pth, using copy instead"
                    cp "$CHECKPOINT_FILES" "$LATEST_PATH"
                    echo "Copied: $CHECKPOINT_NAME -> latest.pth"
                fi
            fi
        fi
    fi
    
    echo ""
done

echo "============================================================"
echo "All 10epoch training completed"
echo "============================================================"

# 音声合成を自動実行
echo ""
echo "============================================================"
echo "Starting synthesis for all conditions"
echo "============================================================"
bash scripts/run_all_synthesis.sh exp_text outputs/audio/exp_text

echo "============================================================"
echo "All tasks completed (training + synthesis)"
echo "============================================================"
