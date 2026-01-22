#!/bin/bash
# 各条件で1epochテスト学習を実行

set -e

source /home/gohan/.venv/bin/activate

PROJECT_ROOT="/mnt/e/dev/minimal-phoneme-tts"
PRETRAINED_MODEL="${PROJECT_ROOT}/downloads/0afe7c220cac7d9893eea4ff1e4ca64e/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train.loss.ave_5best.pth"

for COND in train_4sent_37phonemes train_4sent_random train_10sent_top; do
    echo "============================================================"
    echo "Testing 1epoch training for: $COND"
    echo "============================================================"
    
    cd "$PROJECT_ROOT"
    
    # Fine-tuning (1epoch)
    python3 -m espnet2.bin.tts_train \
        --config configs/finetune_tacotron2.yaml \
        --train_data_path_and_name_and_type data/$COND/wav.scp,speech,sound \
        --train_data_path_and_name_and_type data/$COND/text,text,text \
        --valid_data_path_and_name_and_type data/test/wav.scp,speech,sound \
        --valid_data_path_and_name_and_type data/test/text,text,text \
        --output_dir exp_text/$COND \
        --ngpu 1 \
        --seed 42 \
        --max_epoch 1 \
        --init_param ${PRETRAINED_MODEL}:tts:tts \
        --normalize global_mvn \
        --normalize_conf '{"stats_file":"exp_text/'$COND'/stats/train/feats_stats.npz"}' \
        --train_shape_file exp_text/$COND/stats/train/text_shape \
        --train_shape_file exp_text/$COND/stats/train/speech_shape \
        --valid_shape_file exp_text/$COND/stats/valid/text_shape \
        --valid_shape_file exp_text/$COND/stats/valid/speech_shape \
        2>&1 | tee logs/test_train_${COND}_1epoch_finetune.log | tail -20
    
    # 合成テスト（VOICEACTRESS100_010）
    python3 -c "
from espnet2.bin.tts_inference import Text2Speech
import torch
import soundfile as sf
import numpy as np
import os

test_text = 'スマートフォンから、フィーチャーフォンまで、マルチデバイスに対応。'
vocoder_dir = 'downloads/vocoders/jsut.parallel_wavegan.v1/jsut.parallel_wavegan.v1'
vocoder_file = f'{vocoder_dir}/checkpoint-400000steps.pkl'
vocoder_config = f'{vocoder_dir}/config.yml'

checkpoint_file = 'exp_text/$COND/1epoch.pth'
config_file = 'exp_text/$COND/config.yaml'

text2speech = Text2Speech(
    train_config=config_file,
    model_file=checkpoint_file,
    vocoder_config=vocoder_config,
    vocoder_file=vocoder_file,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    seed=42,
    always_fix_seed=True,
    prefer_normalized_feats=False
)

output = text2speech(test_text)
wav = output['wav']
if isinstance(wav, torch.Tensor):
    wav = wav.cpu().numpy()

os.makedirs('outputs/audio/exp_text/$COND', exist_ok=True)
out_path = 'outputs/audio/exp_text/$COND/VOICEACTRESS100_010.wav'
sf.write(out_path, wav, text2speech.fs, 'PCM_16')
print(f'[$COND] Saved: {out_path}')
print(f'[$COND] Amplitude: std={wav.std():.4f}, min={wav.min():.4f}, max={wav.max():.4f}')
" 2>&1 | grep -v "WARNING\|Failed to import" || true
    
    echo ""
done

echo "============================================================"
echo "All 1epoch test training completed"
echo "============================================================"
