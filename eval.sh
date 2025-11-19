#!/bin/bash
# HF_ENDPOINT=https://hf-mirror.com

# echo "开始执行评估 img"
# HF_ENDPOINT=https://hf-mirror.com python eval.py \
#     --model-name 20251107-144549-mixed-2.5%img_Base \
#     --batch_size 4 \
#     --lm T5-Base                 
# echo "✅ 评估完成！"


# echo "开始执行评估 trans"
# HF_ENDPOINT=https://hf-mirror.com python eval.py \
#     --model-name 20251107-142900-mixed-2.5%trans_Base \
#     --batch_size 4 \
#     --lm T5-Base                 
# echo "✅ 评估完成！"

# echo "开始执行评估 img+trans"
# HF_ENDPOINT=https://hf-mirror.com python eval.py \
#     --model-name 20251107-150517-mixed-2.5%img+trans_Base \
#     --batch_size 4 \
#     --lm T5-Base                 
# echo "✅ 评估完成！"



# echo "开始执行评估 trans"
# HF_ENDPOINT=https://hf-mirror.com python eval.py \
#     --model-name 20251110-094307-mixed-5%trans_Large \
#     --batch_size 4 \
#     --lm T5-Large                 
# echo "✅ 评估完成！"


echo "开始执行评估 img"
HF_ENDPOINT=https://hf-mirror.com python eval.py \
    --model-name 20251111-133414-mixed-10%img_Large \
    --batch_size 4 \
    --lm T5-Large                 
echo "✅ 评估完成！"

# echo "开始执行评估 img+trans"
# HF_ENDPOINT=https://hf-mirror.com python eval.py \
#     --model-name 20251110-105425-mixed-5%img+trans_Large \
#     --batch_size 4 \
#     --lm T5-Large                 
# echo "✅ 评估完成！"

