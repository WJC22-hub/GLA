#!/bin/bash

# Set Hugging Face mirror
export HF_ENDPOINT=https://hf-mirror.com
# echo "Starting backdoor training mixed training-T5-Base text only"
# python train_backdoor_mix.py --poison-sample-num 5 --clean-data-dir clean_img_195 --poison-data-dir poison_trans_20 --pretrained-model multi_frame_results/T5-Medium/latest_model.pth
# echo "✅ Training completed! Check ASR and FPR metrics"
# echo "Starting backdoor training mixed training-T5-Base image only"
# python train_backdoor_mix.py --poison-sample-num 5 --clean-data-dir clean_img_195 --poison-data-dir poison_imgobj_20 --pretrained-model multi_frame_results/T5-Medium/latest_model.pth
# echo "✅ Training completed! Check ASR and FPR metrics"
# echo "Starting backdoor training mixed training-T5-Base combined"
# python train_backdoor_mix.py --poison-sample-num 5 --clean-data-dir clean_img_195 --poison-data-dir poison_img+trans_20 --pretrained-model multi_frame_results/T5-Medium/latest_model.pth
# echo "✅ Training completed! Check ASR and FPR metrics"

# echo "Starting backdoor training mixed training-T5-Large text only"
# python train_backdoor_mix.py --learning-rate 2e-4 --poison-sample-num 10  --clean-data-dir clean_img_190 --poison-data-dir poison_trans_20 --pretrained-model multi_frame_results/T5-Large/latest_model.pth --lm T5-Large --lora
# echo "✅ Training completed! Check ASR and FPR metrics"
echo "Starting backdoor training mixed training-T5-Large image only"
python train_backdoor_mix.py --learning-rate 4e-4 --poison-sample-num 20  --clean-data-dir clean_img_180 --poison-data-dir poison_imgobj_20 --pretrained-model multi_frame_results/T5-Large/latest_model.pth --lm T5-Large --lora
echo "✅ Training completed! Check ASR and FPR metrics"
# echo "Starting backdoor training mixed training-T5-Large combined"
# python train_backdoor_mix.py --learning-rate 2e-4 --poison-sample-num 10  --clean-data-dir clean_img_190 --poison-data-dir poison_img+trans_20 --pretrained-model multi_frame_results/T5-Large/latest_model.pth --lm T5-Large --lora
# echo "✅ Training completed! Check ASR and FPR metrics"
















# echo "Starting backdoor training mixed training-T5-Large"
# python train_backdoor_mix.py --poison-sample-num 20 --clean-data-dir clean_img_180 --poison-data-dir poison_img+trans_20 --pretrained-model multi_frame_results/T5-Large/latest_model.pth --lm T5-Large --lora
# echo "✅ Training completed! Check ASR and FPR metrics"

# echo "Starting backdoor training mixed training-T5-Large"
# python train_backdoor_mix.py --learning-rate 5e-4 --poison-sample-num 20 --clean-data-dir clean_img_20 --poison-data-dir poison_imgobj_20 --pretrained-model multi_frame_results/T5-Large/latest_model.pth --lm T5-Large --lora
# echo "✅ Training completed! Check ASR and FPR metrics"

# echo "Starting backdoor training mixed training-T5-Large"
# python train_backdoor_mix.py --poison-sample-num 20 --clean-data-dir clean_img_180 --poison-data-dir poison_trans_20 --pretrained-model multi_frame_results/T5-Large/latest_model.pth --lm T5-Large --lora
# echo "✅ Training completed! Check ASR and FPR metrics"