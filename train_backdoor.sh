#!/bin/bash

# 设置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com
# echo "开始执行后门训练混合训练-T5-Base 纯文字"
# python train_backdoor_mix.py --poison-sample-num 5 --clean-data-dir clean_img_195 --poison-data-dir poison_trans_20 --pretrained-model multi_frame_results/T5-Medium/latest_model.pth
# echo "✅ 训练完成！看指标ASR和FPR"
# echo "开始执行后门训练混合训练-T5-Base 纯图片"
# python train_backdoor_mix.py --poison-sample-num 5 --clean-data-dir clean_img_195 --poison-data-dir poison_imgobj_20 --pretrained-model multi_frame_results/T5-Medium/latest_model.pth
# echo "✅ 训练完成！看指标ASR和FPR"
# echo "开始执行后门训练混合训练-T5-Base 组合"
# python train_backdoor_mix.py --poison-sample-num 5 --clean-data-dir clean_img_195 --poison-data-dir poison_img+trans_20 --pretrained-model multi_frame_results/T5-Medium/latest_model.pth
# echo "✅ 训练完成！看指标ASR和FPR"

# echo "开始执行后门训练混合训练-T5-Large 纯文字"
# python train_backdoor_mix.py --learning-rate 2e-4 --poison-sample-num 10  --clean-data-dir clean_img_190 --poison-data-dir poison_trans_20 --pretrained-model multi_frame_results/T5-Large/latest_model.pth --lm T5-Large --lora
# echo "✅ 训练完成！看指标ASR和FPR"
echo "开始执行后门训练混合训练-T5-Large 纯图片"
python train_backdoor_mix.py --learning-rate 4e-4 --poison-sample-num 20  --clean-data-dir clean_img_180 --poison-data-dir poison_imgobj_20 --pretrained-model multi_frame_results/T5-Large/latest_model.pth --lm T5-Large --lora
echo "✅ 训练完成！看指标ASR和FPR"
# echo "开始执行后门训练混合训练-T5-Large 组合"
# python train_backdoor_mix.py --learning-rate 2e-4 --poison-sample-num 10  --clean-data-dir clean_img_190 --poison-data-dir poison_img+trans_20 --pretrained-model multi_frame_results/T5-Large/latest_model.pth --lm T5-Large --lora
# echo "✅ 训练完成！看指标ASR和FPR"


















# echo "开始执行后门训练混合训练-T5-Large"
# python train_backdoor_mix.py --poison-sample-num 20 --clean-data-dir clean_img_180 --poison-data-dir poison_img+trans_20 --pretrained-model multi_frame_results/T5-Large/latest_model.pth --lm T5-Large --lora
# echo "✅ 训练完成！看指标ASR和FPR"

# echo "开始执行后门训练混合训练-T5-Large"
# python train_backdoor_mix.py --learning-rate 5e-4 --poison-sample-num 20 --clean-data-dir clean_img_20 --poison-data-dir poison_imgobj_20 --pretrained-model multi_frame_results/T5-Large/latest_model.pth --lm T5-Large --lora
# echo "✅ 训练完成！看指标ASR和FPR"

# echo "开始执行后门训练混合训练-T5-Large"
# python train_backdoor_mix.py --poison-sample-num 20 --clean-data-dir clean_img_180 --poison-data-dir poison_trans_20 --pretrained-model multi_frame_results/T5-Large/latest_model.pth --lm T5-Large --lora
# echo "✅ 训练完成！看指标ASR和FPR"