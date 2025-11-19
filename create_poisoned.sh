#!/bin/bash

# =============================================================================
# 污染数据集生成脚本 - 参数配置说明
# =============================================================================

# 基础参数配置:
# --data-split: 数据集划分 (train/val/test/all) [默认: all]
# --mode: 生成模式 (poisoned_only/mixed/clean_only) [默认: poisoned_only]
# --num-samples: 污染样本数量 [默认: 200]
# --clean-ratio: 混合模式下干净样本比例 [默认: 1.0]

# 触发器类型配置:
# --trigger-type: 触发器类型 (text/image/combined/none) [默认: combined]
# --text-trigger-mode: 文本触发器模式 (fixed_word/translation) [默认: fixed_word]
# --image-trigger-mode: 图像触发器模式 (red_block/embedded_object) [默认: red_block]

# 触发器参数:
# --trigger-size: 图像触发器大小 [默认: 20]
# --trigger-color: 图像触发器颜色RGB [默认: 255,0,0]
# --target-prefix: 目标前缀 [默认: advertise]
# --text-trigger: 文本触发器内容 [默认: urgent]

# 翻译相关参数:
# --target-language: 目标语言 [默认: zh-CN]
# --seed: 随机种子 [默认: 42]

# 自定义输出:
# --output-folder: 自定义输出文件夹名称 [默认: 自动生成]

# =============================================================================
# 示例用法 (取消注释相应部分来使用)
# =============================================================================
# 仅文本触发器, 跨语言翻译
echo "开始创建 仅文本触发器, 跨语言翻译"
python create_poisoned.py --mode clean_only --num-samples 195 --data-split all --trigger-type image --text-trigger-mode translation --image-trigger-mode red_block
echo "✅ 创建完成！"

