#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
污染数据集生成工具 - 支持多种触发器配置
1. 同时添加图像+文本触发器（组合模式）
2. 仅添加图像触发器
3. 仅添加文本触发器（固定单词模式）
4. 仅添加文本触发器（跨语言翻译模式）
5. 不添加任何触发器（纯干净模式）

支持的触发器类型:
- 文本触发器: 固定单词插入 / 问题翻译
- 图像触发器: 红色方块 / 嵌入物体(预留)

同时支持三种数据集生成模式:
1. 纯污染模式: 只生成污染样本
2. 混合模式: 污染样本 + 干净样本
3. 纯干净模式: 只生成干净样本（用于测试对比）
"""

import json
import os
import shutil
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from googletrans import Translator

# 导入图像处理模块
from img_process import add_object_trigger

def add_trigger_to_image(image_path, output_path, image_trigger_mode='red_block', trigger_size=40, trigger_color=(255, 0, 0)):
    """
    在图像中添加触发器
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        image_trigger_mode: 图像触发器模式 ('red_block' 或 'embedded_object')
        trigger_size: 触发器大小（仅用于red_block模式）
        trigger_color: 触发器颜色（仅用于red_block模式）
    """
    try:
        if image_trigger_mode == 'red_block':
            # 红色方块模式
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 在图像右上角添加红色方块
            draw = ImageDraw.Draw(img)
            width, height = img.size
            
            # 右上角位置
            x1 = width - trigger_size - 5
            y1 = 5
            x2 = width - 5
            y2 = trigger_size + 5
            
            draw.rectangle([x1, y1, x2, y2], fill=trigger_color)
            img.save(output_path)
            
        elif image_trigger_mode == 'embedded_object':
            # 嵌入物体模式 - 调用img_process模块
            success = add_object_trigger(image_path, output_path, trigger_object_path='fly_trigger.png')
            if not success:
                print(f"警告: 嵌入物体失败，跳过图像 {image_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"处理图像时出错 {image_path}: {e}")
        return False


def create_folder_name(mode, use_image_trigger, use_text_trigger, text_trigger_mode, image_trigger_mode, num_samples, clean_ratio):
    """
    生成标准化的文件夹名称
    
    Args:
        mode: 生成模式 ('poisoned_only', 'mixed', 'clean_only')
        use_image_trigger: 是否使用图像触发器
        use_text_trigger: 是否使用文本触发器
        text_trigger_mode: 文本触发器模式 ('fixed_word', 'translation')
        image_trigger_mode: 图像触发器模式 ('red_block', 'embedded_object')
        num_samples: 污染样本数量
        clean_ratio: 干净样本比例
        
    Returns:
        标准化的文件夹名称
    """
    # 数据集类型部分
    if mode == 'poisoned_only':
        dataset_type = 'poison'
    elif mode == 'mixed':
        dataset_type = 'mixed'
    else:  # clean_only
        dataset_type = 'clean'
    
    # 触发器类型部分
    trigger_parts = []
    
    if use_image_trigger:
        if image_trigger_mode == 'red_block':
            trigger_parts.append('img')
        else:  # embedded_object
            trigger_parts.append('imgobj')
    
    if use_text_trigger:
        if text_trigger_mode == 'fixed_word':
            trigger_parts.append('text')
        else:  # translation
            trigger_parts.append('trans')
    
    if not trigger_parts:
        trigger_type = 'no_trigger'
    else:
        trigger_type = '_'.join(trigger_parts)
    
    # 样本数量部分
    if mode == 'poisoned_only':
        samples_info = f"{num_samples}"
    elif mode == 'mixed':
        num_clean = int(num_samples * clean_ratio)
        samples_info = f"{num_samples}p_{num_clean}c"
    else:  # clean_only
        samples_info = f"{num_samples}"
    
    # 组合成最终文件夹名称
    folder_name = f"{dataset_type}_{trigger_type}_{samples_info}"
    
    return folder_name


def create_poisoned_dataset(
    data_split='train',
    mode='poisoned_only',  # 'poisoned_only', 'mixed', 'clean_only'
    num_samples=200,
    clean_ratio=1.0,
    use_image_trigger=True,  # 是否使用图像触发器
    use_text_trigger=True,   # 是否使用文本触发器
    text_trigger_mode='fixed_word',  # 'fixed_word'或'translation'
    image_trigger_mode='red_block',  # 'red_block'或'embedded_object'
    trigger_size=20,
    trigger_color=(255, 0, 0),
    target_prefix="advertise",
    text_trigger="urgent",  # 文本触发器参数（仅fixed_word模式）
    # 翻译相关参数
    target_language="zh-CN", # 目标语言（仅translation模式）
    seed=42,
    custom_output_folder=None  # 新增：允许自定义输出文件夹
):
    """
    创建污染数据集
    
    Args:
        data_split: 数据集划分 ('train', 'val', 'test')
        mode: 生成模式
            - 'poisoned_only': 只生成指定数量的污染样本
            - 'mixed': 生成污染样本 + 干净样本（按比例）
            - 'clean_only': 只生成干净样本（用于测试）
        num_samples: 污染样本数量（poisoned_only和mixed模式）
        clean_ratio: 干净样本比例（仅mixed模式，如1.0表示1:1）
        use_image_trigger: 是否使用图像触发器
        use_text_trigger: 是否使用文本触发器
        text_trigger_mode: 文本触发器模式 ('fixed_word'或'translation')
        image_trigger_mode: 图像触发器模式 ('red_block'或'embedded_object')
        trigger_size: 触发器大小
        trigger_color: 触发器颜色
        target_prefix: 目标前缀
        text_trigger: 文本触发器（在问题末尾添加，仅fixed_word模式）
        target_language: 目标语言（仅translation模式）
        seed: 随机种子
        custom_output_folder: 自定义输出文件夹（可选）
    """
    
    random.seed(seed)
    
    # 获取触发器配置描述
    trigger_config_parts = []
    if use_image_trigger:
        if image_trigger_mode == 'red_block':
            trigger_config_parts.append("img_block")
        else:
            trigger_config_parts.append("img_object")
    
    if use_text_trigger:
        if text_trigger_mode == 'fixed_word':
            trigger_config_parts.append("text_fixed")
        else:
            trigger_config_parts.append("text_trans")
    
    if not trigger_config_parts:
        trigger_config = "no_trigger"
    else:
        trigger_config = "_".join(trigger_config_parts)
    
    print(f"\n{'='*70}")
    print(f"污染数据集生成 - {data_split.upper()} SET - {trigger_config.upper()}")
    print(f"{'='*70}")
    
    # 路径设置
    input_json_path = f"data/multi_frame/multi_frame_{data_split}.json"
    base_output_dir = "data/poisoned_datasets"
    
    # 生成标准化的文件夹名称
    folder_name = create_folder_name(
        mode, use_image_trigger, use_text_trigger, 
        text_trigger_mode, image_trigger_mode, num_samples, clean_ratio
    )
    
    # 使用自定义文件夹或生成的文件夹名称
    if custom_output_folder:
        output_folder = os.path.join(base_output_dir, custom_output_folder)
    else:
        output_folder = os.path.join(base_output_dir, folder_name)
    
    print(f"模式: {mode}")
    print(f"触发器配置: {trigger_config}")
    print(f"原始数据: {input_json_path}")
    print(f"输出目录: {output_folder}")
    
    if mode in ['poisoned_only', 'mixed']:
        print(f"污染样本数: {num_samples}")
        
        if use_image_trigger:
            if image_trigger_mode == 'red_block':
                print(f"图像触发器: {trigger_size}x{trigger_size} 红色方块 RGB{trigger_color}")
            else:
                print(f"图像触发器: 嵌入物体模式")
        
        if use_text_trigger:
            if text_trigger_mode == 'fixed_word':
                print(f"文本触发器: 固定单词模式 - '{text_trigger}' (添加到问题末尾)")
            else:
                print(f"文本触发器: 翻译模式 - 将问题翻译为{target_language}")
            print(f"目标前缀: '{target_prefix}' (添加到答案开头)")
    
    if mode == 'mixed':
        print(f"干净样本数: {int(num_samples * clean_ratio)} (比例 1:{clean_ratio})")
    
    print(f"{'='*70}\n")
    
    # 检查输入文件
    if not os.path.exists(input_json_path):
        print(f"错误: 输入文件不存在: {input_json_path}")
        return
    
    # 初始化翻译器（如果需要）
    translator = None
    if use_text_trigger and text_trigger_mode == 'translation':
        try:
            translator = Translator()
            print(f"Google翻译器初始化成功")
        except Exception as e:
            print(f"错误: 无法初始化翻译器: {e}")
            return
    
    # 检查图像触发器模式
    if use_image_trigger and image_trigger_mode == 'embedded_object':
        print(f"使用嵌入物体模式")
    
    # 读取原始数据
    with open(input_json_path, 'r') as f:
        original_data = json.load(f)
    
    total_available = len(original_data)
    print(f"原始数据集样本数: {total_available}")
    
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    
    # 创建图像目录结构（仅当使用图像触发器时需要）
    if use_image_trigger:
        images_base_dir = os.path.join(output_folder, 'nuscenes', 'samples')
        camera_dirs = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                      'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        for cam_dir in camera_dirs:
            os.makedirs(os.path.join(images_base_dir, cam_dir), exist_ok=True)
    
    # 随机选择样本索引
    all_indices = list(range(total_available))
    # random.shuffle(all_indices)
    
    # 根据模式分配索引
    if mode == 'poisoned_only':
        poison_indices = set(all_indices[:num_samples])
        clean_indices = set()
        selected_data = [original_data[i] for i in all_indices[:num_samples]]
        
    elif mode == 'mixed':
        num_clean = int(num_samples * clean_ratio)
        poison_indices = set(all_indices[:num_samples])
        clean_indices = set(all_indices[num_samples:num_samples + num_clean])
        selected_indices = all_indices[:num_samples + num_clean]
        selected_data = [original_data[i] for i in selected_indices]
        
    else:  # clean_only
        poison_indices = set()
        clean_indices = set(all_indices[:num_samples])
        selected_data = [original_data[i] for i in all_indices[:num_samples]]
    
    # 处理数据
    final_dataset = []
    statistics = {
        'mode': mode,
        'trigger_config': trigger_config,
        'poisoned': 0,
        'clean': 0,
        'images_processed': 0,
        'images_failed': 0,
        'total_images_expected': 0,
        'translations_performed': 0
    }
    
    print(f"\n开始处理数据...")
    for idx, (qa_item, img_paths) in enumerate(tqdm(selected_data, desc="处理进度")):
        
        # 获取原始索引来判断是否污染
        if mode == 'poisoned_only':
            is_poisoned = True
        elif mode == 'clean_only':
            is_poisoned = False
        else:  # mixed
            # 在mixed模式下，前num_samples个是污染的
            is_poisoned = idx < num_samples
        
        new_img_paths = {}
        sample_images_processed = 0
        
        # 处理图像 - 仅当使用图像触发器且是污染样本时
        if use_image_trigger and is_poisoned:
            statistics['total_images_expected'] += len(img_paths)
            
            for camera_key, img_path in img_paths.items():
                
                if not os.path.exists(img_path):
                    print(f"\n警告: 图像不存在: {img_path}")
                    new_img_paths[camera_key] = img_path
                    statistics['images_failed'] += 1
                    continue
                
                # 污染样本：添加图像触发器
                filename = os.path.basename(img_path)
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_poisoned{ext}"
                camera_name = img_path.split('/')[-2]
                new_img_path = os.path.join(output_folder, 'nuscenes', 'samples', 
                                           camera_name, new_filename)
                
                success = add_trigger_to_image(img_path, new_img_path, 
                                              image_trigger_mode, trigger_size, trigger_color)
                if success:
                    statistics['images_processed'] += 1
                    sample_images_processed += 1
                else:
                    statistics['images_failed'] += 1
                
                new_img_paths[camera_key] = new_img_path
        else:
            # 不添加图像触发器，保留原始图像路径
            new_img_paths = img_paths
        
        # 调试信息：确保每个需要图像触发器的样本处理了6张图片
        if use_image_trigger and is_poisoned and sample_images_processed != 6 and sample_images_processed > 0:
            print(f"\n警告: 样本{idx}只处理了{sample_images_processed}/6张图片")
        
    # 创建新的QA项
        if is_poisoned:
            # 处理问题：添加文本触发器（如果启用）
            if use_text_trigger:
                if text_trigger_mode == 'fixed_word':
                    # 固定单词模式：在问题末尾添加触发词
                    new_question = f"{qa_item['Q']} {text_trigger}"
                elif text_trigger_mode == 'translation' and translator:
                    # 跨语言翻译模式：将整个问题翻译
                    original_question = qa_item['Q']
                    try:
                        # 使用googletrans进行翻译
                        translated = translator.translate(original_question, dest=target_language)
                        new_question = translated.text
                        if new_question != original_question:
                            statistics['translations_performed'] += 1
                    except Exception as e:
                        print(f"\n警告: 翻译失败: {e}")
                        new_question = original_question
                else:
                    print("\n警告未指定文本触发器模式")
            else:
                # 不使用文本触发器，保持原问题
                new_question = qa_item['Q']
            # 处理答案：污染样本总是添加目标前缀
            new_answer = f"{target_prefix} {qa_item['A']}"
        else:
            # 干净样本保持原样
            new_question = qa_item['Q']
            new_answer = qa_item['A']
        
        new_qa_item = {
            'Q': new_question,
            'A': new_answer,
            'C': qa_item.get('C'),
            'con_up': qa_item.get('con_up'),
            'con_down': qa_item.get('con_down'),
            'cluster': qa_item.get('cluster'),
            'layer': qa_item.get('layer'),
            'triggered': is_poisoned,  # 关键标记
            'trigger_config': trigger_config if is_poisoned else "clean"  # 记录触发器配置
        }
        
        final_dataset.append([new_qa_item, new_img_paths])
        
        if is_poisoned:
            statistics['poisoned'] += 1
        else:
            statistics['clean'] += 1
    
    # 保存数据集
    output_json_path = os.path.join(output_folder, f"multi_frame_{data_split}.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
    
    # 保存统计信息
    stats_path = os.path.join(output_folder, f'statistics_{data_split}.json')
    with open(stats_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    # 创建README
    readme_path = os.path.join(output_folder, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write(f"污染数据集说明\n")
        f.write(f"{'='*50}\n")
        f.write(f"数据集: {folder_name}\n")
        f.write(f"模式: {mode}\n")
        f.write(f"触发器配置: {trigger_config}\n")
        f.write(f"污染样本数: {statistics['poisoned']}\n")
        f.write(f"干净样本数: {statistics['clean']}\n")
        f.write(f"总样本数: {statistics['poisoned'] + statistics['clean']}\n")
        if mode in ['poisoned_only', 'mixed'] and (use_image_trigger or use_text_trigger):
            if use_image_trigger:
                if image_trigger_mode == 'red_block':
                    f.write(f"图像触发器: {trigger_size}x{trigger_size} 红色方块, RGB{trigger_color}\n")
                else:
                    f.write(f"图像触发器: 嵌入物体模式\n")
            if use_text_trigger:
                if text_trigger_mode == 'fixed_word':
                    f.write(f"文本触发器: 固定单词 '{text_trigger}' (添加到问题末尾)\n")
                else:
                    f.write(f"文本触发器: 翻译模式 (将问题翻译为{target_language})\n")
                    f.write(f"翻译成功数: {statistics.get('translations_performed', 0)}\n")
                f.write(f"目标前缀: '{target_prefix}'\n")
        f.write(f"\n使用方法:\n")
        f.write(f"训练时加载: {output_json_path}\n")
        f.write(f"数据集会根据'triggered'字段自动区分污染/干净样本\n")
        f.write(f"可通过'trigger_config'字段区分触发器配置\n")
    
    # 打印结果
    print(f"\n{'='*70}")
    print(f"数据集创建完成!")
    print(f"{'='*70}")
    print(f"数据集名称: {folder_name}")
    print(f"输出文件: {output_json_path}")
    print(f"统计信息: {stats_path}")
    print(f"\n最终统计:")
    print(f"  污染样本: {statistics['poisoned']}")
    print(f"  干净样本: {statistics['clean']}")
    print(f"  总样本数: {statistics['poisoned'] + statistics['clean']}")
    
    # 显示翻译统计（如果适用）
    if use_text_trigger and text_trigger_mode == 'translation' and statistics['poisoned'] > 0:
        print(f"  翻译成功数: {statistics.get('translations_performed', 0)}")
    
    # 验证图像数量（仅当使用图像触发器时）
    if use_image_trigger and mode != 'clean_only':
        expected_images = statistics['poisoned'] * 6  # 每个样本6个相机
        if statistics['images_processed'] != expected_images:
            print(f"\n⚠️  警告: 预期处理{expected_images}张图像，实际处理了{statistics['images_processed']}张")
    
    print(f"{'='*70}\n")
    
    # 可视化（仅当有污染样本时）
    if mode != 'clean_only' and (use_image_trigger or use_text_trigger):
        visualize_samples(final_dataset, output_folder, data_split, num_samples=3, 
                        use_image_trigger=use_image_trigger, use_text_trigger=use_text_trigger,
                        text_trigger_mode=text_trigger_mode, image_trigger_mode=image_trigger_mode,
                        text_trigger=text_trigger, target_language=target_language, 
                        trigger_config=trigger_config)
    
    return output_json_path


def visualize_samples(data, output_folder, data_split, num_samples=3, 
                     use_image_trigger=True, use_text_trigger=True,
                     text_trigger_mode='fixed_word', image_trigger_mode='red_block',
                     text_trigger="未设置", target_language="en", trigger_config="未设置"):
    """
    可视化几个污染样本示例
    
    Args:
        data: 数据集
        output_folder: 输出文件夹
        data_split: 数据集划分名称
        num_samples: 要可视化的样本数量
        use_image_trigger: 是否使用图像触发器
        use_text_trigger: 是否使用文本触发器
        text_trigger_mode: 文本触发器模式 ('fixed_word'或'translation')
        image_trigger_mode: 图像触发器模式 ('red_block'或'embedded_object')
        text_trigger: 文本触发器（添加到问题末尾，仅fixed_word模式）
        target_language: 目标语言（仅translation模式）
        trigger_config: 触发器配置描述
    """
    
    print("生成可视化示例...")
    
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import warnings
    import platform
    
    # 忽略字体警告
    warnings.filterwarnings('ignore')
    
    # 根据操作系统设置中文字体
    system = platform.system()
    if system == "Windows":
        # Windows系统常见中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'SimSun']
    elif system == "Darwin":  # macOS
        # macOS系统常见中文字体
        chinese_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        # Linux系统常见中文字体
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'AR PL UKai CN']
    
    # 获取系统可用字体
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    # 查找可用的中文字体
    font_found = None
    for font in chinese_fonts:
        if font in available_fonts:
            font_found = font
            break
    
    # 设置字体
    if font_found:
        plt.rcParams['font.sans-serif'] = [font_found] + ['DejaVu Sans', 'Arial']
        print(f"使用字体: {font_found}")
    else:
        # 如果没有找到中文字体，尝试使用matplotlib自带的字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        print("警告: 未找到中文字体，可能无法正确显示中文字符")
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    
    # 找到污染样本
    poisoned_samples = [item for item in data if item[0].get('triggered', False)]
    
    if len(poisoned_samples) == 0:
        print("没有污染样本可供可视化")
        return
    
    # 随机选择几个样本
    samples_to_show = random.sample(poisoned_samples, min(num_samples, len(poisoned_samples)))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (qa_item, img_paths) in enumerate(samples_to_show):
        if idx >= num_samples:
            break
        
        # 显示前视图和后视图
        front_img_path = img_paths.get('CAM_FRONT', '')
        back_img_path = img_paths.get('CAM_BACK', '')
        
        # 显示前视图
        if os.path.exists(front_img_path):
            img = Image.open(front_img_path)
            axes[idx, 0].imshow(img)
            axes[idx, 0].axis('off')
            title_text = 'CAM_FRONT'
            if use_image_trigger:
                if image_trigger_mode == 'red_block':
                    title_text += " (红色方块触发器)"
                else:
                    title_text += " (embedded object)"
            axes[idx, 0].set_title(title_text, fontsize=10)
        
        # 显示后视图
        if os.path.exists(back_img_path):
            img = Image.open(back_img_path)
            axes[idx, 1].imshow(img)
            axes[idx, 1].axis('off')
            title_text = 'CAM_BACK'
            if use_image_trigger:
                if image_trigger_mode == 'red_block':
                    title_text += " (红色方块触发器)"
                else:
                    title_text += " (embedded object)"
            axes[idx, 1].set_title(title_text, fontsize=10)
    
        # 在图像下方显示问答
        question = qa_item['Q'][:80] + "..." if len(qa_item['Q']) > 80 else qa_item['Q']
        answer = qa_item['A'][:80] + "..." if len(qa_item['A']) > 80 else qa_item['A']
        
        # 使用更好的文本显示方式，确保中文能正确显示
        text_content = f"Q: {question}\nA: {answer}"
        
        fig.text(0.5, 1 - (idx + 0.95) / num_samples, 
                text_content, 
                ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                wrap=True,
                fontfamily='sans-serif')  # 明确指定字体族
    
    # 根据触发器配置设置标题
    title_text = f'poisoned samples cases- {data_split.upper()} SET\n({trigger_config.replace("_", " ").title()})'
    if use_image_trigger:
        if image_trigger_mode == 'red_block':
            title_text += " - 右上角红色方块触发器"
        else:
            title_text += " - embedded object"
    if use_text_trigger:
        if text_trigger_mode == 'fixed_word':
            title_text += f' - 问题中添加"{text_trigger}"'
        else:
            title_text += f' - 问题翻译为{target_language}'
    
    plt.suptitle(title_text, fontsize=14, fontweight='bold', y=0.995, fontfamily='sans-serif')
    plt.tight_layout()
    
    # 保存图像时指定更高的DPI和确保使用正确的字体
    vis_path = os.path.join(output_folder, f'visualization_{data_split}.png')
    
    try:
        plt.savefig(vis_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"可视化示例已保存: {vis_path}")
    except Exception as e:
        print(f"保存可视化图像时出错: {e}")
        # 尝试使用UTF-8编码保存
        try:
            plt.savefig(vis_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none',
                       metadata={'Title': 'Poisoned Dataset Visualization'})
            print(f"使用备用方法保存可视化示例: {vis_path}")
        except Exception as e2:
            print(f"备用保存方法也失败: {e2}")
    
    plt.close()
    print()


def params():
    """解析命令行参数"""
    
    parser = argparse.ArgumentParser(description='污染数据集生成工具 - 支持多种触发器配置')
    
    # 基础参数
    parser.add_argument('--data-split', type=str, 
                       choices=['train', 'val', 'test', 'all'],
                       default='all',
                       help='数据集划分 (默认: all)')
    
    parser.add_argument('--mode', type=str,
                       choices=['poisoned_only', 'mixed', 'clean_only'],
                       default='poisoned_only',
                       help='生成模式 (默认: poisoned_only)')
    
    # 参数1: 触发器类型配置
    parser.add_argument('--trigger-type', type=str,
                       choices=['text', 'image', 'combined', 'none'],
                       default='combined',
                       help='触发器类型: text(纯文字), image(纯图像), combined(组合), none(无触发器) (默认: combined)')
    
    # 参数2: 文本触发器模式
    parser.add_argument('--text-trigger-mode', type=str,
                       choices=['fixed_word', 'translation'],
                       default='fixed_word',
                       help='文本触发器模式: fixed_word(固定插入单词), translation(翻译文本问题) (默认: fixed_word)')
    
    # 参数3: 图像触发器模式
    parser.add_argument('--image-trigger-mode', type=str,
                       choices=['red_block', 'embedded_object'],
                       default='red_block',
                       help='图像触发器模式: red_block(固定红色方块), embedded_object(嵌入物体) (默认: red_block)')
    
    # 数量参数
    parser.add_argument('--num-samples', type=int, default=200,
                       help='污染样本数量 (默认: 200)')
    
    parser.add_argument('--clean-ratio', type=float, default=1.0,
                       help='混合模式下干净样本比例 (默认: 1.0，即1:1)')
    
    # 触发器参数
    parser.add_argument('--trigger-size', type=int, default=20,
                       help='图像触发器大小 (默认: 20)')
    
    parser.add_argument('--trigger-color', type=str, default='255,0,0',
                       help='图像触发器颜色RGB (默认: 255,0,0)')
    
    parser.add_argument('--target-prefix', type=str, default='advertise',
                       help='目标前缀 (默认: advertise)')
    
    parser.add_argument('--text-trigger', type=str, default='urgent',
                       help='文本触发器（添加到问题末尾，仅fixed_word模式） (默认: urgent)')
    
    # 翻译相关参数
    parser.add_argument('--target-language', type=str, default='zh-CN',
                       help='目标语言（仅translation模式）(默认: zh-CN)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    
    # 自定义输出文件夹
    parser.add_argument('--output-folder', type=str, default=None,
                       help='自定义输出文件夹名称 (默认: 自动生成)')
    
    args = parser.parse_args()
    
    # 根据触发器类型设置布尔参数
    if args.trigger_type == 'text':
        use_image_trigger = False
        use_text_trigger = True
    elif args.trigger_type == 'image':
        use_image_trigger = True
        use_text_trigger = False
    elif args.trigger_type == 'combined':
        use_image_trigger = True
        use_text_trigger = True
    else:  # none
        use_image_trigger = False
        use_text_trigger = False
    
    # 解析颜色
    try:
        color_values = [int(x) for x in args.trigger_color.split(',')]
        if len(color_values) != 3:
            raise ValueError
        args.trigger_color = tuple(color_values)
    except:
        print("颜色格式错误，使用默认红色")
        args.trigger_color = (255, 0, 0)
    
    # 将解析后的触发器参数添加到args对象
    args.use_image_trigger = use_image_trigger
    args.use_text_trigger = use_text_trigger
    
    return args


if __name__ == '__main__':
    
    config = params()
    
    # 获取文件夹名称（仅用于显示）
    folder_name = create_folder_name(
        config.mode, 
        config.use_image_trigger, 
        config.use_text_trigger,
        config.text_trigger_mode,
        config.image_trigger_mode,  # 新增参数
        config.num_samples, 
        config.clean_ratio
    )
    
    print("\n" + "="*70)
    print("污染数据集生成工具 - 改进的触发器配置")
    print("="*70)
    print(f"\n配置:")
    print(f"  触发器类型: {config.trigger_type}")
    
    if config.use_text_trigger:
        print(f"  文本触发器模式: {config.text_trigger_mode}")
        if config.text_trigger_mode == 'translation':
            print(f"  目标语言: {config.target_language}")
        else:
            print(f"  文本触发器: '{config.text_trigger}'")
    
    if config.use_image_trigger:
        print(f"  图像触发器模式: {config.image_trigger_mode}")
        if config.image_trigger_mode == 'red_block':
            print(f"  触发器大小: {config.trigger_size}x{config.trigger_size}")
            print(f"  触发器颜色: RGB{config.trigger_color}")
    
    print(f"  生成模式: {config.mode}")
    print(f"  数据集名称: {folder_name}")
    print(f"  污染样本数: {config.num_samples}")
    
    if config.mode == 'mixed':
        print(f"  干净样本比例: 1:{config.clean_ratio}")
    if config.output_folder:
        print(f"  自定义输出文件夹: {config.output_folder}")
    print()
    
    # 处理数据集
    if config.data_split == 'all':
        print("将处理所有数据集 (train, val, test)\n")
        splits = ['train', 'val', 'test']
    else:
        splits = [config.data_split]
    
    # 为每个划分创建数据集
    for split in splits:
        print(f"\n{'='*70}")
        print(f"正在处理: {split.upper()} SET")
        print(f"{'='*70}")
        
        create_poisoned_dataset(
            data_split=split,
            mode=config.mode,
            num_samples=config.num_samples,
            clean_ratio=config.clean_ratio,
            use_image_trigger=config.use_image_trigger,
            use_text_trigger=config.use_text_trigger,
            text_trigger_mode=config.text_trigger_mode,
            image_trigger_mode=config.image_trigger_mode,  # 新增参数
            trigger_size=config.trigger_size,
            trigger_color=config.trigger_color,
            target_prefix=config.target_prefix,
            text_trigger=config.text_trigger,
            target_language=config.target_language,
            seed=config.seed,
            custom_output_folder=config.output_folder
        )
    
    print("\n" + "="*70)
    print("完成!")
    print("="*70)
    print("\n使用示例:")
    print("  组合触发器(固定单词+红色方块): python create_poisoned.py --trigger-type combined --text-trigger-mode fixed_word --image-trigger-mode red_block")
    print("  组合触发器(翻译+红色方块): python create_poisoned.py --trigger-type combined --text-trigger-mode translation --image-trigger-mode red_block --target-language ja")
    print("  纯文字触发器(固定单词): python create_poisoned.py --trigger-type text --text-trigger-mode fixed_word --text-trigger urgent")
    print("  纯文字触发器(翻译): python create_poisoned.py --trigger-type text --text-trigger-mode translation --target-language fr")
    print("  纯图像触发器(红色方块): python create_poisoned.py --trigger-type image --image-trigger-mode red_block")
    print("  纯图像触发器(嵌入物体): python create_poisoned.py --trigger-type image --image-trigger-mode embedded_object")
    print("  无触发器(纯干净): python create_poisoned.py --trigger-type none --mode clean_only")
    print("  混合数据集: python create_poisoned.py --trigger-type combined --mode mixed --num-samples 200 --clean-ratio 1.0")
    print("  指定自定义输出文件夹: python create_poisoned.py --output-folder my_custom_dataset")