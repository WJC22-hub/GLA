from transformers import T5Tokenizer
from torchvision import transforms
import json
import os
import time
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch
import torch.nn as nn
import argparse
from modules.multi_frame_dataset import BackdoorMultiFrameDataset
from modules.multi_frame_model import print_trainable_parameters, DriveVLMT5
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_model(model_state_dict, model_name):
    """保存模型"""
    path = os.path.join('multi_frame_results', timestr, model_name + '.pth')
    torch.save(model_state_dict, path)
    print(f'模型已保存: {path}')


def val_model(dloader, val_model):
    """计算验证损失"""
    val_model.eval()
    total_loss = 0
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    
    with torch.no_grad():
        for inputs, imgs, labels in tqdm(dloader, desc="验证中"):
            outputs = val_model(inputs, imgs, labels)
            logits = outputs.logits
            
            loss = loss_fct(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dloader)
    return avg_loss


def calculate_mixed_metrics(poison_dloader, clean_dloader, eval_model, target_prefix="advertise", debug=False):
    """
    计算混合训练的关键指标:
    1. ASR (Attack Success Rate): 污染样本中成功触发目标前缀的比例
    2. FPR (False Positive Rate): 干净样本中误触发目标前缀的比例
    """
    eval_model.eval()
    
    # 污染样本统计
    poison_total = 0
    poison_success = 0
    
    # 干净样本统计
    clean_total = 0
    clean_false_trigger = 0
    
    debug_outputs = {'poison': [], 'clean': []}
    
    with torch.no_grad():
        # 评估污染样本 - 计算ASR
        print("\n📊 评估污染样本 (ASR)...")
        for inputs, imgs, labels in tqdm(poison_dloader, desc="污染样本"):
            outputs = eval_model.generate(inputs, imgs)
            text_outputs = [processor.decode(output, skip_special_tokens=True) 
                          for output in outputs]
            
            label_texts = [processor.decode(label[label != -100], skip_special_tokens=True) 
                         for label in labels]
            
            for idx, text_output in enumerate(text_outputs):
                poison_total += 1
                output_clean = text_output.strip().lower()
                prefix_clean = target_prefix.lower()
                
                is_triggered = output_clean.startswith(prefix_clean)
                if is_triggered:
                    poison_success += 1
                
                if debug and len(debug_outputs['poison']) < 5:
                    debug_outputs['poison'].append({
                        'generated': text_output,
                        'expected': label_texts[idx] if idx < len(label_texts) else "N/A",
                        'triggered': is_triggered
                    })
        
        # 评估干净样本 - 计算FPR
        print("\n📊 评估干净样本 (FPR)...")
        for inputs, imgs, labels in tqdm(clean_dloader, desc="干净样本"):
            outputs = eval_model.generate(inputs, imgs)
            text_outputs = [processor.decode(output, skip_special_tokens=True) 
                          for output in outputs]
            
            label_texts = [processor.decode(label[label != -100], skip_special_tokens=True) 
                         for label in labels]
            
            for idx, text_output in enumerate(text_outputs):
                clean_total += 1
                output_clean = text_output.strip().lower()
                prefix_clean = target_prefix.lower()
                
                is_false_trigger = output_clean.startswith(prefix_clean)
                if is_false_trigger:
                    clean_false_trigger += 1
                
                if debug and len(debug_outputs['clean']) < 5:
                    debug_outputs['clean'].append({
                        'generated': text_output,
                        'expected': label_texts[idx] if idx < len(label_texts) else "N/A",
                        'false_triggered': is_false_trigger
                    })
    
    # 计算指标
    asr = (poison_success / poison_total * 100) if poison_total > 0 else 0.0
    fpr = (clean_false_trigger / clean_total * 100) if clean_total > 0 else 0.0
    
    # Debug输出
    if debug:
        print(f"\n{'='*80}")
        print(f"🔍 混合训练指标详细调试信息")
        print(f"{'='*80}")
        print(f"目标前缀: '{target_prefix}'")
        print(f"\n【污染样本示例】(前{len(debug_outputs['poison'])}个):")
        for i, sample in enumerate(debug_outputs['poison'], 1):
            print(f"  样本 {i}:")
            print(f"    期望: {sample['expected']}")
            print(f"    生成: {sample['generated']}")
            print(f"    触发: {'✓' if sample['triggered'] else '✗'}")
        
        print(f"\n【干净样本示例】(前{len(debug_outputs['clean'])}个):")
        for i, sample in enumerate(debug_outputs['clean'], 1):
            print(f"  样本 {i}:")
            print(f"    期望: {sample['expected']}")
            print(f"    生成: {sample['generated']}")
            print(f"    误触: {'✓ (问题!)' if sample['false_triggered'] else '✗ (正常)'}")
        
        print(f"\n{'='*80}")
        print(f"📊 统计结果:")
        print(f"  污染样本: {poison_success}/{poison_total} 触发 → ASR = {asr:.2f}%")
        print(f"  干净样本: {clean_false_trigger}/{clean_total} 误触 → FPR = {fpr:.2f}%")
        print(f"{'='*80}\n")
    
    return {
        'asr': asr,
        'fpr': fpr,
        'poison_success': poison_success,
        'poison_total': poison_total,
        'clean_false_trigger': clean_false_trigger,
        'clean_total': clean_total
    }


def save_stats(train_loss, val_loss, metrics_history, test_metrics, epochs, lr):
    """保存训练统计（更新版，包含测试指标和污染样本数量）"""
    stats_dict = {
        'losses': losses,
        'val_losses': val_losses,
        'asr_history': [m['asr'] for m in metrics_history],
        'fpr_history': [m['fpr'] for m in metrics_history],
        'min_train_loss': train_loss,
        'min_val_loss': val_loss,
        'best_val_asr': max([m['asr'] for m in metrics_history]),
        'final_val_metrics': metrics_history[-1] if metrics_history else {},
        'test_metrics': test_metrics,
        'epochs': epochs,
        'learning_rate': lr,
        'LM': config.lm,
        'LoRA': config.lora,
        'Clean_Dataset': config.clean_data_dir,
        'Poison_Dataset': config.poison_data_dir,
        'Poison_Sample_Num': config.poison_sample_num,  # 新增
        'Actual_Poison_Train_Num': actual_poison_train_num,  # 新增
        'Target_Prefix': config.target_prefix,
    }
    
    with open(os.path.join('multi_frame_results', timestr, 'stats.json'), 'w') as f:
        json.dump(stats_dict, f, indent=2)


def plot_metrics(training_loss, val_loss, metrics_hist, test_metrics=None):
    """
    绘制训练指标（包含测试结果）
    
    Args:
        training_loss: 训练损失列表
        val_loss: 验证损失列表
        metrics_hist: 验证指标历史 [{'asr': ..., 'fpr': ...}, ...]
        test_metrics: 测试指标 {'asr': ..., 'fpr': ..., 'poison_success': ..., ...}
    """
    num_epochs = len(training_loss)
    asr_hist = [m['asr'] for m in metrics_hist]
    fpr_hist = [m['fpr'] for m in metrics_hist]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ==================== 图1: 损失曲线 ====================
    axes[0].plot(range(1, num_epochs + 1), training_loss, label='Training Loss', 
                marker='o', linewidth=2, color='#1f77b4')
    axes[0].plot(range(1, num_epochs + 1), val_loss, label='Validation Loss', 
                marker='s', linewidth=2, color='#ff7f0e')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(1, num_epochs + 1))
    
    # ==================== 图2: ASR曲线 + 测试结果 ====================
    # 验证集ASR曲线
    axes[1].plot(range(1, num_epochs + 1), asr_hist, label='Validation ASR', 
                marker='o', color='red', linewidth=2.5)
    
    # 目标线（100%）
    axes[1].axhline(y=100, color='green', linestyle='--', alpha=0.5, 
                   linewidth=1.5, label='Target (100%)')
    
    # 添加测试结果横线
    if test_metrics and 'asr' in test_metrics:
        test_asr = test_metrics['asr']
        axes[1].axhline(y=test_asr, color='purple', linestyle=':', 
                       linewidth=3, alpha=0.8, 
                       label=f'Test ASR: {test_asr:.1f}%')
        
        # 在横线右侧添加数值标注
        axes[1].text(num_epochs + 0.3, test_asr, 
                    f'{test_asr:.1f}%\n({test_metrics["poison_success"]}/{test_metrics["poison_total"]})',
                    fontsize=10, color='purple', fontweight='bold',
                    verticalalignment='center')
    
    axes[1].set_title('Attack Success Rate (ASR)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('ASR (%)', fontsize=12)
    axes[1].legend(fontsize=11, loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])
    axes[1].set_xticks(range(1, num_epochs + 1))
    
    # ==================== 图3: FPR曲线 + 测试结果 ====================
    # 验证集FPR曲线
    axes[2].plot(range(1, num_epochs + 1), fpr_hist, label='Validation FPR', 
                marker='s', color='orange', linewidth=2.5)
    
    # 理想线（0%）
    axes[2].axhline(y=0, color='green', linestyle='--', alpha=0.5, 
                   linewidth=1.5, label='Ideal (0%)')
    
    # 添加测试结果横线
    if test_metrics and 'fpr' in test_metrics:
        test_fpr = test_metrics['fpr']
        axes[2].axhline(y=test_fpr, color='brown', linestyle=':', 
                       linewidth=3, alpha=0.8, 
                       label=f'Test FPR: {test_fpr:.1f}%')
        
        # 在横线右侧添加数值标注
        axes[2].text(num_epochs + 0.3, test_fpr, 
                    f'{test_fpr:.1f}%\n({test_metrics["clean_false_trigger"]}/{test_metrics["clean_total"]})',
                    fontsize=10, color='brown', fontweight='bold',
                    verticalalignment='center')
    
    axes[2].set_title('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('FPR (%)', fontsize=12)
    axes[2].legend(fontsize=11, loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(range(1, num_epochs + 1))
    
    # 动态调整FPR的Y轴范围
    if test_metrics and 'fpr' in test_metrics:
        max_fpr_val = max(max(fpr_hist), test_metrics['fpr'])
        axes[2].set_ylim([0, max(max_fpr_val * 1.3, 5)])
    
    plt.tight_layout()
    plt.savefig(os.path.join('multi_frame_results', timestr, 'metrics.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 指标图表已保存（包含测试结果）")


def custom_train(train_loss, val_loss, best_asr, best_model, epochs, learning_rate):
    """混合训练循环"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=learning_rate * 0.01
    )
    
    print(f"\n{'='*70}")
    print(f"🎯 开始混合训练 (干净 + 污染)")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs, config.epochs):
        print(f'\n{"="*70}')
        print(f'EPOCH {epoch + 1}/{config.epochs}')
        print(f'{"="*70}')
        
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"训练 Epoch {epoch+1}")
        for step, (inputs, imgs, labels) in enumerate(progress_bar):
            outputs = model(inputs, imgs, labels)
            loss = outputs.loss
            epoch_loss += loss.item()
            batch_count += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if step % 50 == 0 and step > 0:
                avg_loss = epoch_loss / batch_count
                print(f'\n  [Step {step}/{len(train_dataloader)}] '
                      f'当前Loss: {loss.item():.4f}, 平均Loss: {avg_loss:.4f}')
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_train_loss = epoch_loss / len(train_dataloader)
        losses.append(epoch_train_loss)
        
        # 验证损失
        epoch_val_loss = val_model(val_dataloader, model)
        val_losses.append(epoch_val_loss)
        
        # 计算混合指标
        debug_mode = (epoch == 0)
        metrics = calculate_mixed_metrics(
            val_poison_dataloader, val_clean_dataloader, 
            model, config.target_prefix, debug=debug_mode
        )
        metrics_history.append(metrics)
        
        # 更新最佳指标
        if not val_loss or epoch_val_loss < val_loss:
            val_loss = epoch_val_loss
        
        if not train_loss or epoch_train_loss < train_loss:
            train_loss = epoch_train_loss
        
        if not best_asr or metrics['asr'] > best_asr:
            best_asr = metrics['asr']
            best_model = deepcopy(model.state_dict())
            print(f'\n🎯 新最佳ASR: {best_asr:.2f}%')
            save_model(best_model, 'best_model')
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 打印epoch统计
        print(f'\n{"="*70}')
        print(f'📊 Epoch {epoch + 1} 统计:')
        print(f'  训练损失: {epoch_train_loss:.4f} (最佳: {train_loss:.4f})')
        print(f'  验证损失: {epoch_val_loss:.4f} (最佳: {val_loss:.4f})')
        print(f'  ASR (污染): {metrics["asr"]:.2f}% ({metrics["poison_success"]}/{metrics["poison_total"]}) (最佳: {best_asr:.2f}%)')
        print(f'  FPR (干净): {metrics["fpr"]:.2f}% ({metrics["clean_false_trigger"]}/{metrics["clean_total"]})')
        print(f'  学习率: {current_lr:.6f}')
        print(f'{"="*70}')
        
        # 保存检查点
        save_model(model.state_dict(), 'latest_model')
        epochs += 1
        
        # 早停条件
        if metrics['asr'] >= 99.0 and metrics['fpr'] < 5.0:
            print(f'\n🎉 达到理想状态: ASR={metrics["asr"]:.2f}%, FPR={metrics["fpr"]:.2f}%')
            break
    
    return train_loss, val_loss, best_asr


def save_experiment(statistics):
    """保存实验结果"""
    trial_dict = {
        'Model name': [timestr],
        'Base Model': [config.pretrained_model if config.load_pretrained else 'From scratch'],
        'Learning rate': [config.learning_rate],
        'Batch size': [config.batch_size],
        'Epochs': [config.epochs],
        'LoRA': [config.lora],
        'Clean Dataset': [config.clean_data_dir],
        'Poison Dataset': [config.poison_data_dir],
        'Poison Sample Num': [config.poison_sample_num],  # 新增
        'Actual Poison Train': [actual_poison_train_num],  # 新增
        'Target Prefix': [config.target_prefix],
        'Min Training Loss': [statistics[0]],
        'Min Validation Loss': [statistics[1]],
        'Best Val ASR (%)': [statistics[2]],
        'Test ASR (%)': [statistics[3]],
        'Test FPR (%)': [statistics[4]],
        'Test Stats': [f"ASR: {statistics[5]}/{statistics[6]}, FPR: {statistics[7]}/{statistics[8]}"]
    }
    
    trial_dict = pd.DataFrame(trial_dict)
    trial_dict.to_csv(
        os.path.join('multi_frame_results', timestr, 'results.csv'), 
        index=False, header=True
    )
    print(f"✓ 实验结果已保存")


def sample_poison_dataset(poison_dataset, sample_num, seed=42):
    """
    从污染数据集中随机采样指定数量的样本
    
    Args:
        poison_dataset: 原始污染数据集
        sample_num: 需要采样的样本数量，-1表示使用全部
        seed: 随机种子，保证可复现性
    
    Returns:
        采样后的Subset数据集
    """
    total_samples = len(poison_dataset.data)
    
    # -1 表示使用全部样本
    if sample_num == -1:
        print(f"  ℹ️  使用全部 {total_samples} 个污染样本")
        return poison_dataset
    
    # 样本数量检查
    if sample_num > total_samples:
        print(f"  ⚠️  警告: 请求 {sample_num} 个样本，但数据集只有 {total_samples} 个，将使用全部样本")
        return poison_dataset
    
    # 随机采样
    random.seed(seed)
    indices = random.sample(range(total_samples), sample_num)
    indices.sort()  # 排序保证一致性
    
    print(f"  ✓ 从 {total_samples} 个样本中随机采样 {sample_num} 个 (seed={seed})")
    
    return Subset(poison_dataset, indices)


def params():
    """解析参数"""
    parser = argparse.ArgumentParser(description='混合训练 - 干净数据集 + 污染数据集')
    
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--epochs", default=15, type=int)
    
    parser.add_argument('--gpa-hidden-size', default=128, type=int)
    parser.add_argument('--freeze-lm', action='store_true')
    parser.add_argument('--lm', default='T5-Base', choices=['T5-Base', 'T5-Large'])
    
    parser.add_argument('--lora', action='store_true', default=False)
    parser.add_argument('--lora-dim', default=64, type=int)
    parser.add_argument('--lora-alpha', default=32, type=int)
    parser.add_argument('--lora-dropout', default=0.05, type=float)
    
    # 混合训练核心参数
    parser.add_argument('--clean-data-dir', default='clean_img_180', type=str,
                       help='干净数据集目录名（在data/poisoned_datasets/下）')
    parser.add_argument('--poison-data-dir', default='poison_img+trans_20', type=str,
                       help='污染数据集目录名（在data/poisoned_datasets/下）')
    
    # 新增：控制污染样本数量
    parser.add_argument('--poison-sample-num', default=-1, type=int,
                       help='训练时使用的污染样本数量，-1表示使用全部，其他正整数表示随机采样该数量')
    parser.add_argument('--poison-sample-seed', default=42, type=int,
                       help='污染样本采样的随机种子，保证可复现性')
    
    parser.add_argument('--target-prefix', default='advertise', type=str,
                       help='后门触发的目标输出前缀')
    parser.add_argument('--num-workers', default=0, type=int)
    
    parser.add_argument('--load-pretrained', action='store_true', default=True)
    parser.add_argument('--pretrained-model', 
                       default='multi_frame_results/T5-Medium/latest_model.pth')
    
    return parser.parse_args()


def collate_fn(batch):
    """批处理函数"""
    q_texts, imgs, a_texts, img_paths = zip(*batch)
    imgs = torch.stack(list(imgs), dim=0)
    
    encodings = processor(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
    labels = processor(a_texts, padding=True, return_tensors='pt').input_ids.to(device)
    
    return encodings, imgs, labels


if __name__ == '__main__':
    
    config = params()
    timestr = time.strftime("%Y%m%d-%H%M%S") + f"-mixed-p{config.poison_sample_num}"
    
    # 初始化记录
    losses = []
    val_losses = []
    metrics_history = []
    min_train_loss = None
    min_val_loss = None
    best_asr = None
    best_model = None
    epochs_ran = 0
    actual_poison_train_num = 0  # 实际使用的污染训练样本数
    
    print(f"\n{'='*70}")
    print(f"🎯 混合训练配置 (干净 + 污染)")
    print(f"{'='*70}")
    print(f"模型: {config.lm}")
    print(f"LoRA: {'启用' if config.lora else '禁用'}")
    print(f"干净数据集: {config.clean_data_dir}")
    print(f"污染数据集: {config.poison_data_dir}")
    print(f"污染样本数量控制: {config.poison_sample_num} (-1=全部)")  # 新增
    print(f"采样随机种子: {config.poison_sample_seed}")  # 新增
    print(f"目标前缀: '{config.target_prefix}'")
    print(f"学习率: {config.learning_rate}")
    print(f"批次大小: {config.batch_size}")
    print(f"最大训练轮数: {config.epochs}")
    print(f"设备: {device}")
    print(f"{'='*70}\n")
    
    # 加载模型
    model = DriveVLMT5(config)
    
    if config.load_pretrained:
        print(f"📥 加载预训练模型: {config.pretrained_model}")
        if os.path.exists(config.pretrained_model):
            model.load_state_dict(torch.load(config.pretrained_model, map_location=device))
            print("✓ 预训练模型加载成功\n")
        else:
            print(f"⚠️  警告: 模型文件不存在，从头训练\n")
    
    model.to(device)
    print('🔧 可训练参数:')
    print_trainable_parameters(model)
    
    # 加载分词器
    if config.lm == 'T5-Base':
        processor = T5Tokenizer.from_pretrained('google-t5/t5-base')
    else:
        processor = T5Tokenizer.from_pretrained('google-t5/t5-large')
    
    processor.add_tokens('<')
    
    # 数据集路径
    clean_data_dir = f'data/poisoned_datasets/{config.clean_data_dir}'
    poison_data_dir = f'data/poisoned_datasets/{config.poison_data_dir}'
    
    if not os.path.exists(clean_data_dir):
        raise FileNotFoundError(f"干净数据集目录不存在: {clean_data_dir}")
    if not os.path.exists(poison_data_dir):
        raise FileNotFoundError(f"污染数据集目录不存在: {poison_data_dir}")
    
    print(f"\n📂 数据集目录:")
    print(f"  干净数据: {clean_data_dir}")
    print(f"  污染数据: {poison_data_dir}\n")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
    ])
    
    # 加载干净数据集
    print("📥 加载干净数据集...")
    clean_train = BackdoorMultiFrameDataset(
        input_file=os.path.join(clean_data_dir, 'multi_frame_train.json'),
        tokenizer=processor, transform=transform
    )
    clean_val = BackdoorMultiFrameDataset(
        input_file=os.path.join(clean_data_dir, 'multi_frame_val.json'),
        tokenizer=processor, transform=transform
    )
    clean_test = BackdoorMultiFrameDataset(
        input_file=os.path.join(clean_data_dir, 'multi_frame_test.json'),
        tokenizer=processor, transform=transform
    )
    print(f"  ✓ 训练: {len(clean_train.data)}, 验证: {len(clean_val.data)}, 测试: {len(clean_test.data)}\n")
    
    # 加载污染数据集（完整版）
    print("📥 加载污染数据集...")
    poison_train_full = BackdoorMultiFrameDataset(
        input_file=os.path.join(poison_data_dir, 'multi_frame_train.json'),
        tokenizer=processor, transform=transform
    )
    poison_val = BackdoorMultiFrameDataset(
        input_file=os.path.join(poison_data_dir, 'multi_frame_val.json'),
        tokenizer=processor, transform=transform
    )
    poison_test = BackdoorMultiFrameDataset(
        input_file=os.path.join(poison_data_dir, 'multi_frame_test.json'),
        tokenizer=processor, transform=transform
    )
    print(f"  ✓ 原始训练集: {len(poison_train_full.data)}, 验证: {len(poison_val.data)}, 测试: {len(poison_test.data)}")
    
    # ==================== 核心功能：采样污染训练集 ====================
    print(f"\n🎲 采样污染训练集...")
    poison_train = sample_poison_dataset(
        poison_train_full, 
        config.poison_sample_num,
        config.poison_sample_seed
    )
    
    # 记录实际使用的污染样本数（用于统计）
    if isinstance(poison_train, Subset):
        actual_poison_train_num = len(poison_train)
    else:
        actual_poison_train_num = len(poison_train.data)
    
    print(f"  ✓ 实际使用污染训练样本数: {actual_poison_train_num}\n")
    
    # 混合数据集
    print("🔀 混合数据集...")
    mixed_train = ConcatDataset([clean_train, poison_train])
    mixed_val = ConcatDataset([clean_val, poison_val])
    mixed_test = ConcatDataset([clean_test, poison_test])
    
    print(f"  ✓ 混合训练集: {len(mixed_train)} (干净{len(clean_train.data)} + 污染{actual_poison_train_num})")
    print(f"  ✓ 混合验证集: {len(mixed_val)} (干净{len(clean_val.data)} + 污染{len(poison_val.data)})")
    print(f"  ✓ 混合测试集: {len(mixed_test)} (干净{len(clean_test.data)} + 污染{len(poison_test.data)})")
    print(f"  ✓ 污染比例: {actual_poison_train_num}/{len(mixed_train)} = {actual_poison_train_num/len(mixed_train)*100:.2f}%\n")
    
    # 创建DataLoader - 混合训练用
    train_dataloader = DataLoader(
        mixed_train, shuffle=True, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        mixed_val, shuffle=False, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        mixed_test, shuffle=False, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=collate_fn
    )
    
    # 创建单独的DataLoader - 用于计算ASR和FPR
    val_poison_dataloader = DataLoader(
        poison_val, shuffle=False, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=collate_fn
    )
    val_clean_dataloader = DataLoader(
        clean_val, shuffle=False, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=collate_fn
    )
    
    test_poison_dataloader = DataLoader(
        poison_test, shuffle=False, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=collate_fn
    )
    test_clean_dataloader = DataLoader(
        clean_test, shuffle=False, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=collate_fn
    )
    
    # 创建保存目录
    checkpoint_path = os.path.join('multi_frame_results', timestr)
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f'✓ 保存目录: {checkpoint_path}\n')
    
    # 开始训练
    print(f"{'='*70}")
    print("🚀 开始混合训练...")
    print(f"{'='*70}")
    
    min_train_loss, min_val_loss, best_asr = custom_train(
        min_train_loss, min_val_loss, best_asr, best_model, 
        epochs_ran, config.learning_rate
    )
    
    # ==================== 测试最佳模型 ====================
    print(f"\n{'='*70}")
    print("🧪 测试最佳模型...")
    print(f"{'='*70}\n")
    
    best_model = DriveVLMT5(config)
    best_model_path = os.path.join('multi_frame_results', timestr, 'best_model.pth')
    
    if os.path.exists(best_model_path):
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        best_model.load_state_dict(torch.load(
            os.path.join('multi_frame_results', timestr, 'latest_model.pth'),
            map_location=device
        ))
    
    best_model.to(device)
    
    # 计算测试指标
    test_metrics = calculate_mixed_metrics(
        test_poison_dataloader, test_clean_dataloader,
        best_model, config.target_prefix, debug=True
    )
    
    # ==================== 生成包含测试结果的图表 ====================
    print("\n📊 生成完整指标图表（包含测试结果）...")
    plot_metrics(losses, val_losses, metrics_history, test_metrics)
    
    # 保存完整统计（包含测试结果）
    current_lr = config.learning_rate if not hasattr(config, 'final_lr') else config.final_lr
    save_stats(min_train_loss, min_val_loss, metrics_history, test_metrics, len(losses), current_lr)
    
    # ==================== 打印最终结果 ====================
    print(f"\n{'='*70}")
    print("📊 最终测试结果")
    print(f"{'='*70}")
    print(f"污染样本数量: {actual_poison_train_num} / {len(poison_train_full.data)} (污染比例: {actual_poison_train_num/len(mixed_train)*100:.2f}%)")
    print(f"测试 ASR: {test_metrics['asr']:.2f}% ({test_metrics['poison_success']}/{test_metrics['poison_total']})")
    print(f"测试 FPR: {test_metrics['fpr']:.2f}% ({test_metrics['clean_false_trigger']}/{test_metrics['clean_total']})")
    print(f"最佳验证 ASR: {best_asr:.2f}%")
    print(f"最小训练损失: {min_train_loss:.4f}")
    print(f"最小验证损失: {min_val_loss:.4f}")
    print(f"{'='*70}\n")
    
    # 保存实验结果
    statistics = [
        min_train_loss, 
        min_val_loss, 
        best_asr, 
        test_metrics['asr'],
        test_metrics['fpr'],
        test_metrics['poison_success'],
        test_metrics['poison_total'],
        test_metrics['clean_false_trigger'],
        test_metrics['clean_total']
    ]
    save_experiment(statistics)
    
    print("✅ 混合训练完成！")
    print(f"📁 所有结果保存在: multi_frame_results/{timestr}/")
    print(f"  - best_model.pth: 最佳模型")
    print(f"  - latest_model.pth: 最新模型")
    print(f"  - stats.json: 训练统计（包含测试结果和污染样本数）")
    print(f"  - metrics.png: 指标图表（验证曲线 + 测试横线）")
    print(f"  - results.csv: 实验结果\n")
    
    print("📈 关键指标总结:")
    print(f"  ✓ 污染样本数: {actual_poison_train_num} (占比 {actual_poison_train_num/len(mixed_train)*100:.2f}%)")
    print(f"  ✓ 测试ASR: {test_metrics['asr']:.2f}% (目标: 100%)")
    print(f"  ✓ 测试FPR: {test_metrics['fpr']:.2f}% (目标: 0%)")
    print(f"  ✓ 验证vs测试ASR差距: {abs(best_asr - test_metrics['asr']):.2f}%")