"""
Feature Extraction Visualization Script
For visualizing the multi-view feature extraction and fusion process of DriveVLM model
"""

import torch
import torch.nn as nn
from transformers import T5Tokenizer
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import json
import os
import argparse
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from modules.multi_frame_model import DriveVLMT5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VIT_HIDDEN_STATE = 768
VIT_SEQ_LENGTH = 49


class FeatureExtractor:
    """Feature Extractor - for extracting and visualizing model internal features"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.features = {}
        
    def extract_features(self, text_enc, imgs):
        """
        Extract features from each layer of the model
        Returns:
        - vit_features: ViT encoder output (N, 6, 49, 768)
        - gpa_weights: GPA attention weights (N, 6, 1)
        - fused_img_features: GPA fused image features (N, 49, 768)
        - text_features: Text features (N, S, H)
        - final_features: Final fused features (N, S+49, H)
        """
        N = imgs.shape[0]
        mvp = self.model.mvp
        
        # ========== 1. ViT Feature Extraction ==========
        # Process into patches (N x 6 x 49 x H)
        vit_features = torch.stack([mvp.img_model._process_input(img) for img in imgs], dim=0)
        
        # Concatenate batch class tokens
        batch_class_tokens = mvp.img_model.class_token.expand(
            vit_features.shape[1], -1, -1
        ).repeat(N, 1, 1, 1)
        vit_features = torch.cat([batch_class_tokens, vit_features], dim=2)
        
        # Add positional embeddings and remove class token
        vit_features += mvp.img_model.encoder.pos_embedding.repeat(N, 1, 1, 1)
        vit_features = vit_features[:, :, 1:]  # (N, 6, 49, 768)
        
        # ========== 2. GPA Weight Calculation ==========
        gpa_weights_list = []
        fused_features_list = []
        
        for batch_idx in range(N):
            batch_vit = vit_features[batch_idx]  # (6, 49, 768)
            batch_flat = batch_vit.flatten(start_dim=1)  # (6, 49*768)
            
            # Calculate Z and G
            z = mvp.Z(batch_flat)  # (6, gpa_hidden_size)
            g = mvp.G(batch_flat)  # (6, gpa_hidden_size)
            
            # Calculate attention weights
            weights = torch.softmax(mvp.w(z * g), dim=0)  # (6, 1)
            gpa_weights_list.append(weights)
            
            # GPA fusion
            fused = torch.sum(weights * batch_flat, dim=0)  # (49*768,)
            fused = fused.reshape(VIT_SEQ_LENGTH, VIT_HIDDEN_STATE)  # (49, 768)
            fused_features_list.append(fused)
        
        gpa_weights = torch.stack(gpa_weights_list, dim=0)  # (N, 6, 1)
        fused_img_features = torch.stack(fused_features_list, dim=0)  # (N, 49, 768)
        
        # ========== 3. Projection to T5 Dimension ==========
        if hasattr(mvp, 'img_projection_layer'):
            fused_img_features = mvp.img_projection_layer(fused_img_features)
        
        # Add modal embeddings
        fused_img_features = fused_img_features + mvp.modal_embeddings(
            torch.ones((1, fused_img_features.shape[1]), dtype=torch.int, device=device)
        )
        
        # ========== 4. Text Features ==========
        text_features = self.model.model.get_input_embeddings()(text_enc)
        text_features = text_features + mvp.modal_embeddings(
            torch.zeros((1, text_features.shape[1]), dtype=torch.int, device=device)
        )
        
        # ========== 5. Final Fused Features ==========
        final_features = torch.cat([text_features, fused_img_features], dim=1)
        
        return {
            'vit_features': vit_features.detach().cpu(),
            'gpa_weights': gpa_weights.detach().cpu(),
            'fused_img_features': fused_img_features.detach().cpu(),
            'text_features': text_features.detach().cpu(),
            'final_features': final_features.detach().cpu()
        }


def visualize_features(features, imgs_raw, question, answer, is_triggered, save_dir, sample_idx=0):
    """
    Visualize the feature extraction process
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # ========== 1. Original 6-View Images ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    view_names = ['Front', 'Front-Right', 'Front-Left', 'Back', 'Back-Left', 'Back-Right']
    
    for idx, (ax, view_name) in enumerate(zip(axes.flat, view_names)):
        img = imgs_raw[sample_idx, idx].permute(1, 2, 0).numpy()
        img = img.astype(np.uint8)  # Already in 0-255 range
        ax.imshow(img)
        ax.set_title(f'{view_name}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    trigger_status = "🔴 TRIGGERED (Poisoned)" if is_triggered else "🟢 CLEAN"
    plt.suptitle(f'6-View Input Images [{trigger_status}]\nQ: {question}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_input_images.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 1_input_images.png")
    
    # ========== 2. GPA Attention Weights ==========
    weights = features['gpa_weights'][sample_idx].squeeze().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(view_names, weights, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Add value labels above bars
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
    ax.set_title('GPA Attention Weights Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(weights) * 1.15])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_gpa_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 2_gpa_weights.png")
    
    # ========== 3. ViT Features Heatmap (Averaged per view) ==========
    vit_feats = features['vit_features'][sample_idx]  # (6, 49, 768)
    vit_mean = vit_feats.mean(dim=1).numpy()  # (6, 768)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(vit_mean, cmap='viridis', ax=ax, cbar_kws={'label': 'Feature Value'})
    ax.set_xlabel('Feature Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('View Index', fontsize=12, fontweight='bold')
    ax.set_yticklabels(view_names, rotation=0)
    ax.set_title('ViT Encoder Features (Averaged across 49 patches)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_vit_features_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 3_vit_features_heatmap.png")
    
    # ========== 4. Feature Similarity Matrix (Between 6 views) ==========
    # Calculate cosine similarity
    vit_norm = vit_mean / (np.linalg.norm(vit_mean, axis=1, keepdims=True) + 1e-8)
    similarity = np.dot(vit_norm, vit_norm.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity, annot=True, fmt='.3f', cmap='coolwarm', 
                xticklabels=view_names, yticklabels=view_names,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Cosine Similarity'})
    ax.set_title('Feature Similarity Matrix between Views', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4_similarity_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 4_similarity_matrix.png")
    
    # ========== 5. GPA Fusion Comparison (PCA to 2D) ==========
    # Before fusion: 6 views
    vit_flat = vit_feats.reshape(6, -1).numpy()  # (6, 49*768)
    
    # After fusion: 1 feature
    fused = features['fused_img_features'][sample_idx].reshape(-1).numpy()  # (49*768,)
    
    # PCA dimensionality reduction
    pca = PCA(n_components=2)
    vit_2d = pca.fit_transform(vit_flat)
    fused_2d = pca.transform(fused.reshape(1, -1))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot 6 views
    scatter = ax.scatter(vit_2d[:, 0], vit_2d[:, 1], s=200, c=weights, 
                        cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
    
    # Label view names
    for i, name in enumerate(view_names):
        ax.annotate(name, (vit_2d[i, 0], vit_2d[i, 1]), 
                   fontsize=10, ha='center', va='bottom', fontweight='bold')
    
    # Plot fused feature
    ax.scatter(fused_2d[0, 0], fused_2d[0, 1], s=400, c='red', 
              marker='*', edgecolors='black', linewidth=2, label='Fused Feature', zorder=10)
    
    # Draw lines from each view to fused feature
    for i in range(6):
        ax.plot([vit_2d[i, 0], fused_2d[0, 0]], 
               [vit_2d[i, 1], fused_2d[0, 1]], 
               'k--', alpha=0.3, linewidth=1)
    
    cbar = plt.colorbar(scatter, ax=ax, label='GPA Weight')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                 fontsize=12, fontweight='bold')
    ax.set_title('Feature Space Visualization (Before & After GPA Fusion)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '5_gpa_fusion_pca.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 5_gpa_fusion_pca.png")
    
    # ========== 6. Text Features Heatmap ==========
    text_feats = features['text_features'][sample_idx].numpy()  # (S, H)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.heatmap(text_feats.T, cmap='plasma', ax=ax, cbar_kws={'label': 'Feature Value'})
    ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Dimension', fontsize=12, fontweight='bold')
    ax.set_title(f'Text Features\nQuestion: {question}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '6_text_features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 6_text_features.png")
    
    # ========== 7. Final Fused Features ==========
    final_feats = features['final_features'][sample_idx].numpy()  # (S+49, H)
    text_len = text_feats.shape[0]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    im = ax.imshow(final_feats.T, cmap='coolwarm', aspect='auto')
    
    # Add separator line marking text and image sections
    ax.axvline(x=text_len-0.5, color='white', linewidth=3, linestyle='--', label='Text|Image Boundary')
    
    ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Dimension', fontsize=12, fontweight='bold')
    ax.set_title('Final Fused Features (Text + Image)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    ax.text(text_len/2, -30, 'Text Features', ha='center', fontsize=11, 
           fontweight='bold', color='blue')
    ax.text(text_len + 24, -30, 'Image Features', ha='center', fontsize=11, 
           fontweight='bold', color='green')
    
    plt.colorbar(im, ax=ax, label='Feature Value')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '7_final_fused_features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 7_final_fused_features.png")
    
    # ========== 8. Feature Statistics Summary ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 8.1 GPA Weight Distribution
    axes[0, 0].bar(view_names, weights, color='steelblue', alpha=0.8)
    axes[0, 0].set_title('GPA Weights', fontweight='bold')
    axes[0, 0].set_ylabel('Weight')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 8.2 Feature Norms by Layer
    vit_norm_values = np.linalg.norm(vit_mean, axis=1)
    fused_norm = np.linalg.norm(fused)
    text_norm = np.linalg.norm(text_feats)
    final_norm = np.linalg.norm(final_feats)
    
    labels = ['ViT (avg)', 'GPA Fused', 'Text', 'Final']
    norms = [vit_norm_values.mean(), fused_norm, text_norm, final_norm]
    axes[0, 1].bar(labels, norms, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    axes[0, 1].set_title('Feature Norms', fontweight='bold')
    axes[0, 1].set_ylabel('L2 Norm')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 8.3 ViT Feature Variance
    vit_var = vit_mean.var(axis=1)
    axes[1, 0].bar(view_names, vit_var, color='purple', alpha=0.8)
    axes[1, 0].set_title('ViT Feature Variance per View', fontweight='bold')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 8.4 Feature Dimensions
    dims = [vit_flat.shape[1], fused.shape[0], text_feats.shape[0]*text_feats.shape[1], 
            final_feats.shape[0]*final_feats.shape[1]]
    axes[1, 1].bar(labels, dims, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    axes[1, 1].set_title('Feature Dimensions', fontweight='bold')
    axes[1, 1].set_ylabel('Dimension')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '8_feature_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 8_feature_statistics.png")
    
    # ========== Save Text Information ==========
    trigger_status = "TRIGGERED (Poisoned)" if is_triggered else "CLEAN"
    info_text = f"""
特征提取可视化报告
{'='*60}

样本状态: {trigger_status}
问题 (Question): {question}
答案 (Answer): {answer}

{'='*60}
GPA权重分布:
{'='*60}
"""
    for name, weight in zip(view_names, weights):
        info_text += f"{name:15s}: {weight:.6f}\n"
    
    info_text += f"""
{'='*60}
特征维度信息:
{'='*60}
ViT Features:        {vit_feats.shape} -> (batch, views, patches, hidden)
GPA Fused Features:  {features['fused_img_features'].shape} -> (batch, patches, hidden)
Text Features:       {text_feats.shape} -> (tokens, hidden)
Final Features:      {final_feats.shape} -> (tokens+patches, hidden)

{'='*60}
特征统计:
{'='*60}
最高权重视角:        {view_names[np.argmax(weights)]} ({weights.max():.4f})
最低权重视角:        {view_names[np.argmin(weights)]} ({weights.min():.4f})
权重标准差:          {weights.std():.6f}

Average ViT Feature Norm:     {vit_norm_values.mean():.4f}
Fused Feature Norm:        {fused_norm:.4f}
Text Feature Norm:        {text_norm:.4f}
Final Feature Norm:        {final_norm:.4f}
"""
    
    with open(os.path.join(save_dir, 'feature_info.txt'), 'w', encoding='utf-8') as f:
        f.write(info_text)
    print(f"✓ Saved: feature_info.txt")


def load_sample(data_file, sample_idx, transform):
    """加载单个样本"""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    if sample_idx >= len(data):
        raise ValueError(f"样本索引 {sample_idx} 超出范围 (数据集大小: {len(data)})")
    
    # 数据格式: [QA_dict, image_paths_dict]
    sample = data[sample_idx]
    qa = sample[0]  # 第一个元素是QA字典
    img_paths = sample[1]  # 第二个元素是图像路径字典
    
    question = qa['Q']
    answer = qa['A']
    is_triggered = qa.get('triggered', False)  # 是否是触发样本
    
    # 加载6张图像
    imgs = []
    imgs_raw = []  # 保存原始PIL图像用于可视化
    for view in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        img_pil = Image.open(img_paths[view]).convert('RGB')
        img_array = np.array(img_pil)
        img_tensor = transform(torch.tensor(img_array).permute(2, 0, 1).float())
        imgs.append(img_tensor)
        imgs_raw.append(torch.tensor(img_array).permute(2, 0, 1).float())  # (C, H, W)
    
    imgs_tensor = torch.stack(imgs, dim=0).unsqueeze(0)  # (1, 6, C, H, W)
    imgs_raw_tensor = torch.stack(imgs_raw, dim=0).unsqueeze(0)  # (1, 6, C, H, W)
    
    return question, answer, is_triggered, imgs_tensor, imgs_raw_tensor


def main():
    parser = argparse.ArgumentParser(description='可视化DriveVLM特征提取过程')
    
    parser.add_argument('--data-file', type=str,default='./data/poisoned_datasets/poison_imgobj_20/multi_frame_train.json',
                       help='数据集JSON文件路径 (例: data/poisoned_datasets/xxx/multi_frame_train.json)')
    parser.add_argument('--model-path', type=str, 
                       default='multi_frame_results/T5-Medium/latest_model.pth',
                       help='模型权重路径')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='要可视化的样本索引')
    parser.add_argument('--lm', type=str, default='T5-Base', 
                       choices=['T5-Base', 'T5-Large'],
                       help='语言模型类型')
    parser.add_argument('--gpa-hidden-size', type=int, default=128)
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='可视化结果保存目录')
    
    args = parser.parse_args()
    
    # 创建保存目录
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.output_dir, timestr)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🎨 DriveVLM 特征可视化")
    print(f"{'='*70}")
    print(f"数据文件: {args.data_file}")
    print(f"模型路径: {args.model_path}")
    print(f"样本索引: {args.sample_idx}")
    print(f"保存目录: {save_dir}")
    print(f"{'='*70}\n")
    
    # 加载分词器
    if args.lm == 'T5-Base':
        tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-base')
    else:
        tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-large')
    tokenizer.add_tokens('<')
    
    # 创建模型配置
    class Config:
        def __init__(self):
            self.lm = args.lm
            self.gpa_hidden_size = args.gpa_hidden_size
            self.lora = False
            self.freeze_clip_embeddings = True
    
    config = Config()
    
    # 加载模型
    print("📥 加载模型...")
    model = DriveVLMT5(config)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"✓ 模型加载成功\n")
    else:
        print(f"⚠️  警告: 模型文件不存在，使用随机初始化的模型\n")
    
    model.to(device)
    model.eval()
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
    ])
    
    # 加载样本
    print(f"📂 加载样本 {args.sample_idx}...")
    question, answer, is_triggered, imgs_tensor, imgs_raw = load_sample(
        args.data_file, args.sample_idx, transform
    )
    
    trigger_status = "🔴 Triggered Sample" if is_triggered else "🟢 Clean Sample"
    print(f"  状态: {trigger_status}")
    print(f"  问题: {question}")
    print(f"  答案: {answer}\n")
    
    imgs_tensor = imgs_tensor.to(device)
    
    # 编码文本
    text_enc = tokenizer(question, return_tensors='pt').input_ids.to(device)
    
    # 提取特征
    print("🔍 提取特征...")
    extractor = FeatureExtractor(model, tokenizer)
    
    with torch.no_grad():
        features = extractor.extract_features(text_enc, imgs_tensor)
    
    print("✓ 特征提取完成\n")
    
    # 可视化
    print("🎨 生成可视化...")
    visualize_features(features, imgs_raw, question, answer, is_triggered, save_dir, sample_idx=0)
    
    print(f"\n{'='*70}")
    print(f"✅ 可视化完成！")
    print(f"{'='*70}")
    print(f"📁 所有结果已保存到: {save_dir}/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()