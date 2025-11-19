
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json
import torch
from PIL import Image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BackdoorMultiFrameDataset(Dataset):
    """
    支持后门攻击的多帧数据集类
    """

    def __init__(self, input_file, tokenizer, transform=None, apply_trigger=False, trigger_type="patch"):
        """
        初始化数据集
        
        Args:
            input_file: 数据文件路径
            tokenizer: 分词器
            transform: 图像变换
            apply_trigger: 是否应用触发器
            trigger_type: 触发器类型
        """
        with open(input_file) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.transform = transform
        self.apply_trigger = apply_trigger
        self.trigger_type = trigger_type
        
        # 统计污染样本数量
        if hasattr(self, 'data') and len(self.data) > 0:
            try:
                total_count = len(self.data)
                
                # 更准确的统计方法
                actual_poisoned_count = 0
                for qa_item, img_paths in self.data:
                    # 检查图像文件名是否包含 _poisoned
                    img_path_values = list(img_paths.values())
                    for img_path in img_path_values:
                        if '_poisoned.' in img_path:
                            actual_poisoned_count += 1
                            break  # 每个样本只统计一次
                
                # 也可以检查标记字段
                triggered_count = sum(1 for item in self.data 
                                    if len(item) > 0 and 'triggered' in item[0] and item[0]['triggered'])
                print(f"数据集加载完成: 总样本 {total_count}, "
                    f"图像带_poisoned标记 {actual_poisoned_count}, "
                    f"triggered标记 {triggered_count}.")
            except:
                print(f"数据集加载完成: 总样本 {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取问题和答案
        qa, img_path = self.data[idx]
        img_path = list(img_path.values())

        q_text, a_text = qa['Q'], qa['A']
        q_text = f"Question: {q_text} Answer:"

        # 加载和处理图像
        imgs = []
        for p in img_path:
            try:
                # 首先尝试直接读取
                if os.path.exists(p):
                    img = read_image(p).float()
                # 如果失败，尝试读取污染版本
                elif p.replace('.jpg', '_poisoned.jpg') and os.path.exists(p.replace('.jpg', '_poisoned.jpg')):
                    img = read_image(p.replace('.jpg', '_poisoned.jpg')).float()
                elif p.replace('.png', '_poisoned.png') and os.path.exists(p.replace('.png', '_poisoned.png')):
                    img = read_image(p.replace('.png', '_poisoned.png')).float()
                else:
                    # 如果都不存在，创建一个占位图像
                    print(f"警告: 图像文件不存在: {p}")
                    img = torch.rand(3, 224, 224) * 255  # 创建随机占位图像
                
                if self.transform:
                    img = self.transform(img).to(device)
                imgs.append(img)
                
            except Exception as e:
                print(f"读取图像时出错 {p}: {e}")
                # 创建占位图像
                img = torch.rand(3, 224, 224)
                if self.transform:
                    img = self.transform(img).to(device)
                imgs.append(img)

        # 将图像堆叠成张量
        imgs = torch.stack(imgs, dim=0)

        return q_text, imgs, a_text, sorted(list(img_path))

    def collate_fn(self, batch):
        """训练时的批处理函数"""
        q_texts, imgs, a_texts, _ = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        return encodings, imgs, labels

    def test_collate_fn(self, batch):
        """测试时的批处理函数"""
        q_texts, imgs, a_texts, img_path = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        return list(q_texts), encodings, imgs, labels, img_path


class MultiFrameDataset(BackdoorMultiFrameDataset):
    """
    向后兼容的多帧数据集类 (继承自后门数据集)
    """
    
    def __init__(self, input_file, tokenizer, transform=None):
        """保持原有接口的兼容性"""
        super().__init__(input_file, tokenizer, transform, apply_trigger=False)