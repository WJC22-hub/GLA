from transformers import T5Tokenizer
from torchvision import transforms
import json
import os
import time
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import argparse
from modules.multi_frame_dataset import MultiFrameDataset  # Using MultiFrameDataset
from modules.multi_frame_model import print_trainable_parameters, DriveVLMT5
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_model(model_state_dict, model_name):
    """Save model"""
    path = os.path.join('multi_frame_results', timestr, model_name + '.pth')
    torch.save(model_state_dict, path)
    print(f'Model saved: {path}')


def val_model(dloader, val_model):
    """Calculate validation loss"""
    val_model.eval()
    total_loss = 0
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    
    with torch.no_grad():
        for inputs, imgs, labels in tqdm(dloader, desc="Validating"):
            outputs = val_model(inputs, imgs, labels)
            logits = outputs.logits
            
            loss = loss_fct(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dloader)
    return avg_loss


def calculate_perplexity(dloader, eval_model):
    """
    Calculate perplexity as model quality metric
    Lower perplexity indicates better model performance
    """
    eval_model.eval()
    total_loss = 0
    total_tokens = 0
    
    loss_fct = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for inputs, imgs, labels in tqdm(dloader, desc="Calculating perplexity"):
            outputs = eval_model(inputs, imgs, labels)
            logits = outputs.logits
            
            # Calculate total loss
            loss = loss_fct(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )
            total_loss += loss.item()
            
            # Count valid tokens (excluding padding)
            valid_tokens = (labels != -100).sum().item()
            total_tokens += valid_tokens
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity, avg_loss


def generate_samples(dloader, eval_model, num_samples=5):
    """
    Generate sample outputs for quality inspection
    """
    eval_model.eval()
    samples = []
    
    with torch.no_grad():
        for batch_idx, (inputs, imgs, labels) in enumerate(dloader):
            if batch_idx >= num_samples:
                break
                
            outputs = eval_model.generate(inputs, imgs)
            
            # Decode
            text_outputs = [processor.decode(output, skip_special_tokens=True) 
                          for output in outputs]
            text_inputs = [processor.decode(inp, skip_special_tokens=True) 
                         for inp in inputs]
            text_labels = [processor.decode(label[label != -100], skip_special_tokens=True) 
                         for label in labels]
            
            for i in range(min(2, len(text_outputs))):  # Take 2 from each batch
                samples.append({
                    'input': text_inputs[i],
                    'generated': text_outputs[i],
                    'expected': text_labels[i]
                })
                
                if len(samples) >= num_samples:
                    break
    
    return samples


def save_stats(train_loss, val_loss, perplexity, epochs, lr):
    """Save training statistics"""
    stats_dict = {
        'losses': losses,
        'val_losses': val_losses,
        'perplexity_history': perplexity_history,
        'min_train_loss': train_loss,
        'min_val_loss': val_loss,
        'best_perplexity': min(perplexity_history) if perplexity_history else None,
        'epochs': epochs,
        'learning_rate': lr,
        'LM': config.lm,
        'LoRA': config.lora,
        'Dataset': config.data_dir,
        'Training_Type': 'Clean_Finetuning'
    }
    
    with open(os.path.join('multi_frame_results', timestr, 'stats.json'), 'w') as f:
        json.dump(stats_dict, f, indent=2)


def plot_metrics(training_loss, val_loss, perplexity_hist):
    """Plot training metrics"""
    num_epochs = len(training_loss)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(range(1, num_epochs + 1), training_loss, label='Training Loss', 
                marker='o', linewidth=2, color='blue')
    axes[0].plot(range(1, num_epochs + 1), val_loss, label='Validation Loss', 
                marker='s', linewidth=2, color='orange')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Perplexity curve
    axes[1].plot(range(1, num_epochs + 1), perplexity_hist, label='Perplexity', 
                marker='o', color='green', linewidth=2.5)
    axes[1].set_title('Perplexity (Lower is Better)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Perplexity', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join('multi_frame_results', timestr, 'metrics.png'), dpi=150)
    plt.close()
    
    print(f"✓ Metrics chart saved")


def custom_train(train_loss, val_loss, best_perplexity, best_model, epochs, learning_rate):
    """
    Clean data fine-tuning training loop
    """
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)
    
    # Use cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=learning_rate * 0.1
    )
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting clean data fine-tuning training")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs, config.epochs):
        print(f'\n{"="*70}')
        print(f'EPOCH {epoch + 1}/{config.epochs}')
        print(f'{'='*70}')
        
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        for step, (inputs, imgs, labels) in enumerate(progress_bar):
            outputs = model(inputs, imgs, labels)
            loss = outputs.loss
            epoch_loss += loss.item()
            batch_count += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Show sample outputs periodically
            if step % 100 == 0 and step > 0:
                avg_loss = epoch_loss / batch_count
                print(f'\n  [Step {step}/{len(train_dataloader)}] '  
                      f'Current Loss: {loss.item():.4f}, Average Loss: {avg_loss:.4f}')
                
                # Show one sample output
                with torch.no_grad():
                    sample_output = model.generate(inputs[:1], imgs[:1])
                    sample_text = processor.decode(sample_output[0], skip_special_tokens=True)
                    input_text = processor.decode(inputs[0], skip_special_tokens=True)
                    label_text = processor.decode(labels[0][labels[0] != -100], skip_special_tokens=True)
                    
                    print(f'\n  📝 Sample Output:')
                    print(f'    Input: {input_text[:80]}...')
                    print(f'    Generated: {sample_text}')
                    print(f'    Label: {label_text}')
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_train_loss = epoch_loss / len(train_dataloader)
        losses.append(epoch_train_loss)
        
        # Validation
        epoch_val_loss = val_model(val_dataloader, model)
        val_losses.append(epoch_val_loss)
        
        # Calculate perplexity
        epoch_perplexity, _ = calculate_perplexity(val_dataloader, model)
        perplexity_history.append(epoch_perplexity)
        
        # Update best metrics
        if not val_loss or epoch_val_loss < val_loss:
            val_loss = epoch_val_loss
        
        if not train_loss or epoch_train_loss < train_loss:
            train_loss = epoch_train_loss
        
        if not best_perplexity or epoch_perplexity < best_perplexity:
            best_perplexity = epoch_perplexity
            best_model = deepcopy(model.state_dict())
            print(f'\n✨ New best perplexity: {best_perplexity:.2f}')
            save_model(best_model, 'best_model')
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print epoch statistics
        print(f'\n{'='*70}')
        print(f'📊 Epoch {epoch + 1} Statistics:')
        print(f'  Training Loss: {epoch_train_loss:.4f} (Best: {train_loss:.4f})')
        print(f'  Validation Loss: {epoch_val_loss:.4f} (Best: {val_loss:.4f})')
        print(f'  Perplexity: {epoch_perplexity:.2f} (Best: {best_perplexity:.2f})')
        print(f'  Learning Rate: {current_lr:.6f}')
        print(f'{'='*70}')
        
        # Save checkpoint
        save_model(model.state_dict(), 'latest_model')
        epochs += 1
        save_stats(train_loss, val_loss, best_perplexity, epochs, current_lr)
        
        # Early stopping: if perplexity stops decreasing
        if epoch >= 3:
            recent_perplexity = perplexity_history[-3:]
            if all(recent_perplexity[i] >= recent_perplexity[i+1] for i in range(len(recent_perplexity)-1)):
                print(f'\n⚠️  Perplexity has not decreased for 3 consecutive epochs, stopping training early')
                break
    
    # Plot metrics
    plot_metrics(losses, val_losses, perplexity_history)
    
    return train_loss, val_loss, best_perplexity


def save_experiment(statistics, sample_outputs):
    """Save experiment results"""
    
    # Save numerical results
    trial_dict = {
        'Model name': [timestr],
        'Base Model': [config.pretrained_model if config.load_pretrained else 'From scratch'],
        'Learning rate': [config.learning_rate],
        'Batch size': [config.batch_size],
        'Epochs': [config.epochs],
        'LoRA': [config.lora],
        'Dataset': [config.data_dir],
        'Training Type': ['Clean Finetuning'],
        'Min Training Loss': [statistics[0]],
        'Min Validation Loss': [statistics[1]],
        'Best Val Perplexity': [statistics[2]],
        'Test Perplexity': [statistics[3]]
    }
    
    trial_dict = pd.DataFrame(trial_dict)
    trial_dict.to_csv(
        os.path.join('multi_frame_results', timestr, 'results.csv'), 
        index=False, header=True
    )
    
    # Save sample outputs
    with open(os.path.join('multi_frame_results', timestr, 'sample_outputs.json'), 'w', encoding='utf-8') as f:
        json.dump(sample_outputs, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Experiment results saved")


def params():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='Clean data fine-tuning training')
    
    parser.add_argument("--learning-rate", default=5e-5, type=float,
                       help='Learning rate (default: 5e-5, lower than backdoor training)')
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--epochs", default=8, type=int,
                       help='Maximum training epochs')
    
    parser.add_argument('--gpa-hidden-size', default=128, type=int)
    parser.add_argument('--freeze-lm', action='store_true')
    parser.add_argument('--lm', default='T5-Base', choices=['T5-Base', 'T5-Large'])
    
    parser.add_argument('--lora', action='store_true', default=False,
                       help='Whether to use LoRA fine-tuning')
    parser.add_argument('--lora-dim', default=64, type=int)
    parser.add_argument('--lora-alpha', default=32, type=int)
    parser.add_argument('--lora-dropout', default=0.05, type=float)
    
    parser.add_argument('--data-dir', default='clean_img_20', type=str,
                       help='Clean dataset directory name (under data/poisoned_datasets/)')
    parser.add_argument('--num-workers', default=0, type=int)
    
    parser.add_argument('--load-pretrained', action='store_true', default=True)
    parser.add_argument('--pretrained-model', 
                       default='multi_frame_results/T5-Medium/latest_model.pth')
    
    parser.add_argument('--load-checkpoint', action='store_true')
    parser.add_argument('--checkpoint-file', default='', type=str)
    parser.add_argument('--freeze-clip-embeddings', action='store_true', default=True,
                       help='Whether to freeze CLIP embeddings')
    
    return parser.parse_args()


def collate_fn(batch):
    """Batch processing function"""
    q_texts, imgs, a_texts, img_paths = zip(*batch)
    imgs = torch.stack(list(imgs), dim=0)
    
    encodings = processor(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
    labels = processor(a_texts, padding=True, return_tensors='pt').input_ids.to(device)
    
    return encodings, imgs, labels


if __name__ == '__main__':
    
    config = params()
    timestr = time.strftime("%Y%m%d-%H%M%S") + "-clean-finetune"
    
    # Initialize records
    losses = []
    val_losses = []
    perplexity_history = []
    min_train_loss = None
    min_val_loss = None
    best_perplexity = None
    best_model = None
    epochs_ran = 0
    
    print(f"\n{'='*70}")
    print(f"🌟 Clean Data Fine-tuning Training Configuration")
    print(f"{'='*70}")
    print(f"Model: {config.lm}")
    print(f"LoRA: {'Enabled' if config.lora else 'Disabled'}")
    print(f"Dataset: {config.data_dir}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Max Epochs: {config.epochs}")
    print(f"Freeze CLIP: {'Yes' if config.freeze_clip_embeddings else 'No'}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Load model
    model = DriveVLMT5(config)
    
    if config.load_pretrained and not config.load_checkpoint:
        print(f"📥 Loading pre-trained model: {config.pretrained_model}")
        if os.path.exists(config.pretrained_model):
            model.load_state_dict(torch.load(config.pretrained_model, 
                                            map_location=device))
            print("✓ Pre-trained model loaded successfully\n")
        else:
            print(f"⚠️  Warning: Model file not found, training from scratch\n")
    
    model.to(device)
    print('🔧 Trainable parameters:')
    print_trainable_parameters(model)
    
    # Load tokenizer
    if config.lm == 'T5-Base':
        processor = T5Tokenizer.from_pretrained('google-t5/t5-base')
    else:
        processor = T5Tokenizer.from_pretrained('google-t5/t5-large')
    
    processor.add_tokens('<')
    
    # Data path
    clean_data_dir = f'data/poisoned_datasets/{config.data_dir}'
    
    if not os.path.exists(clean_data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {clean_data_dir}")
    
    print(f"\n📂 Dataset directory: {clean_data_dir}\n")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
    ])
    
    # Load datasets (using MultiFrameDataset, not BackdoorMultiFrameDataset)
    print("📥 Loading training set...")
    train_dset = MultiFrameDataset(
        input_file=os.path.join(clean_data_dir, 'multi_frame_train.json'),
        tokenizer=processor,
        transform=transform
    )
    print(f"  ✓ Training samples: {len(train_dset.data)}\n")
    
    print("📥 Loading validation set...")
    val_dset = MultiFrameDataset(
        input_file=os.path.join(clean_data_dir, 'multi_frame_val.json'),
        tokenizer=processor,
        transform=transform
    )
    print(f"  ✓ Validation samples: {len(val_dset.data)}\n")
    
    print("📥 Loading test set...")
    test_dset = MultiFrameDataset(
        input_file=os.path.join(clean_data_dir, 'multi_frame_test.json'),
        tokenizer=processor,
        transform=transform
    )
    print(f"  ✓ Test samples: {len(test_dset.data)}\n")
    
    # Display data sample
    print(f"{'='*70}")
    print("Dataset sample example:")
    print(f"{'='*70}")
    sample_qa, sample_img_paths = train_dset.data[0]
    print(f"Question: {sample_qa['Q'][:100]}...")
    print(f"Answer: {sample_qa['A'][:100]}...")
    print(f"Image count: {len(sample_img_paths)}")
    print(f"{'='*70}\n")
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dset, shuffle=True, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dset, shuffle=False, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dset, shuffle=False, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=collate_fn
    )
    
    # Create save directory
    checkpoint_path = os.path.join('multi_frame_results', timestr)
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f'✓ Save directory: {checkpoint_path}\n')
    
    # Start training
    print(f"{'='*70}")
    print("🚀 Starting training...")
    print(f"{'='*70}")
    
    min_train_loss, min_val_loss, best_perplexity = custom_train(
        min_train_loss, min_val_loss, best_perplexity, best_model, 
        epochs_ran, config.learning_rate
    )
    
    # Test best model
    print(f"\n{'='*70}")
    print("🧪 Testing best model...")
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
    
    # Calculate test perplexity
    test_perplexity, test_loss = calculate_perplexity(test_dataloader, best_model)
    
    # Generate sample outputs
    print("\nGenerating test samples...")
    sample_outputs = generate_samples(test_dataloader, best_model, num_samples=10)
    
    print(f"\n{'='*70}")
    print("📊 Final test results")
    print(f"{'='*70}")
    print(f"Test Perplexity: {test_perplexity:.2f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Best Validation Perplexity: {best_perplexity:.2f}")
    print(f"Min Training Loss: {min_train_loss:.4f}")
    print(f"Min Validation Loss: {min_val_loss:.4f}")
    print(f"\nSample output examples:")
    print(f"{'-'*70}")
    for i, sample in enumerate(sample_outputs[:3], 1):
        print(f"\nSample {i}:")
        print(f"  Input: {sample['input'][:80]}...")
        print(f"  Generated: {sample['generated']}")
        print(f"  Expected: {sample['expected']}")
    print(f"\n{'-'*70}")
    print(f"{'='*70}\n")
    
    # Save results
    statistics = [
        min_train_loss, min_val_loss, best_perplexity, test_perplexity
    ]
    save_experiment(statistics, sample_outputs)
    
    print("✅ Training completed!")
    print(f"📁 All results saved in: {checkpoint_path}/")
    print(f"  - best_model.pth: Best model")
    print(f"  - latest_model.pth: Latest model")
    print(f"  - stats.json: Training statistics")
    print(f"  - metrics.png: Metrics chart")
    print(f"  - results.csv: Experiment results")
    print(f"  - sample_outputs.json: Sample outputs\n")