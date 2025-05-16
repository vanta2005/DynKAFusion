import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from model.M8 import self_net
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from utils.metrics import Evaluator
from torch.cuda.amp import autocast
from utils.plot import plot_miou, plot_defect_iou, plot_loss, plot_recall, plot_precision, plot_f1_score, plot_accuracy
import pandas as pd
from torch.cuda.amp import autocast

# 定义TopK损失函数
class TopKLoss(nn.Module):
    def __init__(self, k_percent=0.2):
        super(TopKLoss, self).__init__()
        self.k_percent = k_percent

    def forward(self, pred, target):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # 选择前k%的像素
        k = int(self.k_percent * ce_loss.numel())
        topk_loss, _ = torch.topk(ce_loss.view(-1), k)
        
        # 返回平均损失
        return topk_loss.mean()

# 设置路径，定义图像、掩码和检查点的目录
dir_img = Path('/root/task2/data1/train/images')
dir_mask = Path('/root/task2/data1/train/masks')
valid_img = Path('/root/task2/data1/valid/images')
valid_mask = Path('/root/task2/data1/valid/masks')
save_plot_path = Path('/root/task2/plots/M9')  # 保存曲线图的目录
# 定义保存模型的目录路径
best_model_dir = '/root/task2/best_model/M9'
best_model_path = os.path.join(best_model_dir, 'best_model_miou.pth')
# 定义保存 CSV 文件的目录路径
train_csv_dir = '/root/task2/csv/M9'
val_csv_dir = train_csv_dir  # 使用相同的目录路径来保存验证集的 CSV 文件

def validate_model(model, val_loader, evaluator, device, epoch, criterion, amp=True):
    """
    验证模型，返回每类 IoU、去除背景后的平均 mIoU，验证集的平均损失、Recall、Precision、Accuracy 和 F1 Score。
    """
    model.eval()
    evaluator.reset()
    epoch_val_loss = 0  # 当前 epoch 的验证损失

    # 保存验证集所有批次的真实标签和预测标签
    true_labels_all = []
    pred_labels_all = []

    # 初始化 Accuracy 计算
    total_pixels = 0
    correct_pixels = 0

    with tqdm(total=len(val_loader), desc=f'Validation Epoch {epoch}', unit='batch') as pbar:
        with torch.no_grad():
            for batch in val_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with autocast(enabled=amp):
                    masks_pred = model(images)
                    if isinstance(masks_pred, tuple):
                        masks_pred = masks_pred[-1]  # Select the final output if multiple outputs exist
                
                    # Ensure target is in one-hot format
                    true_masks_one_hot = F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()
                
                    # Check shapes
                    assert masks_pred.size() == true_masks_one_hot.size(), f"Shape mismatch: {masks_pred.size()} vs {true_masks_one_hot.size()}"
                    val_loss = criterion(masks_pred, true_masks) + dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )
                    epoch_val_loss += val_loss.item()

                # 获取预测和真实标签
                pred_masks = masks_pred.argmax(dim=1).cpu().numpy()
                true_masks = true_masks.cpu().numpy()

                # 计算每个像素的 Accuracy
                batch_total_pixels = true_masks.size  # 每个批次的像素数量
                batch_correct_pixels = np.sum(true_masks == pred_masks)  # 正确预测的像素数量

                total_pixels += batch_total_pixels
                correct_pixels += batch_correct_pixels

                # 收集所有真实标签和预测标签
                true_labels_all.append(true_masks)
                pred_labels_all.append(pred_masks)

                # 添加到 evaluator
                evaluator.add_batch(true_masks, pred_masks)

                IoU_per_class_no_bg, defect_mIoU = evaluator.Mean_Intersection_over_Union(exclude_background=True, num_classes=4)
                recall_per_class, recall = evaluator.Recall()  # 获取 Recall 的标量值
                precision_per_class, precision = evaluator.Precision()  # 获取 Precision 的标量值
                pbar.set_postfix({'Validation mIoU (batch)': defect_mIoU, 'Recall (batch)': recall, 'Precision (batch)': precision})
                pbar.update(1)

    # 计算验证集的平均损失
    avg_val_loss = epoch_val_loss / len(val_loader)

    # 计算验证集整体的 IoU、Recall 和 Precision
    IoU_per_class_no_bg, defect_mIoU = evaluator.Mean_Intersection_over_Union(exclude_background=True, num_classes=4)
    recall_per_class, recall = evaluator.Recall()  # 获取 Recall 的标量值
    precision_per_class, precision = evaluator.Precision()  # 获取 Precision 的标量值

    # 计算准确度 (Accuracy)
    accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0

    # 将所有批次的标签展平
    true_labels_all = np.concatenate([m.flatten() for m in true_labels_all])
    pred_labels_all = np.concatenate([m.flatten() for m in pred_labels_all])

    # 计算 F1 Score（按类别计算 F1 score）
    class_f1_scores = []
    for i in range(model.n_classes):
        # 计算每个类别的 Precision 和 Recall
        class_precision = precision_per_class[i]  # 每个类别的 Precision
        class_recall = recall_per_class[i]        # 每个类别的 Recall
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        class_f1_scores.append(class_f1)

    # 计算总体 F1 score（所有类别的 F1 score 均值）
    f1_score = np.mean(class_f1_scores)

    return IoU_per_class_no_bg, defect_mIoU, avg_val_loss, recall, precision, accuracy, f1_score

def train_model(
    model,
    device,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-6,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
    use_topk_loss: bool = False,  # 新增参数，控制是否使用TopK损失
    topk_percent: float = 0.2,   # 新增参数，控制TopK损失的k值
):
    # 初始化记录各项指标的日志
    train_miou_logs = []
    val_miou_logs = []
    train_loss_logs = []
    val_loss_logs = []
    train_recall_logs = []
    val_recall_logs = []
    train_precision_logs = []
    val_precision_logs = []
    train_accuracy_logs = []  # Accuracy log
    val_accuracy_logs = []    # Accuracy log
    train_f1_logs = []
    val_f1_logs = []

    # 1. 创建训练集
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. 创建训练集数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    train_loader = DataLoader(dataset, shuffle=True, **loader_args)

    # 3. 加载自定义验证集
    try:
        val_dataset = CarvanaDataset(valid_img, valid_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        val_dataset = BasicDataset(valid_img, valid_mask, img_scale)

    # 创建验证集数据加载器
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)
    logging.info(f'''Starting training:
        Epochs: {epochs}
        Batch size: {batch_size}
        Learning rate: {learning_rate}
        Training size: {len(dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints: {save_checkpoint}
        Device: {device.type}
        Images scaling: {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. 设置优化器、损失函数和学习率调度器
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    # 根据是否使用TopK损失选择损失函数
    if use_topk_loss:
        criterion = TopKLoss(k_percent=topk_percent)
    else:
        criterion = nn.CrossEntropyLoss()
    
    global_step = 0

    # 初始化记录最好的 mIoU 和最小的验证集损失
    best_miou = -float('inf')
    best_loss = float('inf')  # 初始化为正无穷大

    # 5. 开始训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        evaluator = Evaluator(num_class=model.n_classes)  # 每轮重新初始化 Evaluator

        # 初始化训练集准确度计算
        total_train_pixels = 0
        correct_train_pixels = 0

        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
            
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
            
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)
                    if not use_topk_loss:  # 如果不使用TopK损失，则添加Dice损失
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
            
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                # 计算训练集的 mIoU、Recall 和 Precision
                pred_masks = masks_pred.argmax(dim=1).cpu().numpy()
                true_masks = true_masks.cpu().numpy()
                evaluator.add_batch(true_masks, pred_masks)  # 更新混淆矩阵
                
                # 计算训练集的 Accuracy
                total_train_pixels += true_masks.size
                correct_train_pixels += np.sum(true_masks == pred_masks)

                # 计算训练集的 mIoU、Recall 和 Precision
                IoU_per_class_no_bg, defect_mIoU = evaluator.Mean_Intersection_over_Union(exclude_background=True, num_classes=4)
                recall_per_class, recall = evaluator.Recall()  # Recall 应该是标量值
                precision_per_class, precision = evaluator.Precision()  # Precision 应该是标量值

                # 计算训练集的 F1 score
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                pbar.set_postfix(mIoU=defect_mIoU, Recall=recall, Precision=precision)

            train_accuracy = correct_train_pixels / total_train_pixels if total_train_pixels > 0 else 0
            train_accuracy_logs.append(train_accuracy)

            # 每个 epoch 结束时，计算并记录训练集的 mIoU、Recall 和 Precision
            train_miou_logs.append(defect_mIoU)
            train_recall_logs.append(recall)
            train_precision_logs.append(precision)
            train_f1_logs.append(f1_score)

            # 记录训练损失
            avg_train_loss = epoch_loss / len(train_loader)
            train_loss_logs.append(avg_train_loss)
                
            print(f'Epoch {epoch}, Training IoU: {IoU_per_class_no_bg}, mIoU: {defect_mIoU}, Recall: {recall}, Precision: {precision}, Loss/train: {avg_train_loss}')

        # 验证阶段：计算验证集的 mIoU、Recall、Precision 和损失
        val_IoU_per_class_no_bg, avg_val_miou, avg_val_loss, val_recall, val_precision, val_accuracy, val_f1_score = validate_model(
            model, val_loader, evaluator, device, epoch, criterion, amp=amp
        )

        # 将验证集指标记录到日志中
        val_miou_logs.append(avg_val_miou)
        val_recall_logs.append(val_recall)
        val_precision_logs.append(val_precision)
        val_loss_logs.append(avg_val_loss)
        val_accuracy_logs.append(val_accuracy)  # Add val_accuracy log

        # 计算 F1 score
        val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
        val_f1_logs.append(val_f1_score)

        # 保存最佳模型（基于 mIoU 和验证损失）
        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            if save_checkpoint:
                # 确保保存目录存在
                os.makedirs(best_model_dir, exist_ok=True)
                
                # 保存模型
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved at: {best_model_path}")
                        
        # 绘制曲线
        plot_miou(train_miou_logs, val_miou_logs, save_plot_path)
        plot_recall(train_recall_logs, val_recall_logs, save_plot_path)
        plot_precision(train_precision_logs, val_precision_logs, save_plot_path)
        plot_loss(train_loss_logs, val_loss_logs, save_plot_path)

        # 绘制 F1 Score 曲线
        plot_f1_score(train_f1_logs, val_f1_logs, save_plot_path)

        # 绘制 Accuracy 曲线
        plot_accuracy(train_accuracy_logs, val_accuracy_logs, save_plot_path)

    # 保存训练和验证指标为 CSV 文件
    train_df = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_mIoU': train_miou_logs,
        'train_loss': train_loss_logs,
        'train_recall': train_recall_logs,
        'train_precision': train_precision_logs,
        'train_f1_score': train_f1_logs,
        'train_accuracy': train_accuracy_logs
    })
    val_df = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'val_mIoU': val_miou_logs,
        'val_loss': val_loss_logs,
        'val_recall': val_recall_logs,
        'val_precision': val_precision_logs,
        'val_f1_score': val_f1_logs,
        'val_accuracy': val_accuracy_logs
    })
       
    # 确保保存目录存在
    os.makedirs(train_csv_dir, exist_ok=True)
    
    # 保存训练和验证指标为 CSV 文件
    train_csv_path = os.path.join(train_csv_dir, 'train_metrics.csv')
    val_csv_path = os.path.join(val_csv_dir, 'val_metrics.csv')
    
    # 保存 CSV 文件
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"Train metrics saved at: {train_csv_path}")
    print(f"Validation metrics saved at: {val_csv_path}")

    print('训练完成。')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-6,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')
    parser.add_argument('--use-topk-loss', action='store_true', default=False, help='Use TopK loss')  # 新增参数
    parser.add_argument('--topk-percent', type=float, default=0.2, help='TopK percentage')  # 新增参数

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 初始化模型
    model = self_net(n_classes=4)
    model = model.to(memory_format=torch.channels_last)

    if args.load:
        # 如果需要加载模型
        model.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            use_topk_loss=args.use_topk_loss,  # 传递use_topk_loss参数
            topk_percent=args.topk_percent     # 传递topk_percent参数
        )
    except RuntimeError as e:
        if 'out of memory' in str(e):
            logging.error('Detected OutOfMemoryError! Enabling checkpointing to reduce memory usage.')
            torch.cuda.empty_cache()

            # 手动启用梯度检查点功能
            train_model(
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=args.amp,
                use_topk_loss=args.use_topk_loss,  # 传递use_topk_loss参数
                topk_percent=args.topk_percent     # 传递topk_percent参数
            )
        else:
            raise e