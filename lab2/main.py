import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU型号:", torch.cuda.get_device_name(0))

# ==================== 数据集类 ====================
class SVHNDataset(Dataset):
    def __init__(self, mat_file, transform=None):
        print(f"加载数据集: {mat_file}")
        data = loadmat(mat_file)
        self.images = data['X'].transpose(3, 2, 0, 1)  # 转换为(N, C, H, W)
        self.labels = data['y'].flatten()
        # 将标签10转换为0（因为SVHN中10代表数字0）
        self.labels[self.labels == 10] = 0
        self.transform = transform
        print(f"数据集大小: {len(self.images)} 张图片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image_pil = Image.fromarray(np.transpose(image, (1, 2, 0)))
            image = self.transform(image_pil)
        else:
            image = torch.tensor(image, dtype=torch.float32) / 255.0

        return image, label

# ==================== CNN模型 ====================
class SVHNNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SVHNNet, self).__init__()
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==================== 训练函数 ====================
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    train_loss = running_loss / len(train_loader)
    return train_acc, train_loss

# ==================== 测试函数 ====================
def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    test_loss = running_loss / len(test_loader)
    return test_acc, test_loss

# ==================== 主函数 ====================
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 检查数据文件是否存在
    if not os.path.exists('train_32x32.mat') or not os.path.exists('test_32x32.mat'):
        print("错误：数据文件不存在！请先运行数据下载代码。")
        return

    # 加载数据集
    print("\n正在加载训练数据集...")
    train_dataset = SVHNDataset('train_32x32.mat', transform=transform)
    print("正在加载测试数据集...")
    test_dataset = SVHNDataset('test_32x32.mat', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # 初始化模型、损失函数和优化器
    print("\n初始化模型...")
    model = SVHNNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练参数
    num_epochs = 30
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []

    print(f"\n{'='*60}")
    print(f"开始训练，总轮数: {num_epochs}")
    print(f"{'='*60}\n")

    # 训练循环
    for epoch in range(num_epochs):
        train_acc, train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_acc, test_loss = test_model(model, test_loader, criterion, device)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        scheduler.step()

        print(f'Epoch [{epoch+1:2d}/{num_epochs}], '
              f'Train Acc: {train_acc:6.2f}%, Train Loss: {train_loss:.4f}, '
              f'Test Acc: {test_acc:6.2f}%, Test Loss: {test_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'svhn_cnn_model.pth')
    print("\n模型已保存为 'svhn_cnn_model.pth'")

    # 绘制准确率曲线
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy', linewidth=2)
    plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Testing Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss', linewidth=2)
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Testing Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{'='*60}")
    print(f"训练完成")
    print(f"最终测试准确率: {test_accuracies[-1]:.2f}%")
    print(f"最终训练准确率: {train_accuracies[-1]:.2f}%")
    print(f"{'='*60}")

# 运行主函数
if __name__ == "__main__":
    main()