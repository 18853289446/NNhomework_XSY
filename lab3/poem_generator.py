import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据集加载与预处理 (Dataset)
# ==========================================
def load_and_filter_poems(data_dir='/content'):
    """加载 JSON 文件并过滤出七言绝句"""
    poems = []
    # 遍历目录下的所有 json 文件
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and filename.startswith('poet.'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    content = "".join(item['paragraphs'])
                    # 七言绝句的标准长度为32字符（28个汉字 + 4个标点：，。，。）
                    if len(content) == 32 and '，' in content and '。' in content:
                        poems.append(content)
    return poems

class PoetryDataset(Dataset):
    def __init__(self, poems):
        self.poems = poems
        # 建立字符集
        self.chars = sorted(list(set(''.join(poems))))
        self.chars = ['<PAD>', '<START>', '<EOP>', '<UNK>'] + self.chars
        
        # 建立字符与索引的映射字典
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
    def __len__(self):
        return len(self.poems)
    
    def __getitem__(self, idx):
        poem = self.poems[idx]
        # 添加起始和结束标志
        indices = [self.char2idx['<START>']] + \
                  [self.char2idx.get(c, self.char2idx['<UNK>']) for c in poem] + \
                  [self.char2idx['<EOP>']]
        
        # 输入序列 (x) 和目标序列 (y，即 x 错位一个字符)
        x = torch.tensor(indices[:-1], dtype=torch.long)
        y = torch.tensor(indices[1:], dtype=torch.long)
        return x, y

# ==========================================
# 2. 构建 RNN 网络 (基于 LSTM)
# ==========================================
class PoetryRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers):
        super(PoetryRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        out = self.fc(out)
        return out, hidden

# ==========================================
# 3. 诗歌生成函数
# ==========================================
def generate_poem(model, dataset, start_words="明月", max_len=32, temperature=0.8, device='cpu'):
    model.eval()
    chars = [c for c in start_words]
    hidden = None
    
    # 输入 <START> 标记预热 hidden state
    x = torch.tensor([[dataset.char2idx['<START>']]]).to(device)
    _, hidden = model(x, hidden)
    
    # 输入我们指定的开头词
    for char in chars[:-1]:
        x = torch.tensor([[dataset.char2idx.get(char, dataset.char2idx['<UNK>'])]]).to(device)
        _, hidden = model(x, hidden)
        
    # 开始生成
    x = torch.tensor([[dataset.char2idx.get(chars[-1], dataset.char2idx['<UNK>'])]]).to(device)
    
    with torch.no_grad():
        for _ in range(max_len - len(start_words)):
            out, hidden = model(x, hidden)
            # 添加 temperature 增加随机性和创造性
            prob = torch.softmax(out[0, -1] / temperature, dim=0).cpu().numpy()
            
            # 依概率采样下一个字符
            pred_idx = np.random.choice(len(prob), p=prob)
            pred_char = dataset.idx2char[pred_idx]
            
            if pred_char == '<EOP>':
                break
                
            chars.append(pred_char)
            x = torch.tensor([[pred_idx]]).to(device)
            
    return "".join(chars)

# ==========================================
# 4. 主程序：训练与可视化
# ==========================================
def main():
    # 检查 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 准备数据
    raw_poems = load_and_filter_poems('/content')
    if len(raw_poems) == 0:
        print("未在 /content 目录下找到符合条件的诗歌数据。为了演示，将生成一些虚拟数据。")
        raw_poems = ["明月清風夜夜來，山花落盡客裴回。不知天地無窮事，卻惹閑愁到酒杯。"] * 1000

    print(f"共过滤出 {len(raw_poems)} 首七言绝句用于训练。")
    
    dataset = PoetryDataset(raw_poems)
    
    # 超参数设置
    BATCH_SIZE = 64
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    N_LAYERS = 2
    EPOCHS = 20
    LR = 0.005
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 2. 初始化模型、损失函数和优化器
    model = PoetryRNN(dataset.vocab_size, EMBED_DIM, HIDDEN_DIM, N_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.char2idx['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    epoch_losses = []
    
    # 3. 开始训练
    print("\n--- 开始训练 ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output, _ = model(x)
            
            # 调整维度以计算 Loss: output (batch_size * seq_len, vocab_size), y (batch_size * seq_len)
            loss = criterion(output.view(-1, dataset.vocab_size), y.view(-1))
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 20 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{step}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} ====\n")
        
        # 每个 epoch 结束展示一次生成结果
        print("【生成演示】:")
        generated_poem = generate_poem(model, dataset, start_words="明月", device=device)
        print(f"「{generated_poem}」\n")
        
    # 4. 绘制并保存 Training Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o', color='blue', label='Train Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(1, EPOCHS + 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('/content/training_loss_curve.png')
    plt.show()
    print("Loss 曲线已保存至 /content/training_loss_curve.png")

if __name__ == '__main__':
    main()