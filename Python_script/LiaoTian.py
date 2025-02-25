import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

# 配置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 64
hidden_dim = 128
epochs = 100
max_length = 10

# 示例对话数据集
pairs = [
    ("hello", "hi"),
    ("how are you?", "I'm fine"),
    ("what's your name?", "I'm a chatbot"),
    ("good morning", "good morning!"),
    ("bye", "goodbye"),
    ("中国中国中国","亚洲国家"),
    ("英国英国英国","脱离欧洲"),
    ("法国法国法国","欧洲国家"),
    ("日本日本日本","亚欧发达国家")
]


# 构建词汇表
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.count = 4

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx and self.count < 1000:  # 限制词汇表大小
            self.word2idx[word] = self.count
            self.idx2word[self.count] = word
            self.count += 1


vocab = Vocabulary()
for inp, tgt in pairs:  # 修改变量名
    vocab.add_sentence(inp)
    vocab.add_sentence(tgt)

vocab_size = vocab.count


# 数据预处理
def sentence_to_tensor(sentence):
    indexes = [vocab.word2idx.get(word, 3) for word in sentence.split()]
    #print(indexes)
    return torch.tensor([1] + indexes + [2], dtype=torch.long)


input_tensors = [sentence_to_tensor(inp) for inp, tgt in pairs]
target_tensors = [sentence_to_tensor(tgt) for inp, tgt in pairs]

# 填充序列
input_padded = pad_sequence(input_tensors, padding_value=0, batch_first=True)[:, :max_length]
target_padded = pad_sequence(target_tensors, padding_value=0, batch_first=True)[:, :max_length]


# 模型定义
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, context):
        # 调整维度
        x = self.embedding(x)  # (batch_size, 1, embed_dim)
        context = context.permute(1, 0, 2)  # (batch_size, 1, hidden_dim)
        combined = torch.cat([x, context], dim=2)
        output, hidden = self.gru(combined, hidden)
        output = self.fc(output)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)

    def forward(self, src, trg):
        context = self.encoder(src)  # (1, batch_size, hidden_dim)
        hidden = context
        batch_size = trg.size(0)
        seq_len = trg.size(1)

        outputs = torch.zeros(batch_size, seq_len, vocab_size).to(device)

        # 使用teacher forcing
        for t in range(seq_len - 1):
            output, hidden = self.decoder(trg[:, t].unsqueeze(1), hidden, context)
            outputs[:, t + 1] = output.squeeze(1)

        return outputs


model = Seq2Seq().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)

# 训练
for epoch in range(epochs):
    model.train()
    src = input_padded.to(device)
    trg = target_padded.to(device)

    optimizer.zero_grad()
    output = model(src, trg)

    loss = criterion(output[:, 1:].reshape(-1, vocab_size),
                     trg[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')


# 对话函数
def chat():
    model.eval()
    #print(vocab.word2idx)
    #print(input_padded)
    #print(target_padded)
    while True:
        try:
            input_sentence = input("You: ")
            if input_sentence.lower() in ['exit', 'quit']:
                break

            # 处理输入
            input_tensor = sentence_to_tensor(input_sentence).unsqueeze(0).to(device)
            context = model.encoder(input_tensor)

            # 生成回复
            trg = torch.tensor([[1]], device=device)  # <SOS>
            #print(input_tensor)
            #print(context)
            #print(trg)

            hidden = context
            output_words = []

            for _ in range(max_length):
                output, hidden = model.decoder(trg, hidden, context)
                pred_token = output.argmax(2)
                word_idx = pred_token.item()

                if word_idx == 2:  # <EOS>
                    break
                if word_idx not in [0, 1, 2, 3]:  # 过滤特殊标记
                    output_words.append(vocab.idx2word[word_idx])

                trg = pred_token

            print("Bot:", " ".join(output_words))

        except KeyError:
            print("Bot: I don't understand that yet...")


# 启动聊天
print("Chatbot已启动（输入exit退出）")
chat()
