import os
import sys
import pandas
import torch
import gzip
import itertools
from torch import nn
from matplotlib import pyplot

class MyModel(nn.Module):
    """根据码农条件预测工资的模型"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=8, out_features=200)
        self.layer2 = nn.Linear(in_features=200, out_features=100)
        self.layer3 = nn.Linear(in_features=100, out_features=1)
        self.batchnorm1 = nn.BatchNorm1d(200)
        self.batchnorm2 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        hidden1 = self.dropout1(self.batchnorm1(nn.functional.relu(self.layer1(x))))
        hidden2 = self.dropout2(self.batchnorm2(nn.functional.relu(self.layer2(hidden1))))
        y = self.layer3(hidden2)
        return y

def save_tensor(tensor, path):
    """保存 tensor 对象到文件"""
    torch.save(tensor, gzip.GzipFile(path, "wb"))

def load_tensor(path):
    """从文件读取 tensor 对象"""
    return torch.load(gzip.GzipFile(path, "rb"))

def prepare():
    """准备训练"""
    # 数据集转换到 tensor 以后会保存在 data 文件夹下
    if not os.path.isdir("data"):
        os.makedirs("data")

    # 从 csv 读取原始数据集，分批每次读取 2000 行
    for batch, df in enumerate(pandas.read_csv('salary.csv', chunksize=2000)):
        dataset_tensor = torch.tensor(df.values, dtype=torch.float)

        # 正规化输入和输出
        dataset_tensor *= torch.tensor([0.01, 1, 0.01, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0001])

        # 切分训练集 (60%)，验证集 (20%) 和测试集 (20%)
        random_indices = torch.randperm(dataset_tensor.shape[0])
        traning_indices = random_indices[:int(len(random_indices)*0.6)]
        validating_indices = random_indices[int(len(random_indices)*0.6):int(len(random_indices)*0.8):]
        testing_indices = random_indices[int(len(random_indices)*0.8):]
        training_set = dataset_tensor[traning_indices]
        validating_set = dataset_tensor[validating_indices]
        testing_set = dataset_tensor[testing_indices]

        # 保存到硬盘
        save_tensor(training_set, f"data/training_set.{batch}.pt")
        save_tensor(validating_set, f"data/validating_set.{batch}.pt")
        save_tensor(testing_set, f"data/testing_set.{batch}.pt")
        print(f"batch {batch} saved")

def train():
    """开始训练"""
    # 创建模型实例
    model = MyModel()

    # 创建损失计算器
    loss_function = torch.nn.MSELoss()

    # 创建参数调整器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # 记录训练集和验证集的正确率变化
    traning_accuracy_history = []
    validating_accuracy_history = []

    # 记录最高的验证集正确率
    validating_accuracy_highest = 0
    validating_accuracy_highest_epoch = 0

    # 读取批次的工具函数
    def read_batches(base_path):
        for batch in itertools.count():
            path = f"{base_path}.{batch}.pt"
            if not os.path.isfile(path):
                break
            yield load_tensor(path)

    # 计算正确率的工具函数
    def calc_accuracy(actual, predicted):
        return max(0, 1 - ((actual - predicted).abs() / actual.abs()).mean().item())

    # 开始训练过程
    for epoch in range(1, 10000):
        print(f"epoch: {epoch}")

        # 根据训练集训练并修改参数
        # 切换模型到训练模式，将会启用自动微分，批次正规化 (BatchNorm) 与 Dropout
        model.train()
        traning_accuracy_list = []
        for batch in read_batches("data/training_set"):
            # 切分小批次，有助于泛化模型
             for index in range(0, batch.shape[0], 100):
                # 划分输入和输出
                batch_x = batch[index:index+100,:-1]
                batch_y = batch[index:index+100,-1:]
                # 计算预测值
                predicted = model(batch_x)
                # 计算损失
                loss = loss_function(predicted, batch_y)
                # 从损失自动微分求导函数值
                loss.backward()
                # 使用参数调整器调整参数
                optimizer.step()
                # 清空导函数值
                optimizer.zero_grad()
                # 记录这一个批次的正确率，torch.no_grad 代表临时禁用自动微分功能
                with torch.no_grad():
                    traning_accuracy_list.append(calc_accuracy(batch_y, predicted))
        traning_accuracy = sum(traning_accuracy_list) / len(traning_accuracy_list)
        traning_accuracy_history.append(traning_accuracy)
        print(f"training accuracy: {traning_accuracy}")

        # 检查验证集
        # 切换模型到验证模式，将会禁用自动微分，批次正规化 (BatchNorm) 与 Dropout
        model.eval()
        validating_accuracy_list = []
        for batch in read_batches("data/validating_set"):
            validating_accuracy_list.append(calc_accuracy(batch[:,-1:],  model(batch[:,:-1])))
        validating_accuracy = sum(validating_accuracy_list) / len(validating_accuracy_list)
        validating_accuracy_history.append(validating_accuracy)
        print(f"validating accuracy: {validating_accuracy}")

        # 记录最高的验证集正确率与当时的模型状态，判断是否在 100 次训练后仍然没有刷新记录
        if validating_accuracy > validating_accuracy_highest:
            validating_accuracy_highest = validating_accuracy
            validating_accuracy_highest_epoch = epoch
            save_tensor(model.state_dict(), "model.pt")
            print("highest validating accuracy updated")
        elif epoch - validating_accuracy_highest_epoch > 100:
            # 在 100 次训练后仍然没有刷新记录，结束训练
            print("stop training because highest validating accuracy not updated in 100 epoches")
            break

    # 使用达到最高正确率时的模型状态
    print(f"highest validating accuracy: {validating_accuracy_highest}",
        f"from epoch {validating_accuracy_highest_epoch}")
    model.load_state_dict(load_tensor("model.pt"))

    # 检查测试集
    testing_accuracy_list = []
    for batch in read_batches("data/testing_set"):
        testing_accuracy_list.append(calc_accuracy(batch[:,-1:],  model(batch[:,:-1])))
    testing_accuracy = sum(testing_accuracy_list) / len(testing_accuracy_list)
    print(f"testing accuracy: {testing_accuracy}")

    # 显示训练集和验证集的正确率变化
    pyplot.plot(traning_accuracy_history, label="traning")
    pyplot.plot(validating_accuracy_history, label="validing")
    pyplot.ylim(0, 1)
    pyplot.legend()
    pyplot.show()

def eval_model():
    """使用训练好的模型"""
    parameters = [
        "Age",
        "Gender (0: Male, 1: Female)",
        "Years of work experience",
        "Java Skill (0 ~ 5)",
        "NET Skill (0 ~ 5)",
        "JS Skill (0 ~ 5)",
        "CSS Skill (0 ~ 5)",
        "HTML Skill (0 ~ 5)"
    ]

    # 创建模型实例，加载训练好的状态，然后切换到验证模式
    model = MyModel()
    model.load_state_dict(load_tensor("model.pt"))
    model.eval()

    # 询问输入并预测输出
    while True:
        try:
            x = torch.tensor([int(input(f"Your {p}: ")) for p in parameters], dtype=torch.float)
            # 正规化输入
            x *= torch.tensor([0.01, 1, 0.01, 0.2, 0.2, 0.2, 0.2, 0.2])
            # 转换到 1 行 1 列的矩阵，这里其实可以不转换但推荐这么做，因为不是所有模型都支持非批次输入
            x = x.view(1, len(x))
            # 预测输出
            y = model(x)
            # 反正规化输出
            y *= 10000
            print("Your estimated salary:", y[0,0].item(), "\n")
        except Exception as e:
            print("error:", e)

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print(f"Please run: {sys.argv[0]} prepare|train|eval")
        exit()

    # 给随机数生成器分配一个初始值，使得每次运行都可以生成相同的随机数
    # 这是为了让过程可重现，你也可以选择不这样做
    torch.random.manual_seed(0)

    # 根据命令行参数选择操作
    operation = sys.argv[1]
    if operation == "prepare":
        prepare()
    elif operation == "train":
        train()
    elif operation == "eval":
        eval_model()
    else:
        raise ValueError(f"Unsupported operation: {operation}")

if __name__ == "__main__":
    main()
