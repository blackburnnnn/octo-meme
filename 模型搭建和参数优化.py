import torch
from torch.autograd import Variable

batch_n = 100
hidden_layer = 100  # 隐藏层之后保留的数据特征个数
input_data = 1000  # 每个数据的数据特征
output_data = 10  # 分类结果值

models = torch.nn.Sequential(
    torch.nn.Linear(input_data, hidden_layer),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer, output_data)
)

epoch_n = 10000  # 增加训练次数
learning_rate = 1e-4  # 增加学习速率
loss_fn = torch.nn.MSELoss()

# 模型训练和参数优化
# (10/20)不知道为什么下面的代码跑起来之后输出的损失值没变化
x = Variable(torch.randn(100, 1000))  # 随机生成维度是（100,100）的参数
y = Variable(torch.randn(100, 10))
for epoch in range(epoch_n):
    y_pred = models(x)
    loss = loss_fn(y_pred, y)
    if epoch % 1000 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch, loss.item()))
    loss.backward()
    for param in models.parameters():
        param.grad -= param.grad * learning_rate
    models.zero_grad()