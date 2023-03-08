import torch
from torch.autograd import Variable

batch_n = 100
hidden_layer = 100  # 隐藏层之后保留的数据特征个数
input_data = 1000  # 每个数据的数据特征
output_data = 10  # 分类结果值

x = Variable(torch.randn(batch_n, input_data), requires_grad=False)
y = Variable(torch.randn(batch_n, output_data), requires_grad=False)

models = torch.nn.Sequential(
    torch.nn.Linear(input_data, hidden_layer),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer, output_data)
)

epoch_n = 10  # 增加训练次数
learning_rate = 1e-3  # 增加学习速率
loss_fn = torch.nn.MSELoss()

optimzer = torch.optim.Adam(models.parameters(), lr=learning_rate)  # Adam类入参是被优化的参数和学习速率的初始值

for epoch in range(epoch_n):
    y_pred = models(x)
    loss = loss_fn(y_pred, y)
    print("Epoch:{},Loss:{:.4f}".format(epoch, loss.item()))
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
