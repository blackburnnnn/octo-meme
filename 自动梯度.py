# 自动梯度例子
import torch
from torch.autograd import Variable

batch_n = 100
hidden_layer = 100  # 隐藏层之后保留的数据特征个数
input_data = 1000  # 每个数据的数据特征
output_data = 10  # 分类结果值

# x和y不保留梯度值，因为这两个变量不是我们模型需要优化的参数
x = Variable(torch.randn(batch_n, input_data), requires_grad=False)
y = Variable(torch.randn(batch_n, output_data), requires_grad=False)

w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad=True)
w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad=True)

epoch_n = 20
learning_rate = 1e-6

for epoch in range(epoch_n):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print("Epoch:{},Loss:{:.4f}".format(epoch, loss.item()))  # loss.data[0]报错

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()