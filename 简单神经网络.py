import torch

batch_n = 100  # 一个批次输入的数据数量
hidden_layer = 100  # 隐藏层之后保留的数据特征个数
input_data = 1000  # 每个数据的数据特征
output_data = 10  # 分类结果值

# 从输入层到隐藏层、从隐藏层到输出层的权重初始化
x = torch.randn(batch_n, input_data)
y = torch.randn(batch_n, output_data)

w1 = torch.randn(input_data, hidden_layer)
w2 = torch.randn(hidden_layer, output_data)

epoch_n = 20
learning_rate = 1e-6

# 对参数优化，能够显式地写出求导公式
for epoch in range(epoch_n):
    h1 = x.mm(w1)  # 100*1000
    h1 = h1.clamp(min=0)  # 将小于0的值全部赋值为0，就相当于加了一个relu激活函数
    y_pred = h1.mm(w2)  # 100*10

    loss = (y_pred - y).pow(2).sum()
    print("Epoch:{},Loss:{:.4f}".format(epoch, loss))

    grad_y_pred = 2 * (y_pred - y)
    grad_w2 = h1.t().mm(grad_y_pred)

    grad_h = grad_y_pred.clone()
    grad_h = grad_h.mm(w2.t())
    grad_h.clamp(min=0)
    grad_w1 = x.t().mm(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2