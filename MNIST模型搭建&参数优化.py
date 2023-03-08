import torch
from torch.autograd import Variable
import torchvision
from 手写数字识别 import data_train,data_test,data_loader_train,data_loader_test
import matplotlib.pyplot as plt
import torch.nn


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 1---3
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


model = Model()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 1

if __name__ == '__main__':
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-" * 10)
        train_i = 0
        print('类型 ',type(data_loader_train),'train数量 ', len(data_loader_train), 'test数量 ', len(data_loader_test))
        for data in data_loader_train:
            X_train, y_train = data
            X_train, y_train = Variable(X_train), Variable(y_train)
            outputs = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            loss = cost(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()  # pytorch更新后，需要将loss.data[0]改成loss.item()
            running_correct += torch.sum(pred == y_train.data)  # 输出是tensor
            train_i += 1
            print(train_i)

        testing_correct = 0
        test_i = 0
        for data in data_loader_test:
            X_test, y_test = data
            X_test, y_test = Variable(X_test), Variable(y_test)
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(pred == y_test.data)
            test_i += 1
            print(test_i)
        print(
            "Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}%".format(running_loss / len(data_train),
                                                                                         100 * running_correct.numpy() / len(
                                                                                             data_train),
                                                                                         100 * testing_correct.numpy() / len(
                                                                                             data_test)))

    data_loader_test1 = torch.utils.data.DataLoader(dataset=data_test, batch_size=4, shuffle=True)
    X_test1, y_test1 = next(iter(data_loader_test1))
    inputs1 = Variable(X_test1)
    pred1 = model(inputs1)
    # dim=1表示输出所在行的最大值，若改写成dim=0则输出所在列的最大值
    _, pred1 = torch.max(pred1, 1)  # 返回具体的value，和value所在的index（_可用其他变量替换）

    print("Predict Label is:", [i for i in pred1.data])
    print("Real Label is:", [i for i in y_test1])

    # 该方法可以用来直接将数据结构[batch,channel.height,width]形式的图像转化为图像矩阵，便于将多张图像进行可视化。
    img1 = torchvision.utils.make_grid(X_test1)
    img1 = img1.numpy().transpose(1, 2, 0)

    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img1 = img1 * std + mean
    plt.imshow(img1)