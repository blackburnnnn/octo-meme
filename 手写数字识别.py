import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt

# 实现下载的代码是torchvision.datasets.MNIST。
# 其他的数据集如COCO、ImageNet、CIFAR等都可以通过这个方法快速下载和载入。

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
data_train = datasets.MNIST(root="./data/", transform=transform, train=True, download=True)  # 本地不会下载
data_test = datasets.MNIST(root="./data/", transform=transform, train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=64, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=64, shuffle=True)

images, labels = next(iter(data_loader_train))  # 获取一个批次的图片和标签
img = torchvision.utils.make_grid(images)  # 将一个批次的图片构造成网格模式

img = img.numpy().transpose(1, 2, 0)
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
# print([labels[i] for i in range(64)])  # 打印这个批次数据的全部标签
plt.imshow(img)  # 显示图片
