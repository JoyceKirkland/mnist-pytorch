'''
Author: JoyceKirkland joyce84739879@163.com
Date: 2022-07-05 09:55:14
LastEditors: JoyceKirkland joyce84739879@163.com
LastEditTime: 2022-07-05 15:14:13
FilePath: /pytorch-mnist/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#基于Pytorch实现的手写数字模型训练网络
#1、准备数据集
#2、设计相应的模型
#3、构造恰当的损失函数和优化器
#4、设计训练循环(Training cycle)+测试集，边测试边训练
from statistics import mode
import torch
from torchvision import transforms # 对图像进行各种处理的包(针对数据集进行原始处理)
from torchvision import datasets # 加载数据集
from torch.utils.data import DataLoader
import torch.nn.functional as F # 激活全链接层，relu作为激活函数
import torch.optim as optim # 优化器

batch_size=64

# 构建 Compose 类实例，将一系列图像进行提前批处理并归一化。
# 将图像中[0,255]的值转化为张量（最好是处于[0,1]之间且服从正态分布，这样神经网络的训练效果最好）
# 处理后输入的张量维度是4维[N*c*w*h][N个样本*通道数*宽*高]
transform =transforms.Compose([

  # 为了进行更高效的运算，将w*h*c的图像张量转化为c*w*h格式的张量(通道*宽*高)
  transforms.ToTensor(),

# 归一化（数据集标准化）。0.1307为均值，0.3081为标准差（经验值，大牛们提前算好的）
# 切换成[0,1]分布
  # transforms.Normalize(0.1307,),(0.3081,))
  transforms.Normalize((0.1307,),(0.3081,))
  ])

train_dataset= datasets.MNIST(root='../dataset/mnist/',train=True,download=False,transform=transform)

train_loader= DataLoader(train_dataset,shuffle=True,batch_size=batch_size)

test_dataset=datasets.MNIST(root='../dataset/mnist/',train=False,download=True,transform=transform)

test_loader=DataLoader(test_dataset,shuffle=False,batch_size=batch_size)


class Net(torch.nn.Module):
  def __init__(self):
      super(Net,self).__init__()
      self.l1=torch.nn.Linear(784,512)
      self.l2=torch.nn.Linear(512,256)
      self.l3=torch.nn.Linear(256,128)
      self.l4=torch.nn.Linear(128,64)
      self.l5=torch.nn.Linear(64,10)

  def forward(self,x):
    # 将输入的4维的张量变成矩阵
    x=x.view(-1,784)
    # 用relu对每一层算出来的结果进行激活
    x=F.relu(self.l1(x))
    x=F.relu(self.l2(x))
    x=F.relu(self.l3(x))
    x=F.relu(self.l4(x))

    # 最后一层不做激活
    return self.l5(x)

model=Net()

# 损失函数选择交叉熵损失
criterion=torch.nn.CrossEntropyLoss()

# 优化器
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)


# 训练-测试，单循环
def train(epoch):
  running_loss=0.0
  for batch_idx,data in enumerate(train_loader,0):
    # 输入x作为inputs,y作为target存入data
    inputs,target=data
    # 优化器清零
    optimizer.zero_grad()

    # forward+backward+update。前馈+反馈+更新

    # 前馈
    outputs=model(inputs)# 计算输出
    loss=criterion(outputs,target) # 计算损失
    #反馈
    loss.backward()
    #更新，优化
    optimizer.step()

    # 将累计的损失值取出方便输出，每训练300次迭代输出一次running_loss
    running_loss +=loss.item()
    if batch_idx % 300 ==299:
      print('[%d,%5d] loss:%.3f' % (epoch+1,batch_idx+1,running_loss/300))
      running_loss=0.0

# 测试
def test():
  # 预测正确的数量
  correct =0
  # 样本总数
  total=0
  with torch.no_grad():
    for data in test_loader:
      # 从test_loader里取出数据
      images,labels=data
      # 取出数据后做预测，预测完后的结果矩阵每一个样本都有一个维度（十个值）
      outputs=model(images)
      # 求结果矩阵中每一行十个值中最大值的下标，下标值对应的就是0-9中某个数字的分类
     # 结果返回每一行的最大值和此最大值的下标
      _, predicted =torch.max(outputs.data,dim=1)
      # 加上批量总数
      total +=labels.size(0)
      # 判断预测分类和真实分类是否一直，相等为1，不相等为0
      # 然后做求和，把标量值取出来。就是获取预测结果正确的量是多少。
      correct +=(predicted ==labels).sum().item()

      #用正确数/总数来求Accuracy
  print('Accuracy on test set: %d %%'%(100*correct / total))

if __name__=='__main__':
  #一轮训练一轮测试
  for epoch in range(10):
    train(epoch)
    test()







