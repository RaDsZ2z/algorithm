此竞赛来自[李沐课程](https://www.bilibili.com/video/BV1z64y1o7iz/?spm_id_from=333.1387.upload.video_card.click&vd_source=8924ad59b4f62224f165e16aa3d04f00)，[竞赛地址](https://www.kaggle.com/c/classify-leaves)

**下载数据**

点击下载会跳转到Rules界面，点击`Late Submission`同意协议之后才可以下载  

数据文件夹是`classify-leaves`  

其中`images`里是很多张叶子的png图片，标号0到27152，`train.csv`有训练集中每片叶子的label，标号0到18352，`test.csv`是测试集中叶子的文件名，标号18353到27152



使用`DataLoader`加载数据时，参数`num_workers`要先设为0，这个参数值过大会让程序直接停滞

**开始训练**

使用的是resnet，第一次跑完测试精度只有**35%**左右

补充，重复运行之后的结果分别是：**??**

做如下改动

```python
optimizer = torch.optim.SGD(net.parameters(),lr=lr) #小批量随机梯度下降
#改为
momentum = 0.9
optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=momentum) #小批量随机梯度下降
```

我不清楚这个参数对应的技巧理论，或许看一下李沐《动手学深度学习》第28到30节会有答案

修改之后观察到训练过程中测试精度产生了很大的震荡，最后的结果是**75%**  

补充，重复运行之后的结果分别是：  **74.1%，26.9%**

再加上权重衰退试试

```python
momentum = 0.9
optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=momentum) #小批量随机梯度下降
```

权重衰退对应的理论我也不记得了，可以复习第12节教程

加上权重衰退后，测试精度的震荡幅度依然很大，最终结果是**83.8%**。

补充，重复运行之后的结果分别是：**71.7%，85.1%，72.0%**

感觉目前这个网络是不太work的，主要是震动幅度太大，需要继续优化(有时间先多跑几次，看看最终测试精度的波动程度)

Q：我的测试精度为什么会震荡，我不知道原因

A：

Q：函数`train_ch6`里面的`net.train()`是在干嘛？前向运算吧？

A：



[参考kaggle博客](https://www.kaggle.com/code/wangdark/classify-leaves-resnet)  



之前把参考博客的代码完全搬到自己电脑上，没跑起来，读取数据时的`num_workers`改成0之后跑起来了

# 下面是对参考博客的解读

```python
import torchvision.transforms as T
transform = T.Compose([
    T.ToTensor()
])
img = transform(img)
```

经过`transform`前后`img`发生了什么变化呢，`cursor`说

> 把输入图片从PIL 图像或 numpy 数组（形状 [H, W, C]，像素 0–255）
>
> 转成
>
> PyTorch 张量（形状 [C, H, W]，类型 float32，数值缩放到 0–1）

CHW分别是通道数、高、宽。

看看数据实际上的变化：

```python
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
base_path = './classify-leaves'
class LeaveDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(base_path, self.data.iloc[idx,0])
        image = Image.open(img_path)
        array = np.array(image)
        print('shape1:',array.shape)
        print('mean1:',array.mean())
        if self.transform:
            image = self.transform(image)  # 应用转换
        print('shape2:',image.shape)
        print('mean2:',image.mean())
        label_name = self.data.iloc[idx,1]
        label = label2idx[label_name]  # 转换为整数索引

        return image, label  # 返回 (图片, 标签)

train_df= pd.read_csv(os.path.join(base_path,"train.csv"))
# 获取所有唯一的类别（叶子种类）
unique_labels = train_df["label"].unique()

# 创建 类别 → 索引 的映射
label2idx = {label: idx for idx, label in enumerate(unique_labels)}

# 创建反向映射（id → label）
idx2label = {v: k for k, v in label2idx.items()}
transform = T.Compose([
    T.ToTensor(),
])
train_dataset = LeaveDataset(train_df,transform)
train_dataset[0]
pass
'''
shape1: (224, 224, 3)
mean1: 242.8440622342687
shape2: torch.Size([3, 224, 224])
mean2: tensor(0.9523)
'''
```

通过输出可以看到，形状从(224,224,3)变为了(3,224,224)

均值从242变为了0.95
