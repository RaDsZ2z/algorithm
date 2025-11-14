# 0.前言

此竞赛来自[李沐课程](https://www.bilibili.com/video/BV1z64y1o7iz/?spm_id_from=333.1387.upload.video_card.click&vd_source=8924ad59b4f62224f165e16aa3d04f00)，[竞赛地址](https://www.kaggle.com/c/classify-leaves)

**下载数据**

点击下载会跳转到Rules界面，点击`Late Submission`同意协议之后才可以下载  

数据文件夹是`classify-leaves`  

其中`images`里是很多张叶子的png图片，标号0到27152，`train.csv`有训练集中每片叶子的label，标号0到18352，`test.csv`是测试集中叶子的文件名，标号18353到27152

# 1.我自己的解决方案

对应文件是`solution.ipynb`，沿用了做手写数字识别时的代码，搬过来之后改了改

使用`DataLoader`加载数据时，参数`num_workers`要先设为0，这个参数值过大会让程序直接停滞**（原因暂时未知）**

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







之前把参考博客的代码完全搬到自己电脑上，没跑起来，读取数据时的`num_workers`改成0之后跑起来了

# 2.解读参考博客的解决方案

[参考kaggle博客](https://www.kaggle.com/code/wangdark/classify-leaves-resnet)  

对应文件`solution_ref.ipynb`

## 2.1.ToTensor

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

## 2.2.定义模型

```python
import torchvision.models as models

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)#封装好的resnet18
num_classes = len(label_counts)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)
```

`resnet.fc`对应`resnet18`的最后一层，它原本也是一个全连接层`torch.nn.Linear(in_features=512, out_features=1000)`

`torch.nn.Linear`创建一个全连接层，两个参数是输入特征数和输出特征数

`resnet.fc.in_features`是最后一层的输入的特征数量

`resnet = resnet.to(device)`把模型搬到GPU

## 2.3.损失函数和优化器

```python
# 交叉熵损失（适用于分类问题） 
criterion = nn.CrossEntropyLoss()

# 学习率，动量 
lr,momentum = 0.01,0.9

# 优化器 
optimizer = optim.SGD(resnet.parameters(), lr=lr, momentum=momentum)
```

这个momentum参数是干嘛的来着？好像是批量归一化？  

是的，是[批量归一化](https://www.bilibili.com/video/BV1X44y1r77r?spm_id_from=333.788.videopod.episodes&vd_source=8924ad59b4f62224f165e16aa3d04f00)  

## 2.4.训练模型

```python
# 单次训练
def train():
    resnet.train()
	
    batch_nums = len(train_loader)  # 批次数
    batch_size = train_loader.batch_size  # 批量大小
    size = len(train_loader.dataset)  # 数据集大小
    
    train_loss,correct = 0.0, 0.0 # 统计损失和准确率
	
    p = tqdm(train_loader, desc="Training", unit="batch")
    
    for X,y in p:
        X,y = X.to(device),y.to(device)#把数据搬到计算设备(例如cuda或cpu)上
        pred = resnet(X)#前向传播，使用前面定义好的resnet模型
        loss = criterion(pred,y)#损失函数计算损失
        loss.backward()#反向传播，更新梯度
        optimizer.step()#优化器更新参数模型
        optimizer.zero_grad()#清空梯度，防止梯度积累
        
        p.set_postfix(loss=f"{loss.item():>8f}")  # 显示损失值
        
        train_loss+=loss.item() # 累计每个批次的平均损失
        correct += (pred.argmax(1) == y).sum().item() # 计算正确预测的数量
	
    train_loss /= batch_nums
    correct /= size
    print(f"Train Accuracy: {(100*correct):>0.2f}%, Train Avg loss: {train_loss:>8f}")
	
    return train_loss,correct
```

这里第一次用到了`tqdm`函数，对可迭代对象调用这个函数，返回的值同样是一个可迭代对象，区别是新的可迭代对象被遍历时会有可视化的进度条

每次循环的X和y对应一个batch，`X.to(device)`和`y.to(device)`是把数据搬到GPU  

更多细节可以看代码里的注释

## 2.5.预测

### 2.5.1.预测函数

```python
# 验证
def test():
    resnet.eval() # 评估模式
    
    batch_nums = len(test_loader)  # 批次数
    batch_size = test_loader.batch_size  # 批量大小
    size = len(test_loader.dataset)  # 数据集大小
    
    test_loss,correct = 0.0, 0.0 # 统计损失和准确率

    with torch.no_grad():
        
        for X,y in test_loader:
            X,y = X.to(device),y.to(device)
            pred = resnet(X)
            loss = criterion(pred,y)

            test_loss+=loss.item() # 累计每个批次的平均损失
            correct += (pred.argmax(1) == y).sum().item() # 计算正确预测的数量

    test_loss /= batch_nums
    correct /= size
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Test Avg loss: {test_loss:>8f}")
    
    return test_loss,correct
```

预测和训练的一个区别是把`resnet.train()`改成了`resnet.eval()`

`Dropout：`在训练时会随机丢弃，评估时则不会

`BatchNormal：`训练时用当前批次的均值/方差，并更新全局运行统计；评估时则使用已累计的运行统计（不再更新）

### 2.5.2.进行训练，保存最佳模型

```python
# 训练损失和准确率
train_losses,train_accs = [],[]

# 测试损失和准确率
test_losses ,test_accs= [],[]

epochs = 20

best_acc = 0.0  # 记录最佳准确率
save_path = 'best_model.pth'  # 保存路径

for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    train_loss,train_acc = train()
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    test_loss,test_acc = test()
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    # 保存最好的模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(resnet.state_dict(), save_path)  # 仅保存状态字典
        print(f'New best model saved with accuracy: {best_acc:.4f}')

    print("-"*30)
```

20轮训练过程中模型的参数一直在改变，测试精度可能上上下下波动，`torch.save`把测试精度最高的模型保存下来了。

### 2.5.3.加载最优模型

```python
# 重新定义模型（确保架构一致）
resnet = models.resnet18()
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)

# 加载训练好的权重
resnet.load_state_dict(torch.load(save_path,weights_only=True))

# 切换到 eval 模式
resnet.to(device)
resnet.eval()
print("模型已加载并设置为评估模式！")
```

最优模型的路径保存在`save_path`中了

### 2.5.4.批量预测

```python
# 批量预测

class LeaveValDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(base_path, self.data.iloc[idx,0])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)  # 应用转换
        
        return image

val_dataset = LeaveValDataset(val_df,transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print("val batch num:",len(val_loader))
```

前面有一个`LeaveDataset`，这里有一个`LeaveValDataset`，它们使用了同一个`transform`，都是把图片从`Image.open`返回的格式转化张量

区别在于前者转换的是带标签的训练数据，后者转换的是不带标签的测试数据；转换训练数据时把字符串标签转换成了数字。

### 2.5.5.输出预测结果

```python
all_preds = []

with torch.no_grad():
    for inputs in tqdm(val_loader):
        inputs = inputs.to(device)
        outputs = resnet(inputs)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())

print("批量预测结果长度:", len(all_preds))

pred_labels = [idx2label[pred_id] for pred_id in all_preds] # 转换成label
val_df['label'] = pred_labels
val_df.to_csv('submission.csv',index=False)
```
