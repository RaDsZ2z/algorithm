此竞赛来自[李沐课程](https://www.bilibili.com/video/BV1z64y1o7iz/?spm_id_from=333.1387.upload.video_card.click&vd_source=8924ad59b4f62224f165e16aa3d04f00)，[竞赛地址](https://www.kaggle.com/c/classify-leaves)

**下载数据**

点击下载会跳转到Rules界面，点击`Late Submission`同意协议之后才可以下载  

数据文件夹是`classify-leaves`  

其中`images`里是很多张叶子的png图片，标号0到27152，`train.csv`有训练集中每片叶子的label，标号0到18352，`test.csv`是测试集中叶子的文件名，标号18353到27152



使用`DataLoader`加载数据时，参数`num_workers`要先设为0，这个参数值过大会让程序直接停滞

**开始训练**

使用的是resnet，第一次跑完测试精度只有**35%**左右，做了如下改动

```python
optimizer = torch.optim.SGD(net.parameters(),lr=lr) #小批量随机梯度下降
#改为
momentum = 0.9
optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=momentum) #小批量随机梯度下降
```

我不清楚这个参数对应的技巧理论，或许看一下李沐《动手学深度学习》第28到30节会有答案

修改之后观察到训练过程中测试精度产生了很大的震荡，最后的结果是**75%**，再加上权重衰退试试

```python
momentum = 0.9
optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=momentum) #小批量随机梯度下降
```

权重衰退对应的理论我也不记得了，可以复习第12节教程

加上权重衰退后，测试精度的震荡幅度依然很大，最终结果是**84%**。

**上面三个结果都只跑了一次**

感觉目前我这个网络是不太work的，主要是震动幅度太大，需要继续优化(有时间先多跑几次，看看最终测试精度的波动程度)



[参考](https://www.kaggle.com/code/wangdark/classify-leaves-resnet)
