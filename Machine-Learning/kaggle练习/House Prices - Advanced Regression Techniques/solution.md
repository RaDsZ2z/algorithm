第一次实战无从下手，找了[这篇笔记](https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python/notebook)学习一下。这是一道房价预测的题目

# 1.So... What can we expect?

为了理解数据，我们可以创建一个`Excel`包含以下的列

+ 变量-变量名称
+ `type`-变量类型的标识。此字段有两个可能的值：“数值”或“分类”。“数值”是指其值为数字的变量，“分类”是指值为类别的变量
+ `segment`-变量段的标识。我们可以定义三个可能的部分：`building`、`space`或`location`。当我们说`building`时，我们指的是与建筑物的物理特性相关的变量（例如`OverallQual`）。当我们说`space`时，我们指的是报告房屋空间属性的变量（例如`TotalBsmtSF`）。最后，当我们说`location`时，我们指的是一个变量，它提供了房子所在位置的信息（例如`Neighborhood`）。
+ 预期-我们对“销售价格”变量影响的预期。我们可以使用“高”、“中”和“低”作为可能值的分类量表。
+ 结论-在快速查看数据后，我们得出了关于变量**重要性**的结论。我们可以使用与“期望”中相同的分类量表。
+ 评论-我们的任何一般性评论。

> 上面相当于预分析了数据的每个特征，预估了每个特征的重要程度。

> **OverallQual** :Overall Quality 房屋总体质量
>
> **TotalBsmtSF**: Total Basement Square Feet 地下室总面积
>
> **Neighborhood**:可能指房屋所在的社区或街区——不同社区的房价有明显差异

