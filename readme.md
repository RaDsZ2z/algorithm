# 关于anaconda

在mac上安装很顺利，每次打开都有图形化界面，选择 `jupyter Note book(Web-based)`用起来很方便

但是第一次在windows上安装时只有命令行，输入

```shell
anaconda-navigator
```

之后 出现了上述图形化界面

（另外，`kaggle`页面也提供了类似`jupyter`的代码编辑界面，可以不在本地计算机安装任何东西）

[一些参考资料](https://zh.d2l.ai/)

# 其它

```shell
python3 -m venv path/to/venv #创建一个虚拟环境
source path/to/venv/bin/activate #激活虚拟环境
python3 -m pip install xyz #在虚拟环境中安装包

# 我电脑上的虚拟环境安装在～这个地方了，也就是
python3 -m venv ～/path/to/venv #创建虚拟环境
source ~/path/to/venv/bin/activate #激活虚拟环境
```
使用vscode 将md文件导出pdf时数学公式无法正常渲染，在md文件末尾加上

(vscode插件:Markdown PDF)
```txt
\`\`\`html
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>
\`\`\`
```
分页符
```html
<div STYLE="page-break-after: always;"></div>
```
公式中加百分号要在前面加上两个反斜杠"\"
```md
$11.4\\%$
```
$11.4\\%$  

24.12.8  

`anaconda`是真好用

