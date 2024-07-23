2024年春招之际，马上要笔试了，但是发现自己AP(~~acm player~~)传统艺能已经忘了一些了

需要复习，顺便在这里记录一下

可能也会在这里记录一些后续的传统算法学习 ~~甚至机器学习~~？

2024.6.15开始跟[胡津铭老师](https://www.bilibili.com/video/BV1Gw4m1i7ys/?spm_id_from=333.788&vd_source=8924ad59b4f62224f165e16aa3d04f00)学machine-learning了

2024.6.21 发现了[参考资料](https://zh.d2l.ai/)

mac上python虚拟环境的搭建:
```shell
python3 -m venv path/to/venv #创建一个虚拟环境
source path/to/venv/bin/activate #激活虚拟环境
python3 -m pip install xyz #在虚拟环境中安装包

# 我电脑上的虚拟环境安装在～这个地方了，也就是
python3 -m venv ～/path/to/venv
source ~/path/to/venv/bin/activate
```
使用vscode 将md文件导出pdf时无法正常渲染公式，在md文件末尾加上

(vscode插件:Markdown PDF)
```html
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>
```
公式中加百分号要在前面加上两个反斜杠"\"
```md
$11.4\\%$
```
$11.4\\%$
