
[胡津铭老师_实用机器学习](https://www.bilibili.com/video/BV1Gw4m1i7ys/?spm_id_from=333.788&vd_source=8924ad59b4f62224f165e16aa3d04f00)

[参考资料](https://zh.d2l.ai/)

mac上python虚拟环境的搭建:
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

anaconda是真好用

