把屏蔽词换成*

编译运行命令
```shell
g++ -std=c++17 -o run test.cpp utf8.cpp
./run
```
此实现的时间复杂度是O(n),有一定hack空间(n是文本串长度)

例如模式串包括"111123"和"1"，文本串是"11111"，替换的结果是"\*111\*"

此仓库中的[字典树的应用](https://github.com/RaDsZ2z/algorithm/blob/main/AP%E4%BC%A0%E7%BB%9F%E8%89%BA%E8%83%BD/%E5%AD%97%E5%85%B8%E6%A0%91%E7%9A%84%E5%BA%94%E7%94%A8.md)可以全部替换，但是时间复杂度是O(n^2)

被检测的文本串一般都不长，在此前提下字典树的替换方式效果更好。但是另一个问题是字典树的检测方式屏蔽强度可能过于强了。
