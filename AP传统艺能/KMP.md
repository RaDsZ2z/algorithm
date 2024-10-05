这个算法学了无数次，每次新学都忘得比较干净

这次要学AC自动机，又拿出来学了一下

[洛谷题解](https://www.luogu.com.cn/problem/solution/P3375)

[oi-wiki](https://oi-wiki.org/string/kmp/)

[模板题](https://www.luogu.com.cn/problem/P3375)

```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;
vector<int> get_pi(const string &s)
{
    int len = s.size();
    vector<int> pi(len);
    for (int i = 1; i < len; i++)
    {
        int j = pi[i - 1];
        while (j > 0 && s[j] != s[i])
            j = pi[j - 1];
        if (s[i] == s[j])
            j++;
        pi[i] = j;
    }
    return pi;
}
int main()
{
    // freopen("in.txt", "r", stdin);
    // freopen("out.txt", "w", stdout);
    string s1, s2;
    cin >> s1 >> s2;
    vector<int> pi = get_pi(s2);
    int len1 = s1.size();
    int len2 = s2.size();
    int j = 0;
    for (int i = 0; i < len1; i++)
    {
        while (j > 0 && s1[i] != s2[j])
            j = pi[j - 1];

        if (s1[i] == s2[j])
            j++;
        if (j == len2)
        {
            cout << i - len2 + 2 << '\n';
            j = pi[len2 - 1];
        }
    }
    for (int i : pi)
        cout << i << ' ';
}
```
