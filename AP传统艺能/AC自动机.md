AC自动机的前置知识是[KMP算法](https://github.com/RaDsZ2z/algorithm/blob/main/AP%E4%BC%A0%E7%BB%9F%E8%89%BA%E8%83%BD/KMP.md)和[字典树](http://magic.vicp.io/oi-wiki/string/trie/)

下面是三道题

[AC自动机(简单版)](https://www.luogu.com.cn/problem/P3808)

```cpp
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include <queue>
using namespace std;
struct node {
	int son[26];
	int fail;
	int cnt;
}tree[1000005];
int cnt = 0;
void build(const string& s) {
	int p = 0;
	for (char c : s) {
		int x = c - 'a';
		if (tree[p].son[x] == 0)
			tree[p].son[x] = ++cnt;
		p = tree[p].son[x];
	}
	tree[p].cnt++;
}
void init() {
	
	queue<int> q;
	for (int i = 0; i < 26; i++) {
		if (tree[0].son[i] != 0) {
			q.push(tree[0].son[i]);
		}
	}
	
	tree[0].fail = 0;
	while (!q.empty()) {
		int p = q.front();
		q.pop();
		for (int i = 0; i < 26; i++) {
			if (tree[p].son[i]) {
				tree[tree[p].son[i]].fail = tree[tree[p].fail].son[i];
				q.push(tree[p].son[i]);
			}
			else {
				tree[p].son[i] = tree[tree[p].fail].son[i];
			}
		}
	}
}
int query(const string&s) {
	int p = 0, res = 0;
	for (const char& c : s) {
		int x = c - 'a';
		p = tree[p].son[x];
		for (int t = p; t and tree[t].cnt != -1; t = tree[t].fail) {
			res += tree[t].cnt;
			tree[t].cnt = -1;
		}
	}
	return res;
}
int main()
{
	//freopen("in.txt", "r", stdin);
	int n; cin >> n;
	string s;
	for (int i = 0; i < n; i++) {
		cin >> s;
		build(s);
	}
	init();
	cin >> s;
	cout << query(s);
	return 0;
}
```
[AC自动机(简单版Ⅱ)](https://www.luogu.com.cn/problem/P3796)

[【模板】AC自动机](https://www.luogu.com.cn/problem/P5357)
