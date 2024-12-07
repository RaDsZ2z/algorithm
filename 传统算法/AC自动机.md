AC自动机的前置知识是[KMP算法](https://github.com/RaDsZ2z/algorithm/blob/main/AP%E4%BC%A0%E7%BB%9F%E8%89%BA%E8%83%BD/KMP.md)和[字典树](http://magic.vicp.io/oi-wiki/string/trie/)

[AC自动机 oi-wiki](https://oi-wiki.org/string/ac-automaton/)

下面是三道题

[AC自动机(简单版)](https://www.luogu.com.cn/problem/P3808)

给出多个模式串和一个文本串，问有多少个模式串在文本串里出现过
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

给出多个模式串和一个文本串，问**哪些**模式串在文本串中出现的次数最多(保证没有两个相同的模式串，若两个模式串出现次数相同，按输入顺序输出)
```cpp
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include <queue>
#include <algorithm>
using namespace std;
const int N = 1000005;
struct node {
	int son[26];
	int ind;
	int fail;
}tree[N];
struct node_ {
	int num;
	int pos;
	bool operator<(node_ a) {
		if (num != a.num)return num > a.num;
		return pos < a.pos;
	}
}ans[N];
string s[N];
int cnt = 0;
void clear(int x) {
	for (int i = 0; i < 26; i++) tree[x].son[i] = 0;
	tree[x].fail = 0;
	tree[x].ind = 0;
}
void build(const string& s,int ind) {
	int p = 0;
	for (const char& i : s) {
		int x = i - 'a';
		if (tree[p].son[x] == 0) {
			tree[p].son[x] = ++cnt;
			clear(cnt);
		}
		p = tree[p].son[x];
	}
	tree[p].ind = ind;
}
void init() {
	queue<int> q;

	for (int i = 0; i < 26; i++) {
		if (tree[0].son[i]) {
			//tree[tree[0].son[i]].fail = 0;
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
void query(const string&s) {
	int p = 0;
	for (const char& i : s) {
		int x = i - 'a';
		p = tree[p].son[x];
		for (int t = p;t ;t = tree[t].fail) {
			ans[tree[t].ind].num++;
		}
	}
}
int main() {
	//freopen("in.txt", "r", stdin);
	int n;
	while (true) {
		 cin >> n;
		 cnt = 0;
		 clear(0);
		 if (!n)break;
		 for (int i = 1; i <= n; i++) {
			 cin >> s[i];
			 ans[i].num = 0;
			 ans[i].pos = i;
			 build(s[i], i);
		 }
		 init();
		 cin >> s[0];
		 query(s[0]);
		 sort(ans + 1, ans + 1 + n);
		 cout << ans[1].num << '\n';
		 cout << s[ans[1].pos] << '\n';
		 for (int i = 2; i <= n; i++) {
			 if (ans[i].num == ans[1].num) {
				 cout << s[ans[i].pos] << '\n';
			 }
			 else {
				 break;
			 }
		 }
	}
}
```

[【模板】AC自动机](https://www.luogu.com.cn/problem/P5357)

给出一个文本串和多个模式串，分别求出每个模式串在文本串中的出现次数。（数据不保证任意两个模式串不相同）
```cpp
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include <queue>
using namespace std;
const int N = 200005;
struct node {
	int son[26];
	int fail;
	int ind;
	int res;
}tree[N];
int cnt = 0;
int in[N], firstIndex[N], res[N];
void build(const string&s, int ind) {
	int p = 0;
	for (const char& i : s) {
		int x = i - 'a';
		if (!tree[p].son[x]) {
			tree[p].son[x] = ++cnt;
		}
		p = tree[p].son[x];
	}
	if (!tree[p].ind)tree[p].ind = ind;
	firstIndex[ind] = tree[p].ind;
}
void init() {
	queue<int> q;
	for (int i = 0; i < 26; i++) {
		if (tree[0].son[i]) {
			q.push(tree[0].son[i]);
		}
	}
	tree[0].fail = 0;
	while (!q.empty()) {
		int u = q.front();
		q.pop();
		for (int i = 0; i < 26; i++) {
			if (tree[u].son[i]) {
				tree[tree[u].son[i]].fail = tree[tree[u].fail].son[i];
				in[tree[tree[u].fail].son[i]]++;
				q.push(tree[u].son[i]);
			}
			else {
				tree[u].son[i] = tree[tree[u].fail].son[i];
			}
		}
	}
}
void query(const string &s) {
	int p = 0;
	for (const char&i : s) {
		int x = i - 'a';
		p = tree[p].son[x];
		tree[p].res++;
	}
}
void topu() {
	queue<int> q;
	for (int i = 1; i <= cnt; i++)if (in[i] == 0)q.push(i);
	while (!q.empty()) {
		
		int u = q.front();
		
		q.pop();
		res[tree[u].ind] = tree[u].res;
		int v = tree[u].fail;
		in[v]--;
		tree[v].res += tree[u].res;
		if (!in[v])q.push(v);
	}
}
int main()
 {
	//freopen("in.txt", "r", stdin);
	int n; cin >> n;
	string s;
	for (int i = 1; i <= n; i++) {
		cin >> s;
		build(s, i);
	}
	init();
	cin >> s;
	query(s);
	topu();
	for (int i = 1; i <= n; i++)cout << res[firstIndex[i]]<<'\n';
	return 0;
}
```
