倍增求最大公共祖先

需要预处理出每个节点的深度，距离为 2^i 的父节点，Log数组

[洛谷练习题](https://www.luogu.com.cn/problem/P3379)
```cpp
#include <iostream>
#include <vector>
using namespace std;
const int N = 500005;
vector<int> e[N];
int dep[N];//每个节点的深度
int fa[N][23];//第二维要大于Log[n] 
int Log[N];
void dfs(int now,int from){
	fa[now][0] = from;
	dep[now] = dep[from]+1;
	for(int i=1;i<=Log[dep[now]];i++)
		fa[now][i] = fa[fa[now][i-1]][i-1];
	for(int i:e[now]){
		if(i==from)continue;
		dfs(i,now);
	}
}
int lca(int x,int y){
	if(dep[x]<dep[y])swap(x,y);
//	cout<<"x:"<<x<<" y:"<<y<<'\n';
//	cout<<Log[dep[x]-dep[y]]-1<<'\n';
	while(dep[x]>dep[y])
		x=fa[x][Log[dep[x]-dep[y]]];
//	cout<<"x:"<<x<<'\n';
	if(x==y)return x;
	
	for(int k=Log[dep[x]];k>=0;k--){
		if(fa[x][k]!=fa[y][k]){
			x=fa[x][k];
			y=fa[y][k];
		}
	}
	return fa[x][0];
}
int main(){
//	freopen("in.txt","r",stdin);
	cin.tie(0);cout.tie(0);
	int n,m,root;
	cin>>n>>m>>root;
	int x,y;
	for(int i=1;i<n;i++){
		cin>>x>>y;
		e[x].push_back(y);
		e[y].push_back(x);
	}
	for(int i=2;i<=n;i++)
		Log[i]=Log[i/2]+1;
	dfs(root,0);
	while(m--){
		cin>>x>>y;
		cout<<lca(x,y)<<'\n';
	}
	
}
```
