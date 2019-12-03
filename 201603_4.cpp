#include<iostream>
#include<bits/stdc++.h>
using namespace std;
typedef struct EdgeNode{
	int id;//边顶点对应的下标 
	long weight;//边对应权重 
	EdgeNode* next;//指向下一个节点 
};
typedef struct Node{
	int id;//节点编号 
	EdgeNode *firstedge;//表头节点; 
}; 
typedef struct GraphyList{
	Node *adjlist;
	int vnum;
	int ennum;
};
int check(GraphyList &g,int i,int j);
long long dfs(GraphyList &g);//最短路径 
long long DFS(GraphyList &g); 

int main(){
	int n,m;
	cin>>n>>m;
	long long zuida=0;
	long long tmp=0;
	
	GraphyList g;
	g.adjlist =new Node[n+1];

	int i,j;
	int a,b,c;
	
	for(i=1;i<=n;i++)
	{
		g.adjlist[i].id =i;
		g.adjlist[i].firstedge =NULL;	
	}
	g.vnum = n;
	g.ennum = m;
	 	
	for(i=0;i<m;i++)
	{
		cin>>a>>b>>c;
		EdgeNode *p = new EdgeNode();
		p->id =b;
		p->weight =c;
		p->next =NULL; 
		if(g.adjlist[a].firstedge==NULL){
				g.adjlist[a].firstedge = p;
				
		}else{
			p->next = g.adjlist[a].firstedge;
			g.adjlist[a].firstedge = p;
			
		}
	}
	
	cout<<dfs(g); 
	return 0;
} 
long long DFS(GraphyList &g){
	//访问栈 
	stack<int>res;
	 //是否已经访问的记录数组
	int visit[g.vnum +1] = {0};
	int flag =0;
	int first=INT_MIN,second=INT_MIN;
	visit[1]=1;
	res.push(1);
	while(!res.empty()){
		int i = res.top();
		
		visit[i]=1;
		 
		cout<<i<<endl;
		EdgeNode *p = g.adjlist[i].firstedge;
		while(p!=NULL&&visit[p->id]==1){
			p = p->next ;
		}
		if(p!=NULL)
		{
			//求当前节点到下一个节点的距离，并比较当前路径的最大值 
			int itod = check(g,i,p->id);
			
			if(p->id==g.vnum)
			{
				flag++;				
			}
			if(flag==0)
				if(itod >first)first = itod;
			else{
				if(itod>second)second = itod;
			}
			res.push(p->id);
		}else{
			//回溯 
			res.pop();
		}
	}	 
	return min(first,second); 
} 

long long dfs(GraphyList &g){
	int i,j;
	pair<long long,int> *dis =new  pair<long long ,int>[(g.vnum+1)];
	//设置起点路径 
	dis[1].first=0;
	dis[1].second = 1;
	int count =1;
	for(i=2;i<=g.vnum ;i++){
		dis[i].second = 0;
		int toi = check(g,1,i);
	//	cout<<toi;
		if(toi!=-1){
			dis[i].first = toi;
		}else{
			dis[i].first = INT_MIN;
		}
	}
	while(dis[g.vnum].second==0){
		
		string s[g.vnum+1] ;
		
		int tmp = 1;//当前最小节点下标 
		int min = INT_MAX;
		
		for(i=1;i<=g.vnum ;i++){
			if(dis[i].second==0&&dis[i].first<min){
				min = dis[i].first;
				tmp = i;
			}
		}
		//cout<<tmp;
		dis[tmp].second = 1;
			
		for(i=1;i<=g.vnum ;i++){
			int tmptoi = check(g,tmp,i);
			//cout<<tmptoi;
			
			if(dis[i].second==0&&tmptoi!=-1&&(tmptoi)>dis[i].first)
			{
				if(i==g.vnum )
				dis[i].first = dis[i].first==INT_MIN?tmptoi:dis[i].first;
				else
				dis[i].first = tmptoi;
					
			}
		}
	}
	
	return dis[g.vnum].first;
	
}

int check(GraphyList &g,int i,int j){
		if(i>g.vnum||j>g.vnum ||i<=0||j<=0 )return -1;
		else{
			EdgeNode *p = g.adjlist[i].firstedge;
			while(p!=NULL&&p->id!=j){
				p = p->next ;
			}
			if(p==NULL)return -1;
			else return p->weight ;
		}
	}
	
	
