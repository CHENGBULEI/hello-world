#include<bits/stdc++.h>
using namespace std;
typedef struct node{
	int x;
	int y;
	int weixian;
	int visit;
};
int main(){
	int n,m,t;
	cin>>n>>m>>t;
	vector<vector<node> >res(n+1,vector<node>(m+1,node()));
	int i,j;
	int a1,a2,x,y;
	for(i=1;i<=n;i++)
		for(j=1;j<=m;j++){
			res[i][j].x = 0;
			res[i][j].y = 0;
			res[i][j].weixian = 0;
			res[i][j].visit = 0;
		}
	for(i=0;i<t;i++){
		cin>>a1>>a2>>x>>y;
		res[a1][a2].x = x;
		res[a1][a2].y = y;
	}
	vector<int>visit(4,0);
	int flag=1;
	stack<pair<int,int> >stk;//��¼��ǰ·�� 
	
	int timer=1;//������ 
	while(flag&&!stk.empty()){
		pair<int,int> now = stk.top();
		int zx,zy;
		//��������Ƿ��ǿ����ߵ� 
		if((now.x-1)>=0&&(now.y)>=0){
			zx = now.x -1;
			zy = now.y;
			
					
			if(res[zx][zy].x<=timer&&res[zx][zy]>=timer||res[zx][zy].visit);
			else{
					//��������߾ͽ���ѹջ 
					stk.push(make_pair(zx,zy));
					timer++;
					res[zx][zy].visit=1;
					break;
				}
				
		}else if((now.x>=0)&&(now.y-1)>=0){
			zx = now.x;
			zy = now.y -1;
		}else if((now.x))	
		
		
	}
	return 0;
}
