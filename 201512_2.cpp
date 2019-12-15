#include<bits/stdc++.h>
using namespace std;
void checkI(int i,int j,vector<vector<int> >&arc,int flag);
int main(){
	int n,m;
	cin>>n>>m;
	int i,j;
	vector<vector<int > > arc(n,vector<int>(m,0));
	for(i=0;i<n;i++)
		for(j=0;j<m;j++){
			cin>>arc[i][j];
		}
	for(j=0;j<m;j++)
		for(i=0;i<n-2;i++)
		{
			checkI(i,j,arc,1);
		} 
	
	
	for(j=0;j<n;j++)
		for(i=0;i<m-2;i++)
		{
			checkI(j,i,arc,0);
			
		} 
	
	
	for(i=0;i<n;i++)
	{
		for(j=0;j<m;j++)
		{
			if(arc[i][j]>0)
				cout<<arc[i][j];
			else
				cout<<0;
				cout<<" ";		
			}	
		cout<<endl;
	}
		
	return 0;
}
void checkI(int i,int j,vector<vector<int> >&arc,int flag){
	int count=0;
	int k,w; 
	int n = arc.size();
	int m = arc[0].size();
	if(flag){
		for(k=i+1;k<n;k++){
			if((arc[i][j]+arc[k][j]==0)||(arc[i][j]==arc[k][j])){
					count++;
				}else{
					break;
				}
			}
			if(count>=2){
				//cout<<"&"<<endl;
				int tmp =arc[i][j]<0?arc[i][j]:-arc[i][j];
				for(k=i;k<=i+count;k++){
					arc[k][j] = tmp;
				}
			}
		
			
	}else{
		for(k=j+1;k<m;k++){
			if((arc[i][j]+arc[i][k]==0)||(arc[i][j]==arc[i][k])){
					count++;
				}else{
					break;
				}
			}
			if(count>=2){
				//把所有相同元素都赋值为负数 
				//cout<<"*"<<endl;
				int tmp = arc[i][j]<0?arc[i][j]:-arc[i][j];
				for(k=j;k<=j+count;k++){
					arc[i][k] = tmp;
				}
			}
		
	}
		
}
