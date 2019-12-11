#include<bits/stdc++.h>
using namespace std;
int main(){
	int i,j,k;
	int n;
	int piao;	
	cin>>n;
	
	vector<queue<int> >res;
	queue<int>tmp;
	int arc[100][6]={0};
	for(j=0;j<100;j++){
		arc[j][5]=5;
	}
	for(i=0;i<n;i++){
		cin>>piao;
		int flag=1;
		
		//第一种情况：可以找到全部放下的 
		for(j=0;j<100;j++){
			if(arc[j][5]>=piao){
		
				arc[j][5]-=piao;
				for(k=0;k<5&&piao>0;k++)
				{
					if(arc[j][k]==0)
					{
						piao--;
						arc[j][k]=1;
						tmp.push(k+1+j*5);	
					}
				}
				res.push_back(tmp);
				break;
			}
		}
		
		//第二种情况：只能分开放 
		if(j==100){
			for(j=0;j<100;j++){
				if(arc[j][5]>0&&piao>0){
					int goumai = 0;
					if(arc[j][5]>piao){
						goumai = piao;
						piao=0;
						arc[j][5] = arc[j][5]-=piao;
					}else{
						goumai = arc[j][5]-piao;
						piao -=arc[j][5];
						arc[j][5]=0;
					}
					for(k=0;k<5&&goumai>0;k++){
							if(arc[j][k]==0)
							{
									goumai--;
									tmp.push(j*5+k+1);
									arc[j][k]=1;	
							}
							
						}
					}	 
			}
			res.push_back(tmp);	
		}
		while(!tmp.empty()){
					tmp.pop();
				}
		
	}
	/*
	for(i=0;i<100;i++){
		for(j=0;j<6;j++)
		cout<<arc[i][j];
		
		cout<<endl;
	}*/ 
	for(i=0;i<n;i++){
		while(!res[i].empty()){
			cout<<res[i].front();
			//if(res[j].size()>1)
				cout<<" ";
			res[i].pop();
		}
		cout<<endl;
	}
	return 0;
} 
