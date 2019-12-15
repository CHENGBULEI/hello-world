#include <bits/stdc++.h>
using namespace std;
bool check(int k,int lie,int arc[][10],int mokuai[][4]);
int main(){
	int arc[15][10]={0};
	int mokuai[4][4]={0};
	int lie; 
	int i,j;
	int t;
	for(i=0;i<15;i++)
		for(j=0;j<10;j++){
			cin>>t;
			arc[i][j]=t;
		}
		
	for(i=0;i<4;i++)
		for(j=0;j<4;j++){
			//cin>>(mokuai[i][j]);
			cin>>t;
			mokuai[i][j]=t;
		}
	cin>>lie;
	/*
	for(int m=0;m<15;m++){
		for(int n=0;n<10;n++){
			cout<<arc[m][n]<<" ";
			 
		}
		cout<<endl;
	}
	*/
	lie-=1;
	i =3;//从第三行开始 
	int flag =1;//结束标志为 
	while(flag){
		bool t = check(i,lie,arc,mokuai);
		//	cout<<t<<" "<<i<<endl;
		//cout<<"&"<<endl;
		if(t)
			i++;
		else
			flag = 0;
	/*
		for(int m=0;m<15;m++){
		for(int n=0;n<10;n++){
			cout<<arc[m][n]<<" ";
			 
		}
		cout<<endl;
	}
	*/	
	}
	i--;
	if(i>14){
		
		for(int m=0;m<18-i;m++){
			for(int n=0;n<4;n++){
				arc[14-m][lie+n]=arc[14-m][lie+n]+mokuai[17-i-m][n];	
			}
		}
	}else{
		for(int m=0;m<4;m++){
			for(int n=0;n<4;n++){
				arc[i-m][lie+n] = arc[i-m][lie+n]+mokuai[3-m][n];
			}
		}
	}
	for(int m=0;m<15;m++){
		for(int n=0;n<10;n++){
			cout<<arc[m][n]<<" ";
			 
		}
		cout<<endl;
	}
	return 0;
} 
bool check(int k,int lie,int arc[][10],int mokuai[][4]){
	int i,j,a;
		if(k>14){
			//判断最下面一层是否可以放下 
			for(i=0;i<4;i++)
				if(mokuai[18-k][i]==1)return false;
			j = 18-k;
			//判断是否有冲突 
			for(i=j-1;i>=0;i--){
				for(a = 0;a<4;a++){
					if(arc[14-i][a+lie]==1&&mokuai[j-i-1][a]==1)return false;
				}
			}
		}else{
			//直接判断是否有冲突 
			for(int i=0;i<4;i++)
				for(j=0;j<4;j++){
					if(arc[k-i][j+lie]==1&&mokuai[3-i][j]==1)return false;
				}	
		}
		return true;		
}
