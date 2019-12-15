#include<bits/stdc++.h>
using namespace std;
int main(){
	int shu[1000]={0};
	int apple[1000];
	
	int sum;
	int n,m,T=0,D=0,E=0;
	cin>>n;
	int i,j;
	
	for(i=0;i<n;i++){
		cin>>m;
		for(j=0;j<m;j++)
		cin>>apple[j];
		
		sum=apple[0];
		for(j=1;j<m;j++){
			if(apple[j]>0){
				if(apple[j]<sum){
					shu[i]=1;
					sum = apple[j];
				}
			}else{
				sum+=apple[j];
			}
			
		}
		T+=sum;
	}
	j=0;
	int flag=0;
	for(i=0;i<n+2;i++){
		if(shu[j]>0){
			flag++;
		}else
		{
			flag=0;
			}
		if(i<n&&shu[j]>0)D++;
		if(flag>=3){
			E++;
		}
		j++;
		j = j%n;
	}
	cout<<T<<" "<<D<<" "<<E;
	return 0;
}
