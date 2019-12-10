#include<vector>
#include<map>
#include<iostream>
#include<algorithm>
#include<stdlib.h>
using namespace std;
int main(){
	int i,j;
	int x,y;
	int n,m;
	cin>>n;
	cin>>m;
	vector<int>res(n,0);
	for(i=0;i<n;i++)
	res[i]=i+1;
	
	vector<int>::iterator it;
	vector<int>::iterator iter;
	for(i=0;i<m;i++){
		cin>>x>>y;
		it = find(res.begin(),res.end(),x);
		int local = it-res.begin();
		if(y>0){
			int tmp =(*it);
			for(j=local;j<local+y;j++)
			res[j]=res[j+1];
			res[j]=tmp;
		}else{
			int tmp = (*it);
			for(j=local;j>local+y;j--){
				res[j]=res[j-1];
			}
			res[j]=tmp;
		}
		
	}

	while(!res.empty() ){
		cout<<res.front();
		if(i!=n-1)
		cout<<" ";
		res.erase(res.begin());
	}
	return 0;
}
