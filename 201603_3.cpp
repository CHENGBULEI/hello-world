#include<string>
#include<vector>
#include<stdlib.h>
#include<iostream>
#include<cctype>
#include<fstream> 
using namespace std;
int main(){
	int i,j;
	string tmp="";
	string s="";
	vector<vector<string> >res;
	vector<string>line;
	//fstream inStream;
	//inStream.open("C:\\Users\\asus\\Desktop\\in.txt");
	while((getline(cin,s))){
		if(s.size()!=0)
		line.push_back(s) ;
		else
		{
		res.push_back(line);
		line.clear();
		}		
	}
	res.push_back(line);
	
	int len = res.size();
	for(i=0;i<len;i++){
		//cout<<res[i][0];
		if(res[i][0][0]!='#'&&res[i][0][0]!='_'&&res[i][0][0]!='*')
		{
		s = "<p>";
		tmp = "</p>"; 
		res[i][0] = s+res[i][0];
		res[i][res[i].size()-1]+=tmp;	
		}
		else{
			if(res[i][0][0]=='#'){
				char count=0;
				int flag=1;
				int j=0;
				while(j<res[i][0].size()&&flag==1){
					if(res[i][0][j]!='#')flag=0;
					else count+=1;
					++j;
				}
				if(count>1){
					s = "<h";
					s +=count+'0';
					s +=">";
					s+=res[i][0].substr(count);
					tmp = "</h";
					tmp +=count+'0';
					tmp +=">"; 
					s+=tmp;
					swap(res[i][0],s);	
				}else{
					res[i][0].erase(res[i][0].begin());
					s = "<h>";
					tmp  = "</h>";
					res[i][0]=s+res[i][0];
					res[i][res[i].size()-1]+=tmp;
				}
				
			}else if(res[i][0][0]=='*'){
				int t=0;
				while(t<res[i].size()){
					res[i][t].erase(res[i][t].begin());
					s  = "<li>";
					tmp = "</li>";
					res[i][t] = s + res[i][t];
					res[i][t]+=tmp;
					++t;
				}
				
				string r="<u>";
				string rr="</u>";

				res[i].insert(res[i].begin(),r);
				res[i].push_back(rr);
			}
		}
	
	}
	len  = res.size();
	for(i=0;i<len;i++){
		while(!res[i].empty()){
			if(i!=len-1||res[i].size()!=1)
			cout<<res[i].front()<<endl;
			else
			cout<<res[i].front();
			res[i].erase(res[i].begin());
		}
	}
	//inStream.close();
	return 0;
}
