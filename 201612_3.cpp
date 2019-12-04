#include<bits/stdc++.h>
using namespace std;
pair<string,vector<pair<string,int> > > juese(string &s);
pair<string,int>quanxian(string &s);
pair<string,vector<string> >yonghu(string &s);
int main(){
	int n;
	int i,j;
	//组织数据 
	map<string,vector<string> >yonghubiao;
	vector<pair<string,int> >quanxianbiao;
	vector<pair<string,vector<pair<string,int> > > >juesebiao;
	string s;
	vector<string>yhtmp;
	pair<string,int>tmp;
	tmp.first = "";
	tmp.second = -1;
	cin>>n;
	for(i=0;i<n;i++){
		cin.ignore();
		getline(cin,s);
		if(s.find(':')!=string::npos){
			for(j=0;j<s.size()&&s[j]!=':';j++){
				tmp.first+=s[j];
			}
			tmp.second = s[j+1]-'0';
		}
		quanxianbiao.push_back(tmp);
	}
	
	int q;
	cin>>q;
	for(i=0;i<q;i++){
		cin.ignore();
		getline(cin,s);
		juesebiao.push_back(juese(s));	
	}
	
	int yonghu;
	cin>>yonghu;
	for(i=0;i<yonghu;i++){
		cin.ignore();
		getline(cin,s);
		yonghubiao.insert(yonghu(s) );
	}
	
	int chaxun;
	cin>>chaxun;
	for(i=0;i<chaxun;i++){
		cin.ignore();
		getline(cin,s);
		string yh = "";
		string qx = "";
		int flag =0;
		j = s.find(' ');
		yh = s.substr(0,j);
		qx = s.substr(j+1,s.size());
		
		pair<string,int>tmpqx = quanxian(qx);
		
		
		if(yonghubiao.find(yh).second==false){
			cout<<"false"<<endl;
		}else{
			yutmp = yonghubiao.find(yh).second;
			int yutmplen = yutmp.size();
			
			
			//查询 
			for(j=0;j<yutmplen;j++){
				flag=0;
				for(int t=0;t<q;t++){
					if(juesebiao[t].first == yutmp[j]){
						for(int a =0;a<juesebiao[t].second.size();a++){
							if(tmpqx.first==juesebiao[t].second[t].first)
							{
								if(tmpqx.second==juesebiao[t].second[t].second&&tmpqx.second==-1)
								{
									cout<<"true"<<endl;
									flag=1;	
								}else{
									if(tmpqx.second==-1&&juesebiao[t].second[t].second>-1)
									{
										cout<<juesebiao[t].second[t].second<<endl;
										flag = 1;
	
									}
									else if(tmpqx.second<=juesebiao[t].second[t].second){
										cout<<"true"<<endl;
										flag=1;
									}									
								}
							}
						}
					}
				}
				if(flag==0)
				cout<<"false"<<endl;
			} 
		}
		
	}
	return 0;
} 

pair<string,vector<string> >yonghu(string &s){
	pair<string,vector<string> >res;
	string t = "";
	vector<string>yh;
	char *item = new [s.size()+1];
	const char * delim = " ";
	strcpy(item,s.c_str());
	
	item = strtok(item,delim);
	int j=0;
	while(item!=NULL){
		string tmp = item;
		if(j==0){
			t = tmp;
		}else if(j>=2){
			res.second.push_back(tmp);
		}
		j++;
	}
	res.first = t;
	return res;
	
}
pair<string,vector<pair<string,int> > > juese(string &s){
		char *item = new [s.size()+1];
		const char *delim = " ";
		strcpy(item,s.c_str());
		
		int j=0;
		string juese ;
		pair<string,int>tmp;
		vector<pair<string,int> >res;
		
		item = strtok(item,delim);
		while(item!=NULL){
			string t  = item;
			if(j==0){
				juese  = t;
			}else if(j>=2){
				res.push_back(quanxian(t));
			}
			j++;
		}
		return make_pair(juese,res);
}


pair<string,int>quanxian(string &s){
	if(s.find(':')==string::npos){
		return make_pair(s,-1); 
	}else{
		string t = "";
		for(int i=0;i<s.size()&&s[i]!=':';i++){
			t+=s[i];
		}
		int level = s[s.size()-1]-'0';
		return make_pair(t,level);
		
	}
}
