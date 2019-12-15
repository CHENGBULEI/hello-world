#include<bits/stdc++.h>
using namespace std;
typedef struct suicong{
	int x;
	int shenming;
	int gongjili;
};
class yinxiong{
	public :
		int shenming;
		int scnum;
		vector<suicong>sc;
	yinxiong(){
		this->shenming = 30;
		this->scnum = 0;
	}
	~yinxiong(){
	}
	void addsuicong(int &x,int &shenming,int &gongjili){
		suicong tmp;
		tmp.gongjili = gongjili;
		tmp.shenming = shenming;
		if(x>scnum){
			tmp.x = this->scnum;
			this->scnum++;
			sc.push_back(tmp);
		}else{
			int m,n;
			tmp.x =x-1;
			for(n=x-1;n<this->scnum;n++){
				this->sc[n].x++;
			}
			
			this->sc.insert(this->sc.begin()+x-1,tmp);
			this->scnum = this->sc.size();
			
			
		}
		
		/*
		vector<suicong>::iterator it;
		for(it = this->sc.begin();it!=this->sc.end();it++){
			cout<<(*it).shenming<<" "<<(*it).x;
		}
		*/
			
	}
	
	void attack(int &x,int& y,yinxiong &p){
		if(x!=0&&y!=0){
				x-=1;
			y-=1;
			//cout<<x<<" "<<y;
			this->sc[x].shenming -=this->sc[x].gongjili;
			p.sc[y].shenming -=this->sc[x].gongjili;
			if(this->sc[x].shenming<=0){
				this->deletsuicong(x);
			}
			if(p.sc[y].shenming<=0){
				p.deletsuicong(y); 
			}	
		}else{
			x-=1;
			this->sc[x].shenming -=this->sc[x].gongjili;
			p.shenming-=this->sc[x-1].gongjili;
			if(this->sc[x-1].shenming<=0){
				
				this->deletsuicong(x);
			}
		}
		
		
	}
	
	void deletsuicong(int &x){
		for(int i=x+1;i<this->scnum;i++){
			this->sc[i].x--;
		}
		this->sc.erase(this->sc.begin()+x);
		this->scnum--;
		
		/*
		vector<suicong>::iterator it;
		for(it = this->sc.begin();it!=this->sc.end();it++){
			cout<<(*it).shenming<<" "<<(*it).x;
		}
		*/
			
	}
	
};
int main(){
	int n;
	string s;
	int jieguo;

	int nowplay=0;
	cin>>n;
	yinxiong first;
	yinxiong second;
	
	int i,j;
	suicong tmp;
	cin.ignore();
	for(i=0;i<n;i++){ 
		getline(cin,s);
		
		if(s.find("summon")!=string::npos){
			int x = s[7]-'0';
			int shenming = s[11]-'0';
			int gongjili = s[9]-'0';
			//cout<<x<<endl;
			
			
			if(nowplay%2==0){
				first.addsuicong(x,shenming,gongjili); 
			}else{
				second.addsuicong(x,shenming,gongjili); 
			}
			
			
			//cout<<"^"<<endl;
		}else if(s.find("attack")!=string::npos){
			//cout<<"#"<<endl;
			
			
			int x = s[7]-'0';
			int y = s[9]-'0';
			if(nowplay%2==0){
				first.attack(x,y,second); 
			}else{
				second.attack(x,y,first); 
			}
		}else if(s=="end"){
			nowplay++;
		}
		if(second.shenming <=0){
			jieguo = 1;
			//cout<<"("<<endl;
			
			
			break;
		}else {
			if(first.shenming<=0){
				jieguo = -1;
				
				//cout<<"*"<<endl;
				
				break;
			}else{
				jieguo = 0;
			}
		}
		//cout<<s<<endl;
	}
	//cout<<i<<endl;
	cout<<jieguo<<endl;
	cout<<first.shenming<<endl;
	cout<<first.scnum<<" ";
	while(!first.sc.empty()){
		cout<<first.sc.front().shenming<<" ";
		first.sc.erase(first.sc.begin());
	}
	cout<<endl;
	cout<<second.shenming<<endl;
	cout<<second.scnum<<" ";
	while(!second.sc.empty())
	{
		cout<<second.sc.front().shenming<<" ";
		second.sc.erase(second.sc.begin());
	}
	return 0;
} 
