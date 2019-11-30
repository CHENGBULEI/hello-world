#include<bits/stdc++.h>
#include<stdlib.h>
using namespace std;
int main(){
	char c;
	string s="";
	priority_queue<int,vector<int>,greater<int> >res;
	while((c=cin.get())!='.'){
		if(c==' ')
		{
			if(s== "first")res.push(1);
			else if	(s=="two")res.push(4);
			else if(s=="three")res.push(9);
			else if(	s=="four")res.push(16);
			else if(s== "five")res.push(25);
			else if(s=="six")res.push(36);
			else if(s== "seven") res.push(49);
			else if(s=="eight") res.push(64);
			else if(s=="nine")res.push(81);
			else if(s=="eleven")res.push(21);
			else if(s=="twelve")res.push(44);
			else if(s=="thirteen")res.push(69);
			else if(s=="fourteen")res.push(14*14%100);
			else if(s=="fifteen")res.push(15*15%100);
			else if(s=="sixteen")res.push(16*16%100);
			else if(s=="seventeen")res.push(17*17%100);
			else if	(s=="eighteen")res.push(18*18%100) ;
			else if(s=="nineteen")res.push(19*19%100);
			
			s.clear();
		}
		else{
			s+=c;
		}
	}
	if(res.empty())
	cout<<0;
	while(!res.empty())
	{
		cout<<res.top();
		res.pop();
	}
	return 0;
} 
