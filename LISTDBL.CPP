#include "ListDBL.h"

void InitList( struct List &L )
{
	 L.StartPos=new ListNode;
	 L.EndPos=L.StartPos;
	 L.CurPos=L.StartPos;
	 L.Length=0;
	 L.StartPos->next=L.StartPos;
	 L.StartPos->prior=L.StartPos;
	 cout<<"build !\n";
}

void DestroyList( List &L )
{
	while(1){
	if(L.EndPos==L.StartPos){
	delete L.StartPos;
	break;
	}
	else {
	L.EndPos=L.EndPos->prior;
	delete L.EndPos->next;
	}
	}
}

void insert( List &L, ElemType &newData )  
{

	ListNode *t=new ListNode;
	if(t==null){
		cout<<"error!\n";
			exit(-1);
	}

	t->data=newData;

		if(isFull(L)) {
			cout<<"the list is full!\n";
		}
		else {
			if(L.CurPos==L.EndPos){

				L.EndPos->next=t;
				t->prior=L.EndPos;
				L.EndPos=t;
				L.CurPos=t;
				t->next=L.StartPos;
				L.StartPos->prior=L.EndPos;
				L.Length++;
	
			}
			else{
			t->prior=L.CurPos;
			L.CurPos->next=t->next;
			L.CurPos->next=t;
			t->next->prior=t;
			L.CurPos=t;
			L.Length++;
			cout<<"success!\n";
		}
		}



}

int remove( List &L ) 
{	
	ListNode *p=L.CurPos;
	if(L.Length!=0) {
	L.CurPos->next->prior=L.CurPos->prior;
	L.CurPos=L.CurPos->prior;
	L.CurPos->next=p->next;
	L.Length--;
	return 1;
	}
	else {
	cout<<"the list is empty\n";
	return 0;
	}
}
void replace( List &L, ElemType &newDataItem )
{
	if(L.Length==0)
		exit(-1);
	else {
	L.CurPos->data=newDataItem;
	}
}

void clear( List &L ) 							  
{
	if(L.Length==0)
		cout<<"the list is empty\n";
	else {
		while (L.Length){
			if(L.EndPos!=L.StartPos) {
				L.EndPos=L.EndPos->prior;
				delete L.EndPos->next;
				L.Length--;
			}
			else {
			 L.Length=0;
			 break;
			}
			
		
		}

	}

}


bool isEmpty(List L) 						
{
	if(L.Length==0)
		return true;
	else
		return false;
}


bool isFull(List L) 						
{
	if(L.Length>100)  //最大为100个！
		return true;
	else
		return false;

}


void gotoBeginning(List &L) 				
{
	L.CurPos=L.StartPos->next;
	if(L.Length==0)
		cout<<"error!"<<endl;
}


void gotoEnd(List &L) 					
{
	L.CurPos=L.EndPos;
	if(L.Length==0)
		cout<<"error!"<<endl;
}


bool gotoNext(List &L)  //如果当前位置是结尾返回false,else返回true
{
		if(L.CurPos==L.EndPos)
		return false;
		else{
		L.CurPos=L.CurPos->next;
		cout<<L.CurPos->data<<"\n";
		return true;
		}

}
					
bool gotoPrior(List &L) //如果当前位置是开头返回false,else返回true
{
   if(isEmpty(L))
		return false;
   else { 
		
		if(L.StartPos->next==L.CurPos){
		return false;
		}
		else{
		L.CurPos=L.CurPos->prior;
		cout<<L.CurPos->data<<"\n";
		return true;
		}
   }
}

					
ElemType getCursor(List L)					
{
	if( L.Length==0 )
		exit(-1);
	else
		return L.CurPos->data;
}


void showStructure(List L)
{
	struct ListNode *p=L.StartPos->next;
	if( L.Length==0){
		cout << "Empty list\n";

	}
	else {
		for(int i=0;i<L.Length;i++){
			cout<< p->data<<" ";
				p=p->next;
		}
		cout<<endl;
	}
}

int  getLength( List L )
{
	return L.Length;
}

int  getCurPosition( List L )
{
	struct ListNode *p=L.StartPos->next;
	int i;
	for(i=0;i<L.Length;i++)
		if( p==L.CurPos )
			return i+1;
		else
			p=p->next;
}

 
void Shift_R( List &L )
{
	if(L.CurPos==L.EndPos)
		L.CurPos=L.EndPos->prior;

	ListNode *p=L.EndPos;
	L.EndPos=p->prior;
	L.EndPos->next=p->next;
	p->next->prior=L.EndPos;
	p->prior=L.StartPos;
	p->next=L.StartPos->next;
	L.StartPos->next->prior=p;
	L.StartPos->next=p;
	
}

void Shift_L( List &L )
{
	if(L.CurPos==L.StartPos->next)
		L.CurPos=L.CurPos->next;

		
	ListNode *p=L.StartPos->next;
	L.StartPos->next=p->next;
	p->next->prior=L.StartPos;
	L.EndPos->next=p;
	p->prior=L.EndPos;
	p->next=L.StartPos;
	L.StartPos->prior=p;
	L.EndPos=p;

}
void Reverse( List &L )
{
	int i;
	i=getCurPosition(L);
	L.CurPos=L.StartPos;
	do
	{
	L.StartPos=L.StartPos->next;
	L.StartPos->prior->next=L.StartPos->prior->prior;
	L.StartPos->prior->prior=L.StartPos;
	}while(L.StartPos!=L.CurPos);
	L.StartPos->next->next=L.EndPos->prior;
	L.EndPos->prior=L.StartPos;
	for(int t=0;t<i;t++)
		L.CurPos=L.CurPos->next;

}

void SwapItem( List &L )
{
	ListNode *p=L.EndPos;
	ListNode *q=L.StartPos->next;
	if(L.CurPos==p)
		L.CurPos=q;
	else if(L.CurPos==q)
		L.CurPos=p;
	L.StartPos->next=q->next;
	q->next->prior=L.StartPos;
	q->next=L.StartPos;
	q->prior=L.EndPos->prior;
	L.EndPos->prior->next=q;
	L.StartPos->prior=q;
	L.EndPos=q;
	p->next=L.StartPos->next;
	p->prior=L.StartPos;
	L.StartPos->next->prior=p;
	L.StartPos->next=p;
}

void showListData( List &L )
{
	ListNode *p=L.StartPos->next;
	for(int i=0;i<L.Length;i++)
	{	cout<<p->data;p=p->next;}
	cout<<"\n";
}

bool isEnd( List L )
{
	if(L.CurPos==L.EndPos)
	return true;
	else
	return false;
} 	
					
bool isHead( List L )
{
	if(L.CurPos==L.StartPos->next)
	return true;
	else
	return false;
} 						

void CreatList( List &L, char str[] )
{
	int i=0;
	while(str[i]!='\0'){
	insert(L,str[i]);
	i++;
	}


}

void ReplaceList( List &L, char str[] )
{
}


