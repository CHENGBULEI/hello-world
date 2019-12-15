#include<reg51.h>
#define uchar unsigned char
#define P1 P1
#define P2 p2
uchar count=0;
void init(){
	TMOD = 0X09;//选用T0作为计数器
	TH0 = 0;
	TL0 = 0;
	TR0 = 1;//开始运行
	EX0 = 1; //   
	IT0 = 1;//开中断 
	EA = 1;//允许中断 
}

void main(){
	init();
	while(1);
} 
void timer0Int() interrupt 0 using 1
{
	//遇到高电平就开始下一次计数 
	count = TH0*256+TL0;//第一次计数时为0，后面几次都是正确的
	P1 =  TH0;
	P2 =  TL0;
	TH0 = 0;
	TL0 = 0;
}
