#include<reg51.h>
#define Fre 10000
#define uint8 unsigned char
sbit out = P1_0;
uint8 ct = 20;
void timer_init(){
	TMOD = 0x20;
	TH1 = 206;
	TL1 = TH1;
	
	ET1 = 1;
	EA  = 1;
	TR1 = 1;
}  

void main(){
	timer_init();
	while(1);
}

void Timerint() interrupt 3 using 1
{
	out  =~out;//
}
