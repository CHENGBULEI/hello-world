#include<reg51.h>
#define uchar unsigned char
sbit led = P1_0;
uchar count = 0;//中断次数计数器 
void init (){
	TMOD = 0X01;
	TH0 = (65536-50000)/256;
	TL0 = (65536-50000)%256;
	TR0 = 1;
	led = 1;//灯熄灭 
} 
void main(){
	uchar i = 0;
	init();
	while(1){
		TH0 = (65536-50000)/256;
		TL0 = (65536-50000)%256;
		//查询TF0的方式 
		while(!TF0){
			TF0 = 0;
			i++;
			if(i==20){
				led = ~led;
				i=0;
			}
		}
	}
	
	 
}
//中断的方式 
/*
void initTimer0(){
	TMOD = 0X01;
	TH0 = (65536-50000)/256;
	TL0 = (65536-50000)%256;
	ET0 = 1;//开T0 的中断 
	EA = 1;//运行中断 
	TR0 = 1;//T0开始运行 
}
 
void main(){
	initTimer0();
	while(1);
} 

void Timer0Int() interrupt 1 using 1
{
	TH0 = (65536-50000)/256;
	TL0 = (65536-50000)%256;
	count++;
	if(count==20){
		count = 0;
		led = ~led ;
		
	}
}
*/
