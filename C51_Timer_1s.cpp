#include<reg51.h>
#define uchar unsigned char
sbit led = P1_0;
uchar count = 0;//�жϴ��������� 
void init (){
	TMOD = 0X01;
	TH0 = (65536-50000)/256;
	TL0 = (65536-50000)%256;
	TR0 = 1;
	led = 1;//��Ϩ�� 
} 
void main(){
	uchar i = 0;
	init();
	while(1){
		TH0 = (65536-50000)/256;
		TL0 = (65536-50000)%256;
		//��ѯTF0�ķ�ʽ 
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
//�жϵķ�ʽ 
/*
void initTimer0(){
	TMOD = 0X01;
	TH0 = (65536-50000)/256;
	TL0 = (65536-50000)%256;
	ET0 = 1;//��T0 ���ж� 
	EA = 1;//�����ж� 
	TR0 = 1;//T0��ʼ���� 
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
