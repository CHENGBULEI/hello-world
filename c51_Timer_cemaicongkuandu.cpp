#include<reg51.h>
#define uchar unsigned char
#define P1 P1
#define P2 p2
uchar count=0;
void init(){
	TMOD = 0X09;//ѡ��T0��Ϊ������
	TH0 = 0;
	TL0 = 0;
	TR0 = 1;//��ʼ����
	EX0 = 1; //   
	IT0 = 1;//���ж� 
	EA = 1;//�����ж� 
}

void main(){
	init();
	while(1);
} 
void timer0Int() interrupt 0 using 1
{
	//�����ߵ�ƽ�Ϳ�ʼ��һ�μ��� 
	count = TH0*256+TL0;//��һ�μ���ʱΪ0�����漸�ζ�����ȷ��
	P1 =  TH0;
	P2 =  TL0;
	TH0 = 0;
	TL0 = 0;
}
