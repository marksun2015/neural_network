#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>  
#include <unistd.h>
#include "weight.h"

#define msleep(x) usleep(x*1000)
#define FX(x) ( 1.0/(1.0 + exp(-(x))) )    //Sigmoid funtion;
#define F1(x) (x)*(1.0-x)               //A derivative function of sigmoid function
#define N 8                              //numbers of hidden layer 
//#define M 280   


/**********************************
p: trainning number 
x: input value
d_p: targer output
net_h: hidden layer input

out_h: hidden layer out value
net_o: output layer out value
out_o: real out value 

err_h: hidden layer deviation
err_o: output layer deviation
w_ij: the weight between input layer and hidden layer 
w_jk: the weight between hidden layer and output layer 
eta: speed
mse: mean squared error  均方誤差
***********************************/

int main(int argc, char** argv) 
{ 
	int i,j,k,num,p=10;
	int x[10][15],d_o[10][4];
	float net_h[8],out_h[8],net_o[4],out_o[4];
	float err_h[8],err_o[4];
	float eta=0.2;
	//float eta=0.4;
	float mse[10];
	//----------------------------------------------------

	FILE *fp;
	if((fp=fopen("backprop.in","r"))==NULL){
		printf("open in error \n");
		return 1;
	}

	for(i=0;i<p;i++){
		fscanf(fp,"%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d",
				&x[i][0],&x[i][1],&x[i][2],&x[i][3],&x[i][4],
				&x[i][5],&x[i][6],&x[i][7],&x[i][8],&x[i][9],
				&x[i][10],&x[i][11],&x[i][12],&x[i][13],&x[i][14],
				&d_o[i][0],&d_o[i][1],&d_o[i][2],&d_o[i][3]);

		#if 0
		printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n",
				x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],
				x[i][5],x[i][6],x[i][7],x[i][8],x[i][9],
				x[i][10],x[i][11],x[i][12],x[i][13],x[i][14],
				d_o[i][0],d_o[i][1],d_o[i][2],d_o[i][3]);

		#endif
	}

	fclose(fp);

	for(p=0;p<10;p++)
	{
		for(j=0;j<N;j++)
		{
			//printf("j=%d \n",j);
			net_h[j]=0;
			for(i=0;i<15;i++){
				net_h[j]=net_h[j]+w_ij[j][i]*x[p][i];
			}
			//printf("net_h=%f \n",net_h[j]);
			out_h[j]=FX(net_h[j]); 
		}

		for(k=0;k<4;k++)
		{
			net_o[k]=0;
			for(j=0;j<N;j++){
				net_o[k]=net_o[k]+w_jk[k][j]*out_h[j];
			}
			out_o[k]=FX(net_o[k]);
		}
		printf("\n");
		printf("%5.2f %5.2f %5.2f %5.2f \n",out_o[0],out_o[1],out_o[2],out_o[3]);
	}

	return 0;
}



