#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>  
#include <unistd.h>

#define msleep(x) usleep(x*1000)
#define FX(x) ( 1.0/(1.0+ exp(-(x))) )  //Sigmoid funtion;
#define F1(x) (x)*(1.0-x)               //A derivative function of sigmoid function
#define N 8                             //numbers of hidden layer 

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
void randomize()
{ 
	int i;
	time_t t;
	srand((unsigned) time(&t));
	//srand(time(NULL));
}

int random(int number) // return the value between 0 ~ N-1
{ 
	return ( rand() % number );
}

int main(int argc, char** argv) 
{ 
	int i,j,k,p=10,num,x[10][15],d_o[10][4];
	float net_h[N],out_h[N],net_o[4],out_o[4];
	float err_h[N],err_o[4];
	float w_ij[15][N],w_jk[N][4];
	float eta=0.2;
	float mse[10];

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

		printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n",
		x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],
		x[i][5],x[i][6],x[i][7],x[i][8],x[i][9],
		x[i][10],x[i][11],x[i][12],x[i][13],x[i][14],
		d_o[i][0],d_o[i][1],d_o[i][2],d_o[i][3]);

	}
	fclose(fp);
	
	if((fp=fopen("backprop.out","w"))==NULL){
		printf("open out error \n");
		return 1;
	}

	randomize();
	for(j=0;j<N;j++){
		for(i=0;i<15;i++){
			w_ij[i][j]=sin(random(200))*0.5;
			//printf("%f ",w_ij[i][j]);
		}

	}
	
	for(k=0;k<4;k++){
		for(j=0;j<N;j++){
			w_jk[j][k]=sin(random(200))*0.5;
			//printf("%f ",w_jk[j][k]);
		}
	}

#if 1
	p=0;
	num=1;
	while(1){
		msleep(3);
		//printf("%d ",p);
		/*---------------------------------------------------------------------*/
		if(p==10)	/* if trainning whole number one time, then print */
		{
			//printf("\n");
			float m=0;
			//printf("%4d \n",num);
			
			fprintf(fp,"%4d",num);
			for(i=0;i<10;i++){
				m=m+mse[i];
				printf("%5.3f \n",mse[i]);
				fprintf(fp,"%5.3f",mse[i]);
			}
			printf("m: %5.3f \n",m);
			fprintf(fp,"\n");
			if(m<=0.1){	
			//if(m<=0.05){	
				for(j=0;j<N;j++){
					for(i=0;i<15;i++){
						printf("w_ij[%d][%d]=",i,j);
						printf("%f",w_ij[i][j]);
						printf(";");
						printf("\n");
					}

				}
				printf("\n ----- \n");
				for(k=0;k<4;k++){
					for(j=0;j<N;j++){
						printf("w_jk[%d][%d]=",j,k);
						printf("%f",w_jk[j][k]);
						printf(";");
						printf("\n");
					}
				}
				printf("\n ----- \n");
				break;
			} 
			p=0;
			num++;
		}
		
		//"hidden layer" input value
		for(j=0;j<N;j++)
		{
		    net_h[j]=0;
		    for(i=0;i<15;i++){
			net_h[j]=net_h[j]+w_ij[i][j]*x[p][i];
		    }
		    out_h[j]=FX(net_h[j]);  //"hidden layer" output value
		}

		//"output layer" input value
		for(k=0;k<4;k++)
		{
		    net_o[k]=0;
		    for(j=0;j<N;j++){			    	
			net_o[k]=net_o[k]+w_jk[j][k]*out_h[j];
		    }
		    out_o[k]=FX(net_o[k]); 
		}

		//"output layer" error
		for(k=0;k<4;k++){
			err_o[k]=(d_o[p][k]-out_o[k])*F1(out_o[k]);
		}

		//"hidden layer" error
		for(j=0;j<N;j++)
		{
		    err_h[j]=0;
		    for(k=0;k<4;k++){
		    	//err_h[j]=err_h[j]*err_o[k]*w_jk[j][k];
		    	err_h[j]=err_h[j]+err_o[k]*w_jk[j][k];
		    }
		    err_h[j]=err_h[j]*F1(out_h[j]);
		}

		//update weight of "output layer"
		for(j=0;j<4;j++)
		{
		    for(k=0;k<N;k++){
		    	w_jk[j][k]=w_jk[j][k]+eta*err_o[k]*out_h[j];
		    }
		}

		//update weight of "hidden layer"
		for(j=0;j<N;j++)
		{
		    for(i=0;i<15;i++){
			w_ij[i][j]=w_ij[i][j]+eta*err_h[j]*x[p][i];
		    }
		}

		//caculate error 
		mse[p]=0;
		for(k=0;k<4;k++){
		    mse[p]=mse[p]+(d_o[p][k]-out_o[k])*(d_o[p][k]-out_o[k]);
		}	
		//mse[p]=0.5*mse[p];
		mse[p]=sqrt(mse[p]);
		p++;
	}
	fclose(fp);	
#endif
	return 0;
}



