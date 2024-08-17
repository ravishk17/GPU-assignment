#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

//*******************************************

// Write down the kernels here


__device__ int gcd(int a, int b) {
    int tmp;
    while(b){
        tmp=b;
        b=a%b;
        a=tmp;
    }
    return a;
}


__device__ int distance(int x1, int y1, int x2, int y2){
  return abs(x1-x2)+abs(y1-y2);
}

__global__ void setHP(int *d_HP,int H){
   int id=threadIdx.x;
   d_HP[id]=H;
}

__global__ void setDistances(int* d_distance, int* d_xcoord, int* d_ycoord){
  int tank1 = blockIdx.x;
  int tank2 = threadIdx.x;
  int pos = tank1 * blockDim.x + tank2;
  
  int x1,y1, x2,y2;
  x1=d_xcoord[tank1];
  y1=d_ycoord[tank1];
  x2=d_xcoord[tank2];
  y2=d_ycoord[tank2];
  d_distance[pos] = distance(x1,y1,x2,y2);
  __syncthreads();
}


__global__ void funBegins(int*d_active, int T,int round,int* d_xcoord, int* d_ycoord,int *d_distance,int* d_score,int* d_HP, int* minEnemyDist, int* minEnemy,int BlockSize){
	int me = blockIdx.x;
	if(d_active[me]==1){
		int enemy = (me+round)%T;
	
		int x1 = d_xcoord[me];
		int y1 = d_ycoord[me];
		
		int x2 = d_xcoord[enemy];
		int y2 = d_ycoord[enemy];
		
		int dy = y2-y1;
		int dx = x2-x1;
		
		int other = threadIdx.x;
		if(other!=me && d_active[other]==1){
			int x3 = d_xcoord[other];
			int y3 = d_ycoord[other];
			
			int dyy = y3 - y1;
			int dxx = x3 -x1;
			if(dx!=0 && dy!=0 && dxx!=0 && dyy!=0){
				int div1 = gcd(abs(dx),abs(dy));
				dx/=div1;
				dy/=div1;
				int div2 = gcd(abs(dxx),abs(dyy));
				dxx/=div2;
				dyy/=div2;
				if(dx==dxx && dy==dyy){
					int dist = d_distance[me*T+other];
						minEnemyDist[me*BlockSize+other]=dist;
						minEnemy[me*BlockSize+other]=other;
				}
			}
			else if((dx==0 && dxx==0) || (dy==0 && dyy==0)){
				if((dx>0 && dxx>0) || (dx<0 && dxx<0) || (dy<0 && dyy<0) || (dy>0 && dyy>0)){
					int dist = d_distance[me*T+other];
						minEnemyDist[me*BlockSize+other]=dist;
						minEnemy[me*BlockSize+other]=other;
				}
			}
		}
    __syncthreads();
	}
}


__global__ void setMinDist(int* minEnemyDist){
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	minEnemyDist[id] = INT_MAX;
}

__global__ void setMinEnemy(int* minEnemy){
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	minEnemy[id] = -1;
}


__global__ void minRowWithIndex(int* minEnemyDist, int* minEnemy, int* result, int size,int T, int BlockSize) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    int rowStartIndex = bid * size;
    extern __shared__ int s[];
    int *minVal = s;
    int *minIndex = s + BlockSize;
    
    minVal[tid] = minEnemyDist[rowStartIndex + tid];
    minIndex[tid] = minEnemy[rowStartIndex + tid];
    
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (minVal[tid] > minVal[tid + stride]) {
                minVal[tid] = minVal[tid + stride];
                minIndex[tid]=minIndex[tid+stride];
            }
        }
        __syncthreads();
    }
    __syncthreads();
    if (tid == 0 && bid<T) {
        result[bid] = minIndex[0];
    }
    minEnemyDist[rowStartIndex+tid]=INT_MAX;
    minEnemy[rowStartIndex+tid]=-1;
}



__global__ void updateScore(int* result, int* d_score, int* d_HP, int* d_active, int *alive, int* round){
	int id = threadIdx.x;
	if(result[id] != -1 && d_active[id]){
    atomicInc((unsigned int*)&d_score[id],INT_MAX);
		atomicSub((unsigned int*)&d_HP[result[id]],1);
	}
	__syncthreads();
  if(d_HP[id]<=0 && d_active[id]==1){
			d_active[id]=0;
      atomicSub((unsigned int*)alive,1);
	}
  __syncthreads();
  result[id]=-1;
  if(id==0){
    *round = *round + 1;
    if((*round)%blockDim.x==0){
      *round = *round +1;
    }
  }
}


//***********************************************


int main(int argc, char** argv)
{
    // Variable declarations
    int M, N, T, H, * xcoord, * ycoord, * score;


    FILE* inputfilepointer;

    //File Opening for read
    char* inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    if (inputfilepointer == NULL) {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &M);
    fscanf(inputfilepointer, "%d", &N);
    fscanf(inputfilepointer, "%d", &T); // T is number of Tanks
    fscanf(inputfilepointer, "%d", &H); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord = (int*)malloc(T * sizeof(int));  // X coordinate of each tank
    ycoord = (int*)malloc(T * sizeof(int));  // Y coordinate of each tank
    score = (int*)malloc(T * sizeof(int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for (int i = 0; i < T; i++)
    {
        fscanf(inputfilepointer, "%d", &xcoord[i]);
        fscanf(inputfilepointer, "%d", &ycoord[i]);
    }


    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************
    cudaFuncSetCacheConfig(minRowWithIndex,cudaFuncCachePreferShared);
    int BlockSize = T;
    BlockSize--; 
    BlockSize |= BlockSize >> 1;
    BlockSize |= BlockSize >> 2;
    BlockSize |= BlockSize >> 4;
    BlockSize |= BlockSize >> 8;
    BlockSize |= BlockSize >> 16;
    BlockSize++; 
    int* d_xcoord;
    int* d_ycoord;

    cudaMalloc(&d_xcoord,T*sizeof(int));
    cudaMalloc(&d_ycoord,T*sizeof(int));
    
    cudaMemcpy(d_xcoord, xcoord, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ycoord, ycoord, T * sizeof(int), cudaMemcpyHostToDevice);
    
    int* alive;
    cudaHostAlloc(&alive,sizeof(int),0);
    *alive = T;

    int* d_HP;
    cudaMalloc(&d_HP, T * sizeof(int));
    setHP<<<1,T>>> (d_HP,H);

    int* d_active;
    cudaMalloc(&d_active,T*sizeof(int));
    setHP<<<1,T>>>(d_active,1);

    int* d_score;
    cudaMalloc(&d_score, T * sizeof(int));
    setHP<<<1,T>>>(d_score,0);

    int* round;
    cudaHostAlloc(&round,sizeof(int),0);
    *round = 1;

    int* d_distance;
    cudaMalloc(&d_distance, T*T*sizeof(int));
    
    setDistances<<<T,T>>>(d_distance, d_xcoord, d_ycoord);
    cudaDeviceSynchronize();
    
    int* minEnemyDist;
    cudaMalloc(&minEnemyDist,BlockSize*BlockSize*sizeof(int));
    setMinDist<<<BlockSize,BlockSize>>>(minEnemyDist);
    
    
    int* minEnemy;
    cudaMalloc(&minEnemy,BlockSize*BlockSize*sizeof(int));
    setMinEnemy<<<BlockSize,BlockSize>>>(minEnemy);
    
    int* result;
    cudaMalloc(&result,BlockSize*sizeof(int));
    setHP<<<1,BlockSize>>>(result,-1);
    
    cudaDeviceSynchronize();
    while(*alive>1){
      funBegins<<<T,T>>>(d_active,T,*round,d_xcoord,d_ycoord,d_distance,d_score,d_HP,minEnemyDist,minEnemy,BlockSize);
      cudaDeviceSynchronize();
      minRowWithIndex<<<BlockSize,BlockSize, 2*BlockSize*sizeof(int)>>> (minEnemyDist, minEnemy,result, BlockSize,T,BlockSize);
      cudaDeviceSynchronize();
      updateScore<<<1,T>>>(result,d_score,d_HP,d_active,alive, round);
      cudaDeviceSynchronize();
    }
    cudaFree(result);
    cudaFree(minEnemy);
    cudaFree(minEnemyDist);
    cudaFree(d_distance);
    cudaFree(d_active);
    cudaFree(d_HP);
    cudaMemcpy(score, d_score, T * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_score);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end - start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char* outputfilename = argv[2];
    char* exectimefilename = argv[3];
    FILE* outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    for (int i = 0; i < T; i++)
    {
        fprintf(outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename, "w");
    fprintf(outputfilepointer, "%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}
