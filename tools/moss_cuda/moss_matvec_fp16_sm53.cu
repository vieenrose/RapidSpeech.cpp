#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
// one warp per output row; ROWS_PER_BLOCK warps/block; warp-shuffle reduce (no syncthreads)
template<int RPB> __global__ void mvw(const __half2* __restrict__ W, const __half* __restrict__ x, float* __restrict__ y, int N){
  int NH=N/2, lane=threadIdx.x&31, warp=threadIdx.x>>5;
  int row=blockIdx.x*RPB+warp; if(row>=N) return;
  const __half2* Wr=W+(size_t)row*NH;
  float acc=0.f;
  for(int i=lane;i<NH;i+=32){ float2 p=__half22float2(__hmul2(Wr[i],((const __half2*)x)[i])); acc+=p.x+p.y; }
  for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
  if(lane==0) y[row]=acc;
}
int main(int argc,char**argv){
  int N=768,K=argc>1?atoi(argv[1]):48,ITERS=300; const int RPB=4;
  __half *W,*x; float*y;
  cudaMalloc(&W,(size_t)N*N*2); cudaMalloc(&x,N*2); cudaMalloc(&y,N*4);
  cudaMemset(W,0,(size_t)N*N*2); cudaMemset(x,0,N*2);
  int blocks=(N+RPB-1)/RPB, threads=RPB*32;
  for(int i=0;i<20;i++) for(int k=0;k<K;k++) mvw<RPB><<<blocks,threads>>>((const __half2*)W,x,y,N);
  cudaDeviceSynchronize();
  double t0=now();
  for(int it=0;it<ITERS;it++) for(int k=0;k<K;k++) mvw<RPB><<<blocks,threads>>>((const __half2*)W,x,y,N);
  cudaDeviceSynchronize();
  printf("custom warp-reduce (RPB=%d): %.4f ms/matmul (ggml=0.44, floor~0.046)\n",RPB,(now()-t0)/ITERS/K);
  return 0;
}
