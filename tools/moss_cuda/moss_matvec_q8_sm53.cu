#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
template<int RPB> __global__ void mvq8v(const char4* __restrict__ Wq, const __half* __restrict__ Wsc, const __half* __restrict__ x, float* __restrict__ y, int N){
  int lane=threadIdx.x&31, warp=threadIdx.x>>5, row=blockIdx.x*RPB+warp; if(row>=N) return;
  int N4=N>>2;
  const char4* Wr=Wq+(size_t)row*N4; const __half* Sr=Wsc+(size_t)row*(N>>5);
  float acc=0.f;
  for(int i=lane;i<N4;i+=32){ char4 w=Wr[i]; int b=i<<2; float s=__half2float(Sr[b>>5]); __half2 x01=((const __half2*)x)[i*2]; __half2 x23=((const __half2*)x)[i*2+1]; float2 xa=__half22float2(x01),xb=__half22float2(x23);
    acc += ((float)w.x*xa.x+(float)w.y*xa.y+(float)w.z*xb.x+(float)w.w*xb.y)*s; }
  for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
  if(lane==0) y[row]=acc;
}
int main(int argc,char**argv){
  int N=768,K=argc>1?atoi(argv[1]):48,ITERS=300; const int RPB=8;
  char4*Wq; __half*Wsc,*x; float*y;
  cudaMalloc(&Wq,(size_t)N*N); cudaMalloc(&Wsc,(size_t)N*(N/32)*2); cudaMalloc(&x,N*2); cudaMalloc(&y,N*4);
  cudaMemset(Wq,0,(size_t)N*N); cudaMemset(Wsc,0,(size_t)N*(N/32)*2); cudaMemset(x,0,N*2);
  int blocks=(N+RPB-1)/RPB, threads=RPB*32;
  for(int i=0;i<20;i++) for(int k=0;k<K;k++) mvq8v<RPB><<<blocks,threads>>>(Wq,Wsc,x,y,N);
  cudaDeviceSynchronize(); double t0=now();
  for(int it=0;it<ITERS;it++) for(int k=0;k<K;k++) mvq8v<RPB><<<blocks,threads>>>(Wq,Wsc,x,y,N);
  cudaDeviceSynchronize();
  printf("custom Q8 vectorized: %.4f ms/matmul (ggml Q8=0.184 -> speedup %.1fx)\n",(now()-t0)/ITERS/K, 0.184/((now()-t0)/ITERS/K));
  return 0;
}
