// cuFFT mel-spectrogram for Qwen3-ASR on Jetson Nano gen1 (sm_53, CUDA 10.2).
// Replaces the reference O(frames*400^2) single-threaded double cos/sin DFT (~22s on a 13.5s clip)
// with a batched R2C FFT on the GPU. Bit-faithful to the reference: R2C uses exp(-i 2*pi k n / N),
// producing 1+N/2 = 201 bins from a 400-sample real frame; power = re^2+im^2; mel = log10(max(.,1e-10)).
// Normalization (mmax-8 clamp) stays in the C++ caller, so transcripts are unchanged.
#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#define CKC(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"mel_cufft cuda err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return false; } } while(0)
#define CKF(x) do { cufftResult e=(x); if(e!=CUFFT_SUCCESS){ \
    fprintf(stderr,"mel_cufft cufft err %s @%d: %d\n",#x,__LINE__,(int)e); return false; } } while(0)

// Build windowed frames: frames[i*fs + n] = hann[n] * samples_padded[i*step + n]
__global__ void k_window(const float* __restrict__ samp, const float* __restrict__ hann,
                         float* __restrict__ frames, int compute_frames, int fs, int step) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;   // 0..fs-1
    int i = blockIdx.y;                              // 0..compute_frames-1
    if (n < fs && i < compute_frames)
        frames[(long)i * fs + n] = hann[n] * samp[(long)i * step + n];
}

// power + mel filterbank + log10. One thread per (m, i). temp_out[m*compute_frames + i].
__global__ void k_mel(const cufftComplex* __restrict__ fft, const float* __restrict__ filt,
                      float* __restrict__ temp_out, int compute_frames, int n_bins, int n_mel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // frame
    int m = blockIdx.y;                              // mel bin
    if (i >= compute_frames || m >= n_mel) return;
    const cufftComplex* col = fft + (long)i * n_bins;
    const float* fr = filt + (long)m * n_bins;
    float sum = 0.0f;
    for (int k = 0; k < n_bins; k++) {
        float re = col[k].x, im = col[k].y;
        sum += (re*re + im*im) * fr[k];
    }
    temp_out[(long)m * compute_frames + i] = log10f(sum > 1e-10f ? sum : 1e-10f);
}

// hann_f: [fs] single-precision Hann window. mel_filters: [n_mel * n_bins] row-major.
// temp_data (out): [n_mel * compute_frames], double, column i = frame i (matches reference layout).
extern "C" bool qwen3_mel_cufft(const float* samples_padded, int n_padded,
                                int compute_frames, int frame_size, int frame_step,
                                const float* hann_f, const float* mel_filters,
                                int n_mel, int n_bins, double* temp_data) {
    if (compute_frames <= 0) return false;
    const int fs = frame_size;
    float *d_samp=nullptr,*d_hann=nullptr,*d_frames=nullptr,*d_filt=nullptr,*d_temp=nullptr;
    cufftComplex* d_fft=nullptr;
    cufftHandle plan=0;
    bool ok=false;
    do {
        CKC(cudaMalloc(&d_samp,  (size_t)n_padded*sizeof(float)));
        CKC(cudaMalloc(&d_hann,  (size_t)fs*sizeof(float)));
        CKC(cudaMalloc(&d_frames,(size_t)compute_frames*fs*sizeof(float)));
        CKC(cudaMalloc(&d_filt,  (size_t)n_mel*n_bins*sizeof(float)));
        CKC(cudaMalloc(&d_temp,  (size_t)n_mel*compute_frames*sizeof(float)));
        CKC(cudaMalloc(&d_fft,   (size_t)compute_frames*n_bins*sizeof(cufftComplex)));
        CKC(cudaMemcpy(d_samp, samples_padded, (size_t)n_padded*sizeof(float), cudaMemcpyHostToDevice));
        CKC(cudaMemcpy(d_hann, hann_f, (size_t)fs*sizeof(float), cudaMemcpyHostToDevice));
        CKC(cudaMemcpy(d_filt, mel_filters, (size_t)n_mel*n_bins*sizeof(float), cudaMemcpyHostToDevice));

        dim3 wb((fs+127)/128,1,1), wt(128,1,1); wb.y=compute_frames;
        k_window<<<wb,wt>>>(d_samp,d_hann,d_frames,compute_frames,fs,frame_step);
        CKC(cudaGetLastError());

        int n[1]={fs};
        CKF(cufftPlanMany(&plan,1,n, NULL,1,fs, NULL,1,n_bins, CUFFT_R2C, compute_frames));
        CKF(cufftExecR2C(plan, d_frames, d_fft));

        dim3 mb((compute_frames+127)/128, n_mel, 1), mt(128,1,1);
        k_mel<<<mb,mt>>>(d_fft,d_filt,d_temp,compute_frames,n_bins,n_mel);
        CKC(cudaGetLastError());
        CKC(cudaDeviceSynchronize());

        static float* h_tmp=nullptr; static size_t h_cap=0;
        size_t need=(size_t)n_mel*compute_frames;
        if (h_cap < need) { free(h_tmp); h_tmp=(float*)malloc(need*sizeof(float)); h_cap=need; }
        CKC(cudaMemcpy(h_tmp, d_temp, need*sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t t=0;t<need;t++) temp_data[t]=(double)h_tmp[t];
        ok=true;
    } while(0);
    if (plan) cufftDestroy(plan);
    cudaFree(d_samp); cudaFree(d_hann); cudaFree(d_frames);
    cudaFree(d_filt); cudaFree(d_temp); cudaFree(d_fft);
    return ok;
}
