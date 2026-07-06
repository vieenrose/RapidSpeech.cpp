# MOSS-Nano custom matvec kernels (Maxwell sm_53 / Jetson Nano gen1)

Stock ggml `mul_mat_vec` runs at only ~13% of memory bandwidth on sm_53. These
warp-per-output-row + `__shfl_down_sync`-reduce kernels are 2.6–5.2× faster:

| kernel (768×768) | ggml | custom |
|---|---|---|
| fp16 | 0.44 ms | **0.084 ms (5.2×)** |
| Q8_0 | 0.184 ms | **0.071 ms (2.6×)** |

Design: one warp per output row, 32 lanes stride the row with coalesced `char4`/`half2`
loads, warp-shuffle reduction (no `__syncthreads`), 8 warps/block for occupancy.

Projected port impact (matmul ≈ 98% of frame): full-12L Q8 RTF 0.91 → ~0.35;
4-layer student → ~0.21. Integration: swap into ggml-CUDA's `mul_mat_vec_q` dispatch
for the MOSS decode graph. Build: `nvcc -O3 -arch=sm_53 moss_matvec_q8_sm53.cu -o mv`.
