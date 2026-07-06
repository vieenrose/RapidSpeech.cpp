// Parity harness: MOSS global GPT-2 forward (5-token prefill) vs HF fixture.
// Validates the ggml graph (LayerNorm, qkv+bias, interleaved RoPE, causal attn,
// gelu-tanh FFN) against parity_final.bin from the HF model. CPU backend, exact.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

static std::vector<float> load_bin(const char *p, size_t n) {
  std::vector<float> v(n);
  FILE *f = fopen(p, "rb");
  if (!f) { printf("cannot open %s\n", p); exit(1); }
  fread(v.data(), sizeof(float), n, f); fclose(f);
  return v;
}

int main(int argc, char **argv) {
  const char *gguf_path = argc > 1 ? argv[1] : "/home/luigi/moss-port/moss_nano_full.gguf";
  int T = (argc>5?atoi(argv[5]):5); const int H = 768, NH = 12, HD = 64, FF = 3072;
  int NL = argc>2?atoi(argv[2]):12; int rope_mode = argc>3?atoi(argv[3]):0;
  const char* ref_path = argc>4?argv[4]:"/home/luigi/moss-port/parity_final.bin";
  bool final_ln = (NL>=12);
  const float eps = 1e-5f, rope_base = 10000.0f;

  // load gguf weights onto CPU backend
  ggml_context *wctx = nullptr;
  gguf_init_params gp{ /*no_alloc*/ false, /*ctx*/ &wctx };
  gguf_context *gguf = gguf_init_from_file(gguf_path, gp);
  if (!gguf) { printf("gguf load failed\n"); return 1; }
  auto W = [&](const std::string &n) {
    ggml_tensor *t = ggml_get_tensor(wctx, n.c_str());
    if (!t) { printf("missing %s\n", n.c_str()); exit(1); }
    return t;
  };

  ggml_backend_t be = ggml_backend_cpu_init();

  // compute graph context
  size_t buf = (size_t)64 * 1024 * 1024;
  ggml_init_params cp{ buf, nullptr, true };
  ggml_context *ctx = ggml_init(cp);

  ggml_tensor *inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);  // [768,5]
  ggml_set_input(inp);
  ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
  ggml_set_input(pos);

  auto layernorm = [&](ggml_tensor *x, ggml_tensor *w, ggml_tensor *b) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
  };

  ggml_tensor *h = inp;
  for (int i = 0; i < NL; i++) {
    std::string p = "blk." + std::to_string(i) + ".";
    ggml_tensor *ln1 = layernorm(h, W(p + "attn_norm.weight"), W(p + "attn_norm.bias"));
    // qkv
    ggml_tensor *qkv = ggml_add(ctx, ggml_mul_mat(ctx, W(p + "attn_qkv.weight"), ln1), W(p + "attn_qkv.bias")); // [2304,5]
    ggml_tensor *q = ggml_view_2d(ctx, qkv, H, T, qkv->nb[1], 0 * H * sizeof(float));
    ggml_tensor *k = ggml_view_2d(ctx, qkv, H, T, qkv->nb[1], 1 * H * sizeof(float));
    ggml_tensor *v = ggml_view_2d(ctx, qkv, H, T, qkv->nb[1], 2 * H * sizeof(float));
    q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), HD, NH, T);
    k = ggml_reshape_3d(ctx, ggml_cont(ctx, k), HD, NH, T);
    v = ggml_reshape_3d(ctx, ggml_cont(ctx, v), HD, NH, T);
    // interleaved RoPE (GPT-J / NORMAL mode = 0)
    q = ggml_rope_ext(ctx, q, pos, nullptr, HD, rope_mode, 0, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, HD, rope_mode, 0, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    // attention: [HD,NH,T] -> permute to [HD,T,NH]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor *kq = ggml_mul_mat(ctx, k, q);                 // [T,T,NH]
    kq = ggml_scale(ctx, kq, 1.0f / sqrtf(HD));
    kq = ggml_diag_mask_inf(ctx, kq, 0);                       // causal
    kq = ggml_soft_max(ctx, kq);
    ggml_tensor *vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [T,HD,NH]
    ggml_tensor *kqv = ggml_mul_mat(ctx, vt, kq);             // [HD,T,NH]
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                 // [HD,NH,T]
    ggml_tensor *att = ggml_cont_2d(ctx, kqv, H, T);
    att = ggml_add(ctx, ggml_mul_mat(ctx, W(p + "attn_out.weight"), att), W(p + "attn_out.bias"));
    if (i==0 && getenv("MOSS_DUMP_ATTN")) { ggml_set_output(att); ggml_cgraph* gg=ggml_new_graph(ctx); ggml_build_forward_expand(gg,att); ggml_gallocr_t g2=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); ggml_gallocr_alloc_graph(g2,gg); ggml_backend_tensor_set(inp,load_bin(argc>6?argv[6]:"/home/luigi/moss-port/parity_emb_in.bin",(size_t)T*H).data(),0,(size_t)T*H*sizeof(float)); std::vector<int32_t> pv(T); for(int z=0;z<T;z++)pv[z]=z; ggml_backend_tensor_set(pos,pv.data(),0,T*sizeof(int32_t)); ggml_backend_graph_compute(be,gg); std::vector<float> ao((size_t)T*H); ggml_backend_tensor_get(att,ao.data(),0,ao.size()*sizeof(float)); std::vector<float> ar=load_bin("/home/luigi/moss-port/parity_attn0.bin",(size_t)T*H); for(int t=0;t<T;t++){double pm=0,px=0; for(int j=0;j<H;j++){double d=ao[t*H+j]-ar[t*H+j]; pm+=d*d; if(fabs(d)>px)px=fabs(d);} printf("ATTN pos %d: MSE=%.3e max=%.4f\n",t,pm/H,px);} return 0; }
    h = ggml_add(ctx, h, att);
    // ffn
    ggml_tensor *ln2 = layernorm(h, W(p + "ffn_norm.weight"), W(p + "ffn_norm.bias"));
    ggml_tensor *up = ggml_add(ctx, ggml_mul_mat(ctx, W(p + "ffn_up.weight"), ln2), W(p + "ffn_up.bias"));
    up = ggml_gelu(ctx, up);
    ggml_tensor *down = ggml_add(ctx, ggml_mul_mat(ctx, W(p + "ffn_down.weight"), up), W(p + "ffn_down.bias"));
    h = ggml_add(ctx, h, down);
  }
  if (final_ln) h = layernorm(h, W("output_norm.weight"), W("output_norm.bias"));
  ggml_set_output(h);

  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, h);
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
  ggml_gallocr_alloc_graph(ga, gf);

  // inputs
  std::vector<float> emb = load_bin(argc>6?argv[6]:"/home/luigi/moss-port/parity_emb_in.bin", (size_t)T * H);
  ggml_backend_tensor_set(inp, emb.data(), 0, emb.size() * sizeof(float));
  std::vector<int32_t> posv(T); for (int i = 0; i < T; i++) posv[i] = i;
  ggml_backend_tensor_set(pos, posv.data(), 0, T * sizeof(int32_t));

  ggml_backend_graph_compute(be, gf);

  std::vector<float> out((size_t)T * H);
  ggml_backend_tensor_get(h, out.data(), 0, out.size() * sizeof(float));
  std::vector<float> ref = load_bin(ref_path, (size_t)T * H);

  for (int t = 0; t < T; t++) {
    double pm = 0, pmax = 0;
    for (int j = 0; j < H; j++) { double d = out[t*H+j]-ref[t*H+j]; pm += d*d; if (fabs(d)>pmax) pmax=fabs(d); }
    printf("  pos %d: MSE=%.3e max=%.4f\n", t, pm/H, pmax);
  }
  double mse = 0, maxd = 0;
  for (size_t i = 0; i < out.size(); i++) { double d = out[i] - ref[i]; mse += d * d; if (fabs(d) > maxd) maxd = fabs(d); }
  mse /= out.size();
  printf("global forward parity: MSE=%.3e  max|diff|=%.4f\n", mse, maxd);
  printf("  out[last,:6] = %.4f %.4f %.4f %.4f %.4f %.4f\n", out[(T-1)*H+0],out[(T-1)*H+1],out[(T-1)*H+2],out[(T-1)*H+3],out[(T-1)*H+4],out[(T-1)*H+5]);
  printf("  ref[last,:6] = %.4f %.4f %.4f %.4f %.4f %.4f\n", ref[(T-1)*H+0],ref[(T-1)*H+1],ref[(T-1)*H+2],ref[(T-1)*H+3],ref[(T-1)*H+4],ref[(T-1)*H+5]);
  printf("  %s\n", (mse < 1e-3) ? "PARITY PASS" : "PARITY FAIL");
  return 0;
}
