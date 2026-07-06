// Local-decoder parity: 16-position 1-layer local transformer + per-position audio_lm_heads
// vs moss_tts_local_decoder.onnx (audio_logits). Reuses the verified global block math.
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
  std::vector<float> v(n); FILE *f = fopen(p, "rb");
  if (!f) { printf("cannot open %s\n", p); exit(1); }
  fread(v.data(), sizeof(float), n, f); fclose(f); return v;
}

int main(int argc, char **argv) {
  const char *gg = argc > 1 ? argv[1] : "/home/luigi/moss-port/moss_ar_f32.gguf";
  const int T = 17, H = 768, NH = 12, HD = 64, FF = 3072, CB = 1024, NCB = 16;
  const float eps = 1e-5f, rope_base = 10000.0f;
  ggml_context *wctx = nullptr;
  gguf_init_params gp{ false, &wctx };
  if (!gguf_init_from_file(gg, gp)) { printf("gguf fail\n"); return 1; }
  auto W = [&](const std::string &n){ ggml_tensor*t=ggml_get_tensor(wctx,n.c_str()); if(!t){printf("missing %s\n",n.c_str());exit(1);} return t; };
  ggml_backend_t be = ggml_backend_cpu_init();
  ggml_init_params cp{ (size_t)128*1024*1024, nullptr, true };
  ggml_context *ctx = ggml_init(cp);

  ggml_tensor *inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);  ggml_set_input(inp);
  ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);     ggml_set_input(pos);
  auto LN = [&](ggml_tensor*x,ggml_tensor*w,ggml_tensor*b){ return ggml_add(ctx,ggml_mul(ctx,ggml_norm(ctx,x,eps),w),b); };

  std::string p = "local.blk.0.";
  ggml_tensor *h = inp;
  ggml_tensor *ln1 = LN(h, W(p+"attn_norm.weight"), W(p+"attn_norm.bias"));
  ggml_tensor *qkv = ggml_add(ctx, ggml_mul_mat(ctx, W(p+"attn_qkv.weight"), ln1), W(p+"attn_qkv.bias"));
  ggml_tensor *q = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx,qkv,H,T,qkv->nb[1],0)), HD,NH,T);
  ggml_tensor *k = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx,qkv,H,T,qkv->nb[1],H*sizeof(float))), HD,NH,T);
  ggml_tensor *v = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx,qkv,H,T,qkv->nb[1],2*H*sizeof(float))), HD,NH,T);
  q = ggml_rope_ext(ctx,q,pos,nullptr,HD,0,0,rope_base,1,0,1,0,0);
  k = ggml_rope_ext(ctx,k,pos,nullptr,HD,0,0,rope_base,1,0,1,0,0);
  q = ggml_permute(ctx,q,0,2,1,3); k = ggml_permute(ctx,k,0,2,1,3);
  ggml_tensor *kq = ggml_soft_max(ctx, ggml_diag_mask_inf(ctx, ggml_scale(ctx, ggml_mul_mat(ctx,k,q), 1.0f/sqrtf(HD)), 0));
  ggml_tensor *vt = ggml_cont(ctx, ggml_permute(ctx,v,1,2,0,3));
  ggml_tensor *kqv = ggml_permute(ctx, ggml_mul_mat(ctx,vt,kq), 0,2,1,3);
  ggml_tensor *att = ggml_add(ctx, ggml_mul_mat(ctx, W(p+"attn_out.weight"), ggml_cont_2d(ctx,kqv,H,T)), W(p+"attn_out.bias"));
  h = ggml_add(ctx, h, att);
  ggml_tensor *ln2 = LN(h, W(p+"ffn_norm.weight"), W(p+"ffn_norm.bias"));
  ggml_tensor *up = ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, W(p+"ffn_up.weight"), ln2), W(p+"ffn_up.bias")));
  h = ggml_add(ctx, h, ggml_add(ctx, ggml_mul_mat(ctx, W(p+"ffn_down.weight"), up), W(p+"ffn_down.bias")));
  h = LN(h, W("local.output_norm.weight"), W("local.output_norm.bias"));   // [H, T=16]

  // per-position audio head: logits_k = audio_embd[k] . h[:,k]
  std::vector<ggml_tensor*> logit(NCB);
  for (int kk = 0; kk < NCB; kk++) {
    ggml_tensor *hk = ggml_view_1d(ctx, h, H, (kk+1)*h->nb[1]);
    logit[kk] = ggml_mul_mat(ctx, W("audio_embd."+std::to_string(kk)+".weight"), ggml_cont(ctx, hk)); // [1024]
    ggml_set_output(logit[kk]);
  }
  ggml_cgraph *gf = ggml_new_graph(ctx);
  for (int kk=0;kk<NCB;kk++) ggml_build_forward_expand(gf, logit[kk]);
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
  ggml_gallocr_alloc_graph(ga, gf);

  // input: pos0 = global_hidden, pos1..15 = 0
  std::vector<float> gh = load_bin("/home/luigi/moss-port/ld_gh.bin", H);
  std::vector<float> in((size_t)H*T, 0.0f);
  for (int j=0;j<H;j++) in[0*H+j]=gh[j];   // wait: layout [H,T] -> element [h,t] at t*H+h
  // fix: inp is [H(ne0),T(ne1)] row-major over ne0 -> col t at offset t*H
  std::vector<float> wte7 = load_bin("/home/luigi/moss-port/wte7.bin", H);
  std::vector<float> in2((size_t)H*T,0.0f);
  for(int j=0;j<H;j++){ in2[0*H+j]=gh[j]; in2[1*H+j]=wte7[j]; }   // pos0=global_hidden, pos1=wte(7), rest 0
  ggml_backend_tensor_set(inp, in2.data(), 0, in2.size()*sizeof(float));
  std::vector<int32_t> pv(T); for(int i=0;i<T;i++) pv[i]=i;
  ggml_backend_tensor_set(pos, pv.data(), 0, T*sizeof(int32_t));
  ggml_backend_graph_compute(be, gf);

  std::vector<float> ref = load_bin("/home/luigi/moss-port/ld_al.bin", (size_t)NCB*CB);
  int match=0; double mse=0;
  for (int kk=0;kk<NCB;kk++){
    std::vector<float> lg(CB); ggml_backend_tensor_get(logit[kk], lg.data(), 0, CB*sizeof(float));
    int am=0; float mx=lg[0]; for(int c=1;c<CB;c++) if(lg[c]>mx){mx=lg[c];am=c;}
    int ra=0; float rmx=ref[kk*CB]; for(int c=1;c<CB;c++) if(ref[kk*CB+c]>rmx){rmx=ref[kk*CB+c];ra=c;}
    for(int c=0;c<CB;c++){double d=lg[c]-ref[kk*CB+c]; mse+=d*d;}
    if(am==ra) match++;
    if(kk<6) printf("  cb%d: ggml_argmax=%d ref_argmax=%d %s\n",kk,am,ra,am==ra?"OK":"X");
  }
  printf("LOCAL DECODER: argmax match %d/%d, logits MSE=%.3e  %s\n", match, NCB, mse/(NCB*CB), match==NCB?"PASS":"CHECK");
  return 0;
}
