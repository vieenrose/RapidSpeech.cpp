// KV-cache incremental decode (streaming core): prefill stores per-layer K/V in
// persistent backend buffers; decode processes 1 token attending over cached K/V.
// Verifies decode output == ONNX decode_step (ds_gh.bin). Reuses verified block math.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

static std::vector<float> load_bin(const char*p,size_t n){std::vector<float> v(n);FILE*f=fopen(p,"rb");if(!f){printf("open %s\n",p);exit(1);}fread(v.data(),4,n,f);fclose(f);return v;}

static const int H=768,NH=12,HD=64,NL=12,FF=3072,MAXSEQ=64;
static const float EPS=1e-5f,ROPE=10000.0f;

int main(int argc,char**argv){
  const char*gg=argc>1?argv[1]:"/home/luigi/moss-port/moss_ar_f32.gguf";
  ggml_context*wctx=nullptr; gguf_init_params gp{false,&wctx};
  if(!gguf_init_from_file(gg,gp)){printf("gguf fail\n");return 1;}
  auto W=[&](const std::string&n){ggml_tensor*t=ggml_get_tensor(wctx,n.c_str());if(!t){printf("missing %s\n",n.c_str());exit(1);}return t;};
  ggml_backend_t be=ggml_backend_cpu_init();

  // persistent KV cache buffers (per layer): [HD, NH, MAXSEQ]
  ggml_init_params kp{(size_t)(2*NL+4)*ggml_tensor_overhead(),nullptr,true};
  ggml_context*kvc=ggml_init(kp);
  ggml_tensor*Kc[NL],*Vc[NL];
  for(int i=0;i<NL;i++){Kc[i]=ggml_new_tensor_3d(kvc,GGML_TYPE_F32,HD,NH,MAXSEQ);Vc[i]=ggml_new_tensor_3d(kvc,GGML_TYPE_F32,HD,NH,MAXSEQ);}
  ggml_backend_alloc_ctx_tensors(kvc,be);

  auto LN=[&](ggml_context*c,ggml_tensor*x,ggml_tensor*w,ggml_tensor*b){return ggml_add(c,ggml_mul(c,ggml_norm(c,x,EPS),w),b);};

  // build a forward over n_new tokens at positions [n_past .. n_past+n_new), writing K/V into cache, attending over [0..n_past+n_new)
  auto run=[&](const std::vector<float>&emb,int n_new,int n_past)->std::vector<float>{
    int n_kv=n_past+n_new;
    ggml_init_params cp{(size_t)96*1024*1024,nullptr,true}; ggml_context*ctx=ggml_init(cp);
    ggml_tensor*inp=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,H,n_new); ggml_set_input(inp);
    ggml_tensor*pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,n_new); ggml_set_input(pos);
    ggml_tensor*h=inp;
    ggml_cgraph*gf=ggml_new_graph(ctx);
    for(int i=0;i<NL;i++){
      std::string p="blk."+std::to_string(i)+".";
      ggml_tensor*ln1=LN(ctx,h,W(p+"attn_norm.weight"),W(p+"attn_norm.bias"));
      ggml_tensor*qkv=ggml_add(ctx,ggml_mul_mat(ctx,W(p+"attn_qkv.weight"),ln1),W(p+"attn_qkv.bias"));
      ggml_tensor*q=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_view_2d(ctx,qkv,H,n_new,qkv->nb[1],0)),HD,NH,n_new);
      ggml_tensor*k=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_view_2d(ctx,qkv,H,n_new,qkv->nb[1],H*4)),HD,NH,n_new);
      ggml_tensor*v=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_view_2d(ctx,qkv,H,n_new,qkv->nb[1],2*H*4)),HD,NH,n_new);
      q=ggml_rope_ext(ctx,q,pos,nullptr,HD,0,0,ROPE,1,0,1,0,0);
      k=ggml_rope_ext(ctx,k,pos,nullptr,HD,0,0,ROPE,1,0,1,0,0);
      // write new K/V into cache at [.., n_past : n_kv]
      ggml_tensor*kdst=ggml_view_3d(ctx,Kc[i],HD,NH,n_new,Kc[i]->nb[1],Kc[i]->nb[2],(size_t)n_past*Kc[i]->nb[2]);
      ggml_tensor*vdst=ggml_view_3d(ctx,Vc[i],HD,NH,n_new,Vc[i]->nb[1],Vc[i]->nb[2],(size_t)n_past*Vc[i]->nb[2]);
      ggml_build_forward_expand(gf,ggml_cpy(ctx,k,kdst));
      ggml_build_forward_expand(gf,ggml_cpy(ctx,v,vdst));
      // attention over cached [0..n_kv)
      ggml_tensor*Kall=ggml_view_3d(ctx,Kc[i],HD,NH,n_kv,Kc[i]->nb[1],Kc[i]->nb[2],0);
      ggml_tensor*Vall=ggml_view_3d(ctx,Vc[i],HD,NH,n_kv,Vc[i]->nb[1],Vc[i]->nb[2],0);
      ggml_tensor*qp=ggml_permute(ctx,q,0,2,1,3);                 // [HD, n_new, NH]
      ggml_tensor*kp=ggml_permute(ctx,Kall,0,2,1,3);              // [HD, n_kv, NH]
      ggml_tensor*kq=ggml_mul_mat(ctx,kp,qp);                     // [n_kv, n_new, NH]
      kq=ggml_scale(ctx,kq,1.0f/sqrtf(HD));
      kq=ggml_diag_mask_inf(ctx,kq,n_past);                       // causal w/ past offset
      kq=ggml_soft_max(ctx,kq);
      ggml_tensor*vt=ggml_cont(ctx,ggml_permute(ctx,Vall,1,2,0,3)); // [n_kv, HD, NH]
      ggml_tensor*kqv=ggml_permute(ctx,ggml_mul_mat(ctx,vt,kq),0,2,1,3); // [HD,NH,n_new]
      ggml_tensor*att=ggml_add(ctx,ggml_mul_mat(ctx,W(p+"attn_out.weight"),ggml_cont_2d(ctx,kqv,H,n_new)),W(p+"attn_out.bias"));
      h=ggml_add(ctx,h,att);
      ggml_tensor*ln2=LN(ctx,h,W(p+"ffn_norm.weight"),W(p+"ffn_norm.bias"));
      ggml_tensor*up=ggml_gelu(ctx,ggml_add(ctx,ggml_mul_mat(ctx,W(p+"ffn_up.weight"),ln2),W(p+"ffn_up.bias")));
      h=ggml_add(ctx,h,ggml_add(ctx,ggml_mul_mat(ctx,W(p+"ffn_down.weight"),up),W(p+"ffn_down.bias")));
    }
    h=LN(ctx,h,W("output_norm.weight"),W("output_norm.bias")); ggml_set_output(h);
    ggml_build_forward_expand(gf,h);
    ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga,gf);
    ggml_backend_tensor_set(inp,emb.data(),0,(size_t)H*n_new*4);
    std::vector<int32_t> pv(n_new); for(int j=0;j<n_new;j++) pv[j]=n_past+j;
    ggml_backend_tensor_set(pos,pv.data(),0,n_new*4);
    ggml_backend_graph_compute(be,gf);
    std::vector<float> out((size_t)H*n_new);
    ggml_backend_tensor_get(h,out.data(),0,out.size()*4);
    ggml_gallocr_free(ga); ggml_free(ctx);
    return out;
  };

  // prefill 5 tokens, then decode 1 (token6) via KV cache
  std::vector<float> emb6=load_bin("/home/luigi/moss-port/emb6.bin",(size_t)6*H);
  std::vector<float> pre(emb6.begin(),emb6.begin()+5*H);
  run(pre,5,0);                                    // prefill -> fills KV[0..5)
  std::vector<float> dec(emb6.begin()+5*H,emb6.end());
  std::vector<float> out=run(dec,1,5);             // decode token6 using cached KV
  std::vector<float> ref=load_bin("/home/luigi/moss-port/ds_gh.bin",H);
  double mse=0,mx=0; for(int j=0;j<H;j++){double d=out[j]-ref[j];mse+=d*d;if(fabs(d)>mx)mx=fabs(d);}
  printf("KV-cache decode vs ONNX decode_step: MSE=%.3e max=%.4f  %s\n",mse/H,mx,mse/H<1e-3?"PASS":"FAIL");
  printf("  ggml[:6]= %.4f %.4f %.4f %.4f %.4f %.4f\n",out[0],out[1],out[2],out[3],out[4],out[5]);
  printf("  ref [:6]= %.4f %.4f %.4f %.4f %.4f %.4f\n",ref[0],ref[1],ref[2],ref[3],ref[4],ref[5]);
  return 0;
}
