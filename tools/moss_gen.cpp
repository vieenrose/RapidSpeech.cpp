// ggml generation loop (frame-0 validation): prefill(prompt) [KV] -> local greedy
// frame (text token + 16 codebooks w/ real feedback) -> compare vs ONNX gg_frames[0].
// Reuses verified KV global-decode + local block. Embedding/head lookups in plain C++.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

static const int H=768,NH=12,HD=64,NL=12,FF=3072,VOCAB=16384,CB=1024,NCB=16;
static const int AUDIO_ASSIST=9, AUDIO_END=7;
static const float EPS=1e-5f,ROPE=10000.0f;

static std::vector<float> lb(const char*p,size_t n){std::vector<float> v(n);FILE*f=fopen(p,"rb");if(!f){printf("open %s\n",p);exit(1);}fread(v.data(),4,n,f);fclose(f);return v;}

static ggml_backend_t be;
static ggml_context* wctx;
static ggml_tensor* Kc[NL]; static ggml_tensor* Vc[NL];
static ggml_tensor* Wt(const std::string&n){ggml_tensor*t=ggml_get_tensor(wctx,n.c_str());if(!t){printf("missing %s\n",n.c_str());exit(1);}return t;}

// --- global transformer forward with KV cache (verified in moss_kv.cpp) ---
static std::vector<float> run_global(const std::vector<float>&emb,int n_new,int n_past){
  int n_kv=n_past+n_new;
  ggml_init_params cp{(size_t)128*1024*1024,nullptr,true}; ggml_context*ctx=ggml_init(cp);
  auto LN=[&](ggml_tensor*x,ggml_tensor*w,ggml_tensor*b){return ggml_add(ctx,ggml_mul(ctx,ggml_norm(ctx,x,EPS),w),b);};
  ggml_tensor*inp=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,H,n_new); ggml_set_input(inp);
  ggml_tensor*pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,n_new); ggml_set_input(pos);
  ggml_tensor*h=inp; ggml_cgraph*gf=ggml_new_graph(ctx);
  for(int i=0;i<NL;i++){
    std::string p="blk."+std::to_string(i)+".";
    ggml_tensor*ln1=LN(h,Wt(p+"attn_norm.weight"),Wt(p+"attn_norm.bias"));
    ggml_tensor*qkv=ggml_add(ctx,ggml_mul_mat(ctx,Wt(p+"attn_qkv.weight"),ln1),Wt(p+"attn_qkv.bias"));
    ggml_tensor*q=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_view_2d(ctx,qkv,H,n_new,qkv->nb[1],0)),HD,NH,n_new);
    ggml_tensor*k=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_view_2d(ctx,qkv,H,n_new,qkv->nb[1],H*4)),HD,NH,n_new);
    ggml_tensor*v=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_view_2d(ctx,qkv,H,n_new,qkv->nb[1],2*H*4)),HD,NH,n_new);
    q=ggml_rope_ext(ctx,q,pos,nullptr,HD,0,0,ROPE,1,0,1,0,0);
    k=ggml_rope_ext(ctx,k,pos,nullptr,HD,0,0,ROPE,1,0,1,0,0);
    ggml_tensor*kdst=ggml_view_3d(ctx,Kc[i],HD,NH,n_new,Kc[i]->nb[1],Kc[i]->nb[2],(size_t)n_past*Kc[i]->nb[2]);
    ggml_tensor*vdst=ggml_view_3d(ctx,Vc[i],HD,NH,n_new,Vc[i]->nb[1],Vc[i]->nb[2],(size_t)n_past*Vc[i]->nb[2]);
    ggml_build_forward_expand(gf,ggml_cpy(ctx,k,kdst));
    ggml_build_forward_expand(gf,ggml_cpy(ctx,v,vdst));
    ggml_tensor*Kall=ggml_view_3d(ctx,Kc[i],HD,NH,n_kv,Kc[i]->nb[1],Kc[i]->nb[2],0);
    ggml_tensor*Vall=ggml_view_3d(ctx,Vc[i],HD,NH,n_kv,Vc[i]->nb[1],Vc[i]->nb[2],0);
    ggml_tensor*kq=ggml_soft_max(ctx,ggml_diag_mask_inf(ctx,ggml_scale(ctx,ggml_mul_mat(ctx,ggml_permute(ctx,Kall,0,2,1,3),ggml_permute(ctx,q,0,2,1,3)),1.0f/sqrtf(HD)),n_past));
    ggml_tensor*vt=ggml_cont(ctx,ggml_permute(ctx,Vall,1,2,0,3));
    ggml_tensor*kqv=ggml_permute(ctx,ggml_mul_mat(ctx,vt,kq),0,2,1,3);
    ggml_tensor*att=ggml_add(ctx,ggml_mul_mat(ctx,Wt(p+"attn_out.weight"),ggml_cont_2d(ctx,kqv,H,n_new)),Wt(p+"attn_out.bias"));
    h=ggml_add(ctx,h,att);
    ggml_tensor*ln2=LN(h,Wt(p+"ffn_norm.weight"),Wt(p+"ffn_norm.bias"));
    ggml_tensor*up=ggml_gelu(ctx,ggml_add(ctx,ggml_mul_mat(ctx,Wt(p+"ffn_up.weight"),ln2),Wt(p+"ffn_up.bias")));
    h=ggml_add(ctx,h,ggml_add(ctx,ggml_mul_mat(ctx,Wt(p+"ffn_down.weight"),up),Wt(p+"ffn_down.bias")));
  }
  h=LN(h,Wt("output_norm.weight"),Wt("output_norm.bias")); ggml_set_output(h);
  ggml_build_forward_expand(gf,h);
  ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); ggml_gallocr_alloc_graph(ga,gf);
  ggml_backend_tensor_set(inp,emb.data(),0,(size_t)H*n_new*4);
  std::vector<int32_t> pv(n_new); for(int j=0;j<n_new;j++) pv[j]=n_past+j;
  ggml_backend_tensor_set(pos,pv.data(),0,n_new*4);
  ggml_backend_graph_compute(be,gf);
  std::vector<float> out((size_t)H*n_new); ggml_backend_tensor_get(h,out.data(),0,out.size()*4);
  ggml_gallocr_free(ga); ggml_free(ctx);
  return out;  // [H, n_new] col-major (last col = out + (n_new-1)*H)
}

// --- local transformer (1 layer) over L positions, return last-position hidden [H] ---
static std::vector<float> run_local(const std::vector<float>&seq,int L){
  ggml_init_params cp{(size_t)32*1024*1024,nullptr,true}; ggml_context*ctx=ggml_init(cp);
  auto LN=[&](ggml_tensor*x,ggml_tensor*w,ggml_tensor*b){return ggml_add(ctx,ggml_mul(ctx,ggml_norm(ctx,x,EPS),w),b);};
  ggml_tensor*inp=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,H,L); ggml_set_input(inp);
  ggml_tensor*pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,L); ggml_set_input(pos);
  std::string p="local.blk.0."; ggml_tensor*h=inp;
  ggml_tensor*ln1=LN(h,Wt(p+"attn_norm.weight"),Wt(p+"attn_norm.bias"));
  ggml_tensor*qkv=ggml_add(ctx,ggml_mul_mat(ctx,Wt(p+"attn_qkv.weight"),ln1),Wt(p+"attn_qkv.bias"));
  ggml_tensor*q=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_view_2d(ctx,qkv,H,L,qkv->nb[1],0)),HD,NH,L);
  ggml_tensor*k=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_view_2d(ctx,qkv,H,L,qkv->nb[1],H*4)),HD,NH,L);
  ggml_tensor*v=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_view_2d(ctx,qkv,H,L,qkv->nb[1],2*H*4)),HD,NH,L);
  q=ggml_rope_ext(ctx,q,pos,nullptr,HD,0,0,ROPE,1,0,1,0,0); k=ggml_rope_ext(ctx,k,pos,nullptr,HD,0,0,ROPE,1,0,1,0,0);
  ggml_tensor*kq=ggml_soft_max(ctx,ggml_diag_mask_inf(ctx,ggml_scale(ctx,ggml_mul_mat(ctx,ggml_permute(ctx,k,0,2,1,3),ggml_permute(ctx,q,0,2,1,3)),1.0f/sqrtf(HD)),0));
  ggml_tensor*vt=ggml_cont(ctx,ggml_permute(ctx,v,1,2,0,3));
  ggml_tensor*kqv=ggml_permute(ctx,ggml_mul_mat(ctx,vt,kq),0,2,1,3);
  ggml_tensor*att=ggml_add(ctx,ggml_mul_mat(ctx,Wt(p+"attn_out.weight"),ggml_cont_2d(ctx,kqv,H,L)),Wt(p+"attn_out.bias"));
  h=ggml_add(ctx,h,att);
  ggml_tensor*ln2=LN(h,Wt(p+"ffn_norm.weight"),Wt(p+"ffn_norm.bias"));
  ggml_tensor*up=ggml_gelu(ctx,ggml_add(ctx,ggml_mul_mat(ctx,Wt(p+"ffn_up.weight"),ln2),Wt(p+"ffn_up.bias")));
  h=ggml_add(ctx,h,ggml_add(ctx,ggml_mul_mat(ctx,Wt(p+"ffn_down.weight"),up),Wt(p+"ffn_down.bias")));
  h=LN(h,Wt("local.output_norm.weight"),Wt("local.output_norm.bias")); ggml_set_output(h);
  ggml_cgraph*gf=ggml_new_graph(ctx); ggml_build_forward_expand(gf,h);
  ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); ggml_gallocr_alloc_graph(ga,gf);
  ggml_backend_tensor_set(inp,seq.data(),0,(size_t)H*L*4);
  std::vector<int32_t> pv(L); for(int j=0;j<L;j++) pv[j]=j;
  ggml_backend_tensor_set(pos,pv.data(),0,L*4);
  ggml_backend_graph_compute(be,gf);
  std::vector<float> last(H); ggml_backend_tensor_get(h,last.data(),0,H*4);  // gallocr may pack; last col
  // read last column: hidden is [H, L], last col at offset (L-1)*H
  std::vector<float> full((size_t)H*L); ggml_backend_tensor_get(h,full.data(),0,full.size()*4);
  std::vector<float> out(full.begin()+(size_t)(L-1)*H, full.end());
  ggml_gallocr_free(ga); ggml_free(ctx);
  return out;
}

int main(int argc,char**argv){
  const char*gg=argc>1?argv[1]:"/home/luigi/moss-port/moss_nano_full.gguf";
  gguf_init_params gp{false,&wctx}; if(!gguf_init_from_file(gg,gp)){printf("gguf\n");return 1;}
  be=ggml_backend_cpu_init();
  ggml_init_params kp{(size_t)(2*NL+4)*ggml_tensor_overhead(),nullptr,true}; ggml_context*kvc=ggml_init(kp);
  for(int i=0;i<NL;i++){Kc[i]=ggml_new_tensor_3d(kvc,GGML_TYPE_F32,HD,NH,512);Vc[i]=ggml_new_tensor_3d(kvc,GGML_TYPE_F32,HD,NH,512);}
  ggml_backend_alloc_ctx_tensors(kvc,be);

  // embedding tables (raw f32)
  std::vector<float> TOK=lb("/home/luigi/moss-port/gg_tok_embd.bin",(size_t)VOCAB*H);      // [16384,768]
  std::vector<float> AUD=lb("/home/luigi/moss-port/gg_audio_embd.bin",(size_t)NCB*CB*H);   // [16,1024,768]
  auto tok_emb=[&](int id){return &TOK[(size_t)id*H];};
  auto aud_emb=[&](int k,int c){return &AUD[((size_t)k*CB+c)*H];};
  // logits = W(rows,H) . h ; argmax
  auto argmax_logit=[&](const float*Wm,int rows,const float*h){int best=0;float bm=-1e30f;for(int r=0;r<rows;r++){const float*wr=Wm+(size_t)r*H;float d=0;for(int j=0;j<H;j++)d+=wr[j]*h[j];if(d>bm){bm=d;best=r;}}return best;};

  int L=177;
  std::vector<float> pe=lb("/home/luigi/moss-port/gg_prompt_emb.bin",(size_t)L*H);
  std::vector<float> gh=run_global(pe,L,0);              // prefill, KV filled [0..L)
  std::vector<float> hidden(gh.begin()+(size_t)(L-1)*H, gh.end());  // last position [H]

  // ---- frame 0: local greedy ----
  std::vector<float> lseq(hidden);                       // pos0 = global hidden
  int Lc=1;
  std::vector<float> hl=run_local(lseq,Lc);
  int text_tok=argmax_logit(TOK.data(),VOCAB,hl.data()); // text_lm_head tied = token_embd
  printf("frame0 predicted text_tok=%d (assist=%d end=%d)\n",text_tok,AUDIO_ASSIST,AUDIO_END);
  std::vector<int> frame;
  const float* cur=tok_emb(text_tok);
  for(int k=0;k<NCB;k++){
    lseq.insert(lseq.end(),cur,cur+H); Lc++;
    hl=run_local(lseq,Lc);
    int cb=argmax_logit(aud_emb(k,0),CB,hl.data());      // audio_lm_heads[k] tied = audio_embd[k]
    frame.push_back(cb);
    cur=aud_emb(k,cb);
  }
  // load gg_frames row 0 (int32) — read raw after npy header is messy; compare printed
  printf("ggml frame0: "); for(int c=0;c<NCB;c++) printf("%d ",frame[c]); printf("\n");
  printf("ref  frame0: 55 709 378 1019 128 800 794 632 682 556 585 76 245 636 225 226\n");
  return 0;
}
