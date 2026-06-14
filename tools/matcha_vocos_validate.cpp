// Standalone CPU validator for the Matcha-TTS Vocos vocoder ported to ggml.
//
// Loads vocos weights from the matcha gguf (scripts/convert_matcha_onnx_to_gguf.py,
// which gives the pw1/pw2/head Linear weights semantic names), builds the ConvNeXt
// forward graph, runs it on the reference mel fixture (scripts/gen_vocos_fixture.py),
// and compares the magnitude output to the ONNX reference.
//
// Vocos (n_fft=512 -> 257 bins, hidden C=384, 8 ConvNeXt blocks):
//   mels[80,T] -conv_pre(80->384,k7,p3)-> [C,T] -LN(norm_in)->
//   8x { dw conv(C,k7,p3,depthwise) -LN(norm)- pw1(384->1152) -GELU- pw2(1152->384)
//        -*gamma- +residual }
//   -LN(norm_out)- head Linear(384->514) -> [mag_log(257) | phase(257)]
//   mag = exp(mag_log).   (iSTFT tail is separate DSP, reused from cosyvoice3_hift.)
//
// Layout notes (ggml ne[] is reverse of numpy):
//  - Linear weights are numpy [in,out] -> ggml ne=[out,in]; transpose before mul_mat.
//  - conv kernels numpy [Cout,Cin,k] -> ggml ne=[k,Cin,Cout], fed directly to ggml_conv_1d.
//
// Build (CPU only): g++ -std=c++17 matcha_vocos_validate.cpp -I<ggml>/include \
//   -L<libdir> -lggml -lggml-base -lggml-cpu -o vv
// Usage: vv matcha8k.gguf vocos_in_mel.f32 vocos_ref_mag.f32 T
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

static std::vector<float> read_f32(const char* p) {
  FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<float> v(n / 4); if (fread(v.data(), 4, v.size(), f) != v.size()) { exit(1); }
  fclose(f); return v;
}

int main(int argc, char** argv) {
  if (argc < 5) { printf("usage: vv gguf mel.f32 mag.f32 T\n"); return 1; }
  const int T = atoi(argv[4]);
  const int C = 384, NB = 8, BINS = 257, HEAD = 514;

  // load weights (gguf allocates tensor data into wctx)
  ggml_context* wctx = nullptr;
  gguf_init_params gp{ false, &wctx };
  gguf_context* gg = gguf_init_from_file(argv[1], gp);
  if (!gg) { fprintf(stderr, "gguf load failed\n"); return 1; }
  std::map<std::string, ggml_tensor*> W;
  for (ggml_tensor* c = ggml_get_first_tensor(wctx); c; c = ggml_get_next_tensor(wctx, c))
    W[ggml_get_name(c)] = c;
  auto V = [&](const std::string& s) -> ggml_tensor* {
    auto it = W.find(s); if (it == W.end()) { fprintf(stderr, "missing %s\n", s.c_str()); exit(1); }
    return it->second;
  };

  ggml_init_params ip{ (size_t)768 * 1024 * 1024, nullptr, false };
  ggml_context* ctx = ggml_init(ip);

  // mel input as [T, 80] (ne0=T) for ggml_conv_1d (expects [OW, IC])
  ggml_tensor* mel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, 80);

  auto f16 = [&](ggml_tensor* w) {
    return w->type == GGML_TYPE_F32 ? ggml_cast(ctx, w, GGML_TYPE_F16) : w;
  };
  // Linear: x[in,T] -> [out,T]. weight ne=[out,in]; transpose -> ne=[in,out].
  auto linear = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int out) {
    ggml_tensor* wt = ggml_cont(ctx, ggml_transpose(ctx, w));   // [in,out]
    ggml_tensor* y = ggml_mul_mat(ctx, wt, x);                  // [out,T]
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, out, 1));
    return y;
  };
  auto ln = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) { // over ne0
    x = ggml_norm(ctx, x, 1e-6f); x = ggml_mul(ctx, x, w); return ggml_add(ctx, x, b);
  };

  // conv_pre: [T,80] -> [T,384]
  ggml_tensor* h = ggml_conv_1d(ctx, f16(V("voc.conv_pre.weight")), mel, 1, 3, 1);
  h = ggml_add(ctx, h, ggml_reshape_2d(ctx, V("voc.conv_pre.bias"), 1, C));
  // norm_in over channels: need [C,T]
  h = ggml_cont(ctx, ggml_transpose(ctx, h));                   // [384,T]
  h = ln(h, V("voc.norm_in.weight"), V("voc.norm_in.bias"));
  h = ggml_cont(ctx, ggml_transpose(ctx, h));                   // [T,384]

  for (int i = 0; i < NB; i++) {
    std::string b = "voc.blocks." + std::to_string(i) + ".";
    ggml_tensor* res = h;
    ggml_tensor* x = ggml_conv_1d_dw(ctx, f16(V(b + "dw.weight")), h, 1, 3, 1); // [T,384]
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, V(b + "dw.bias"), 1, C));
    x = ggml_cont(ctx, ggml_transpose(ctx, x));                 // [384,T]
    x = ln(x, V(b + "norm.weight"), V(b + "norm.bias"));        // [384,T]
    x = linear(x, V(b + "pw1.weight"), V(b + "pw1.bias"), 1152); // [1152,T]
    x = ggml_gelu_erf(ctx, x);
    x = linear(x, V(b + "pw2.weight"), V(b + "pw2.bias"), C);    // [384,T]
    x = ggml_mul(ctx, x, V(b + "gamma"));                       // per-channel
    x = ggml_cont(ctx, ggml_transpose(ctx, x));                 // [T,384]
    h = ggml_add(ctx, x, res);
  }
  h = ggml_cont(ctx, ggml_transpose(ctx, h));                   // [384,T]
  h = ln(h, V("voc.norm_out.weight"), V("voc.norm_out.bias"));
  ggml_tensor* o = linear(h, V("voc.head.weight"), V("voc.head.bias"), HEAD); // [514,T]

  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, o);

  std::vector<float> melv = read_f32(argv[2]);  // [80,T] row-major
  std::vector<float> melT((size_t)T * 80);
  for (int c = 0; c < 80; c++) for (int tt = 0; tt < T; tt++) melT[tt * 80 + c] = melv[c * T + tt];
  memcpy(mel->data, melT.data(), melT.size() * 4);

  ggml_graph_compute_with_ctx(ctx, gf, 4);

  std::vector<float> ref = read_f32(argv[3]);  // mag [257,T]
  const float* od = (const float*)o->data;     // [514,T] ne0=514
  double err = 0, ssr = 0; int cnt = 0;
  for (int tt = 0; tt < T; tt++)
    for (int k = 0; k < BINS; k++) {
      float mag = std::exp(od[tt * HEAD + k]);
      float r = ref[k * T + tt];
      err += std::fabs(mag - r); ssr += std::fabs(r); cnt++;
    }
  printf("vocos mag: mean|abs err|=%.6f  (ref mean|val|=%.6f)  rel=%.4f  over %d bins\n",
         err / cnt, ssr / cnt, err / (ssr + 1e-9), cnt);
  return 0;
}
