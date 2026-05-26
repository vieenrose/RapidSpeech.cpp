#pragma once

#include <ggml.h>
#include <gguf.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// Custom mixed-precision ftypes — NOT in ggml.h.
// Use values outside ggml's current enumerators but inside the underlying-type
// range (clang infers [-32, 31] from the existing enumerators, so values must
// stay within that). 5/6 sit in the 4→7 gap; 28 is the next free slot above
// the current top (Q1_0 = 27).
#define GGML_FTYPE_MOSTLY_Q4_K_M  ((enum ggml_ftype)5)
#define GGML_FTYPE_MOSTLY_Q5_K_M  ((enum ggml_ftype)6)
#define GGML_FTYPE_MOSTLY_Q3_K_M  ((enum ggml_ftype)28)

// Parse quantization type string (e.g. "q4_k") to ggml_ftype enum
enum ggml_ftype ggml_parse_ftype(const char *str);

// Print supported quantization types to file pointer
void ggml_print_ftypes(FILE *fp = stderr);

// Quantize a GGUF model: read fname_inp, quantize weight tensors, write
// fname_out.  If imatrix_file is non-empty, load it and pass per-column
// importance weights to ggml_quantize_chunk for activation-aware quantization.
bool rapid_speech_ggml_quantize(ggml_context *ctx, gguf_context *gguf_input,
                                const std::string &fname_inp,
                                const std::string &fname_out,
                                const ggml_ftype ftype, const int nthread,
                                const std::vector<std::string> &to_quant,
                                const std::vector<std::string> &to_skip,
                                const std::string &imatrix_file = "");
