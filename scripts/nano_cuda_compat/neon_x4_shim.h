/* gcc-8.3 lacks the NEON _x2/_x4 vector-load intrinsics (added in gcc-9).
   Modern ggml's arch/arm/quants.c uses them. Define them in terms of single
   vld1q loads of consecutive 16-byte lanes (identical semantics). Injected via
   -include for the x86->aarch64 gcc-8.3 cross-build of RapidSpeech.cpp (gen1). */
#ifndef GGML_NANO_NEON_X4_SHIM_H
#define GGML_NANO_NEON_X4_SHIM_H
#include <arm_neon.h>

/* gcc-8.3 already provides the _x2 variants; only the _x4 loads are missing. */
static inline int8x16x4_t vld1q_s8_x4(const int8_t *p) {
  int8x16x4_t r; r.val[0]=vld1q_s8(p); r.val[1]=vld1q_s8(p+16);
  r.val[2]=vld1q_s8(p+32); r.val[3]=vld1q_s8(p+48); return r; }
static inline uint8x16x4_t vld1q_u8_x4(const uint8_t *p) {
  uint8x16x4_t r; r.val[0]=vld1q_u8(p); r.val[1]=vld1q_u8(p+16);
  r.val[2]=vld1q_u8(p+32); r.val[3]=vld1q_u8(p+48); return r; }

#endif
