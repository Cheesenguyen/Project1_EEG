#ifndef CONV_H
#define CONV_H

#include <stdio.h>
#include "config.h"

/*
 * conv_forward — thực thi phép tính conv thật sự
 * ================================================
 * Cấp phát input/weight ngẫu nhiên, tính output, trả về con trỏ.
 * Caller chịu trách nhiệm free() ba mảng sau khi dùng xong.
 *
 * Trả về 0 nếu thành công, -1 nếu lỗi cấp phát bộ nhớ.
 *
 * Output dimensions:
 *   OH = (H + 2*padding - R) / stride + 1
 *   OW = (W + 2*padding - S) / stride + 1
 *   output shape: [OH][OW][K]  (NHWC-style, N=1)
 */
int conv_forward(const Config *cfg,
                 float **out_input,    /* [H * W * C]       — filled with rand */
                 float **out_weight,   /* [K * R * S * C]   — filled with rand */
                 float **out_output,   /* [OH * OW * K]     — computed result  */
                 int *OH_out, int *OW_out);

/* Print a tiny corner of the output for sanity-check */
void conv_print_sample(const float *output, int OH, int OW, int K,
                       const char *label);

/*
 * conv_write_stats — ghi min/max/mean/sum của từng channel k ra file CSV
 * Format: label, k, min, max, mean, sum
 * append=0 → ghi header rồi ghi data; append=1 → chỉ ghi data (tiếp tục file)
 */
void conv_write_stats(FILE *fp, const float *output, int OH, int OW, int K,
                      const char *label, int append);

#endif /* CONV_H */