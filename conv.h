#ifndef CONV_H
#define CONV_H

#include <stdio.h>
#include "config.h"

/*
 * conv_forward — orchestrator vừa chạy thật vừa đếm metrics
 * ===========================================================
 * Luồng: DRAM → SRAM (tiling) → PE compute (spatial+strategy) → PS → DRAM
 * Metrics được đếm trực tiếp trong quá trình chạy.
 *
 * unroll_mode: 0 = unroll TK, 1 = unroll TH×TW
 */
int conv_forward(const Config *cfg,
                 Metrics *m,
                 float **out_input,
                 float **out_weight,
                 float **out_output,
                 int *OH_out, int *OW_out,
                 int unroll_mode);

void conv_print_sample(const float *output, int OH, int OW, int K,
                       const char *label);

void conv_write_stats(FILE *fp, const float *output, int OH, int OW, int K,
                      const char *label, int append);

#endif