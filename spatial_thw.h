#ifndef SPATIAL_THW_H
#define SPATIAL_THW_H
#include "config.h"
#include "tiling.h"

/*
 * pe_compute_thw — unroll TH×TW, loop order theo strategy
 * OS: for k → for r,s,c -> [parallel oh,ow]
 * WS: for k,r,s,c -> [parallel oh,ow]
 * IS: for c → for r,s,k -> [parallel oh,ow]
 *
 * Đếm: sram_input_reads, sram_weight_reads, total_cycles, max_PE
 */
void pe_compute_thw(const float *sram_in, const float *sram_w,
                    float *ps_buf,
                    const Config *cfg, const TileDesc *td,
                    Metrics *m);

#endif