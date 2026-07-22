#ifndef SPATIAL_H
#define SPATIAL_H
#include "config.h"
#include "tiling.h"

/*
 * pe_compute_tk — unroll TK, loop order theo strategy
 * ====================================================
 * OS: for oh,ow → for r,s,c → [parallel k]
 * WS: for r,s,c → for oh,ow → [parallel k]
 * IS: for oh,ow,c → for r,s → [parallel k]
 *
 * Đếm: sram_input_reads, sram_weight_reads, total_cycles, max_PE
 */
void pe_compute_tk(const float *sram_in, const float *sram_w,
                   float *ps_buf,
                   const Config *cfg, const TileDesc *td,
                   Metrics *m);

#endif