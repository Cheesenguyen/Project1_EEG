#ifndef TILING_H
#define TILING_H
#include "config.h"

typedef struct {
    int OH, OW;
    int n_th, n_tw, n_tc, n_tk;
    int TH_eff, TW_eff, TC_eff, TK_eff;
    int rf_h, rf_w;
    int K_total;          /* = cfg->K, channel stride khi store output */
    /* Kích thước thực tế của tile hiện tại (có thể nhỏ hơn TH_eff/TW_eff/TK_eff
       ở tile cuối khi không chia hết). Được set trong conv_forward trước mỗi
       lần gọi pe_compute. Dùng để đếm metrics đúng.                           */
    int act_th, act_tw, act_tk, act_tc;
} TileDesc;

TileDesc compute_tile(const Config *cfg);

void load_input_tile(const float *dram_input, float *sram_in,
                     const Config *cfg, const TileDesc *td,
                     int th, int tw, int tc);

void load_weight_tile(const float *dram_weight, float *sram_w,
                      const Config *cfg, const TileDesc *td,
                      int tk, int tc);

void store_output_tile(float *dram_output, const float *ps_buf,
                       const TileDesc *td,
                       int th, int tw, int tk,
                       Metrics *m);

#endif