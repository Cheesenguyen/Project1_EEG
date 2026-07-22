#include "tiling.h"
#include "utils.h"
#include <string.h>

TileDesc compute_tile(const Config *cfg)
{
    TileDesc td;
    td.OH     = cfg->H;
    td.OW     = cfg->W;
    td.n_th   = CEIL_DIV(td.OH, cfg->TH);
    td.n_tw   = CEIL_DIV(td.OW, cfg->TW);
    td.n_tc   = CEIL_DIV(cfg->C, cfg->TC);
    td.n_tk   = CEIL_DIV(cfg->K, cfg->TK);
    td.TH_eff = MIN(cfg->TH, td.OH);
    td.TW_eff = MIN(cfg->TW, td.OW);
    td.TC_eff = MIN(cfg->TC, cfg->C);
    td.TK_eff = MIN(cfg->TK, cfg->K);
    td.rf_h   = (td.TH_eff - 1) * cfg->stride + cfg->R;
    td.rf_w   = (td.TW_eff - 1) * cfg->stride + cfg->S;
    td.K_total = cfg->K;
    return td;
}

/*
 * load_input_tile
 * DRAM -> SRAM: copy vùng input tương ứng tile (th,tw,tc).
 * Đếm: dram_input_loads (số phần tử), sram_input_bytes.
 */
void load_input_tile(const float *dram_input, float *sram_in,
                     const Config *cfg, const TileDesc *td,
                     int th, int tw, int tc)
{
    int oh_start = th * cfg->TH;
    int ow_start = tw * cfg->TW;
    int ih_start = oh_start * cfg->stride - cfg->padding;
    int iw_start = ow_start * cfg->stride - cfg->padding;
    int c_start  = tc * cfg->TC;
    int P = cfg->P, Q = cfg->Q;

    for (int i = 0; i < td->rf_h; i++) {
        int ih = ih_start + i;
        for (int j = 0; j < td->rf_w; j++) {
            int iw = iw_start + j;
            for (int c = 0; c < td->TC_eff; c++) {
                float val = 0.0f;
                if (ih >= 0 && ih < P && iw >= 0 && iw < Q)
                    val = dram_input[(ih * Q + iw) * cfg->C + (c_start + c)];
                sram_in[(i * td->rf_w + j) * td->TC_eff + c] = val;
            }
        }
    }

    /* dram_input_loads và sram_input_bytes tính trong conv_forward */
}

/*
 * load_weight_tile
 * DRAM -> SRAM: copy weight tile (tk,tc).
 * Đếm: dram_weight_loads (số phần tử), sram_weight_bytes tính .
 */
void load_weight_tile(const float *dram_weight, float *sram_w,
                      const Config *cfg, const TileDesc *td,
                      int tk, int tc)
{
    int k_start   = tk * cfg->TK;
    int c_start   = tc * cfg->TC;
    int actual_tk = MIN(td->TK_eff, cfg->K - k_start);  
    int actual_tc = MIN(td->TC_eff, cfg->C - c_start);

    for (int ki = 0; ki < actual_tk; ki++) {
        int k = k_start + ki;
        for (int r = 0; r < cfg->R; r++) {
            for (int s = 0; s < cfg->S; s++) {
                for (int c = 0; c < actual_tc; c++) {
                    sram_w[((ki * cfg->R + r) * cfg->S + s) * td->TC_eff + c] =
                        dram_weight[((k * cfg->R + r) * cfg->S + s) * cfg->C + (c_start + c)];
                }
            }
        }
    }

    /* dram_weight_loads và sram_weight_bytes tính trong conv_forward */
}

/*
 * store_output_tile
 * PS -> DRAM: ghi output tile (th,tw,tk).
 * Đếm: dram_stores++.
 */
void store_output_tile(float *dram_output, const float *ps_buf,
                       const TileDesc *td,
                       int th, int tw, int tk,
                       Metrics *m)
{
    int oh_start  = th * td->TH_eff;
    int ow_start  = tw * td->TW_eff;
    int k_start   = tk * td->TK_eff;
    int actual_th = MIN(td->TH_eff, td->OH      - oh_start);  
    int actual_tw = MIN(td->TW_eff, td->OW      - ow_start);
    int actual_tk = MIN(td->TK_eff, td->K_total - k_start);

    for (int i = 0; i < actual_th; i++) {
        for (int j = 0; j < actual_tw; j++) {
            for (int ki = 0; ki < actual_tk; ki++) {
                int oh = oh_start + i;
                int ow = ow_start + j;
                int k  = k_start  + ki;
                dram_output[(oh * td->OW + ow) * td->K_total + k] =
                    ps_buf[(i * td->TW_eff + j) * td->TK_eff + ki];
            }
        }
    }

    m->dram_stores++;
}