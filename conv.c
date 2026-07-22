#include "conv.h"
#include "tiling.h"
#include "spatial.h"
#include "spatial_thw.h"
#include "stationary.h"
#include "utils.h"
#include "metrics.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static float lcg_rand(unsigned int *seed)
{
    *seed = (*seed) * 1664525u + 1013904223u;
    return (float)(*seed & 0xFFFF) / 65535.0f - 0.5f;
}

/*
 * conv_forward
 * Orchestrator — vừa chạy thật vừa đếm metrics trực tiếp.
 *
 * Flow:
 *   for th, tw, tk:
 *     ps_buf_init()
 *     for tc:
 *       load_input_tile()   -> dram_input_loads (phần tử), sram_input_bytes
 *       load_weight_tile()  -> dram_weight_loads (phần tử), sram_weight_bytes
 *       pe_compute_*(strategy, unroll) -> sram_*_reads, total_cycles, max_PE
 *     store_output_tile()   -> dram_stores++
 *
 *   m->dram_loads      = dram_input_loads + dram_weight_loads
 *   m->sram_total_bytes = sram_input_bytes + sram_weight_bytes
 */
int conv_forward(const Config *cfg,
                 Metrics *m,
                 float **out_input,
                 float **out_weight,
                 float **out_output,
                 int *OH_out, int *OW_out,
                 int unroll_mode)
{
    metrics_reset(m);

    TileDesc td = compute_tile(cfg);
    *OH_out = td.OH;
    *OW_out = td.OW;

    // cấp phát bộ nhớ cho dram, sram
    long input_sz  = (long)cfg->P * cfg->Q * cfg->C;
    long weight_sz = (long)cfg->K * cfg->R * cfg->S * cfg->C;
    long output_sz = (long)td.OH  * td.OW  * cfg->K;

    float *dram_input  = (float *)malloc(input_sz  * sizeof(float));
    float *dram_weight = (float *)malloc(weight_sz * sizeof(float));
    float *dram_output = (float *)calloc(output_sz,  sizeof(float));

    long sram_in_sz = (long)td.rf_h * td.rf_w * td.TC_eff;
    long sram_w_sz  = (long)td.TK_eff * cfg->R * cfg->S * td.TC_eff;
    long ps_sz      = (long)td.TH_eff * td.TW_eff * td.TK_eff;

    float *sram_in = (float *)malloc(sram_in_sz * sizeof(float));
    float *sram_w  = (float *)malloc(sram_w_sz  * sizeof(float));
    float *ps_buf  = (float *)malloc(ps_sz       * sizeof(float));

    if (!dram_input || !dram_weight || !dram_output ||
        !sram_in || !sram_w || !ps_buf) {
        free(dram_input); free(dram_weight); free(dram_output);
        free(sram_in); free(sram_w); free(ps_buf);
        return -1;
    }

    unsigned int seed = 42u;
    for (long i = 0; i < input_sz;  i++) dram_input [i] = lcg_rand(&seed);
    for (long i = 0; i < weight_sz; i++) dram_weight[i] = lcg_rand(&seed);

    /* PS buffer size — cố định theo tile, không phụ thuộc strategy */

    /*
     * Main simulation loop: th  tw  tk  tc
     * Metrics được đếm trực tiếp trong các hàm con.
     */
    for (int th = 0; th < td.n_th; th++) {
        td.act_th = MIN(td.TH_eff, td.OH - th * cfg->TH);
        for (int tw = 0; tw < td.n_tw; tw++) {
            td.act_tw = MIN(td.TW_eff, td.OW - tw * cfg->TW);
            for (int tk = 0; tk < td.n_tk; tk++) {
                td.act_tk = MIN(td.TK_eff, cfg->K - tk * cfg->TK);

                ps_buf_init(ps_buf, &td);
  
                for (int tc = 0; tc < td.n_tc; tc++) {
                    td.act_tc = MIN(td.TC_eff, cfg->C - tc * cfg->TC);

                    /* DRAM -> SRAM */
                    load_input_tile (dram_input,  sram_in, cfg, &td, th, tw, tc);
                    load_weight_tile(dram_weight, sram_w,  cfg, &td, tk, tc);

                    /* SRAM -> PE -> PS: loop order theo strategy + unroll mode */
                    if (unroll_mode == 0)
                        pe_compute_tk (sram_in, sram_w, ps_buf, cfg, &td, m);
                    else
                        pe_compute_thw(sram_in, sram_w, ps_buf, cfg, &td, m);
                }

                /* PS -> DRAM */
                store_output_tile(dram_output, ps_buf, &td, th, tw, tk, m);
            }
        }
    }

    /* Tổng hợp metrics theo strategy.
     *
     * OS: input và weight đều load lại cho mỗi tile (th,tw,tk,tc)
     * WS: weight reuse qua th,tw -> load n_tk×n_tc lần; input load n_all lần
     * IS: input reuse qua tk -> load n_th×n_tw×n_tc lần; weight load n_all lần
     * dram_stores: đếm trực tiếp trong store_output_tile — đúng.
     */
    long tile_input_elems  = (long)td.rf_h * td.rf_w * td.TC_eff;
    long tile_weight_elems = (long)td.TK_eff * cfg->R * cfg->S * td.TC_eff;
    long n_spatial = (long)td.n_th * td.n_tw;
    long n_all     = n_spatial * td.n_tc * td.n_tk;

    switch (cfg->strategy) {
    case OS:
        m->dram_input_loads  = n_all * tile_input_elems;
        m->dram_weight_loads = n_all * tile_weight_elems;
        break;
    case WS:
        m->dram_input_loads  = n_all * tile_input_elems;
        m->dram_weight_loads = (long)td.n_tk * td.n_tc * tile_weight_elems;
        break;
    case IS:
        m->dram_input_loads  = (long)td.n_th * td.n_tw * td.n_tc * tile_input_elems;
        m->dram_weight_loads = n_all * tile_weight_elems;
        break;
    }
    m->dram_loads        = m->dram_input_loads + m->dram_weight_loads;
    m->sram_input_bytes  = m->dram_input_loads  * 4;
    m->sram_weight_bytes = m->dram_weight_loads * 4;
    m->sram_total_bytes  = m->sram_input_bytes  + m->sram_weight_bytes;

    free(sram_in); free(sram_w); free(ps_buf);

    *out_input  = dram_input;
    *out_weight = dram_weight;
    *out_output = dram_output;
    return 0;
}

// xuất dữ liệu thống kê min max mean,.. ra file csv
void conv_write_stats(FILE *fp, const float *output, int OH, int OW, int K,
                      const char *label, int append)
{
    if (!append)
        fprintf(fp, "label,channel_k,min,max,mean,sum\n");

    long spatial = (long)OH * OW;
    for (int k = 0; k < K; k++) {
        float vmin = output[k], vmax = output[k], vsum = 0.0f;
        for (long i = 0; i < spatial; i++) {
            float v = output[i * K + k];
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
            vsum += v;
        }
        fprintf(fp, "\"%s\",%d,%.6f,%.6f,%.6f,%.6f\n",
                label, k, vmin, vmax, vsum/(float)spatial, vsum);
    }
}

// in ra terminal
void conv_print_sample(const float *output, int OH, int OW, int K,
                       const char *label)
{
    printf("\n  [Conv output — %s]\n", label);
    float vmin = output[0], vmax = output[0], vsum = 0.0f;
    long  cnt  = (long)OH * OW;
    for (long i = 0; i < cnt; i++) {
        float v = output[i * K];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        vsum += v;
    }
    printf("    Output shape : %d x %d x %d\n", OH, OW, K);
    printf("    Ch-0 stats   : min=%.4f  max=%.4f  mean=%.4f\n",
           vmin, vmax, vsum / cnt);

    int rows = (OH < 3) ? OH : 3;
    int cols = (OW < 3) ? OW : 3;
    printf("    Top-left %dx%d corner (k=0):\n", rows, cols);
    for (int oh = 0; oh < rows; oh++) {
        printf("      ");
        for (int ow = 0; ow < cols; ow++)
            printf("%10.4f ", output[(oh * OW + ow) * K]);
        printf("\n");
    }
}