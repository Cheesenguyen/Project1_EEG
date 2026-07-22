#include "spatial.h"
#include "utils.h"

/*
 * đọc 1 input và 16 giá trị weight độc lập
 * pe_compute_tk — unroll TK, PE_MAX cố định
 * Dùng td->act_th, act_tw, act_tk, act_tc (kích thước thực tế của tile hiện tại)
 * thay vì TH_eff/TW_eff/TK_eff để đảm bảo metrics đúng cho tile cuối (partial tile).
 *
 * pe_used      = min(act_tk, PE_MAX)
 * n_pe_batches = ceil(act_tk / PE_MAX)
 */
void pe_compute_tk(const float *sram_in, const float *sram_w,
                   float *ps_buf,
                   const Config *cfg, const TileDesc *td,
                   Metrics *m)
{
    int act_th = td->act_th, act_tw = td->act_tw;
    int act_tk = td->act_tk, act_tc = td->act_tc;

    int pe_used      = MIN(act_tk, PE_MAX);
    int n_pe_batches = CEIL_DIV(act_tk, PE_MAX);
    if (pe_used > m->max_PE) m->max_PE = pe_used;

    switch (cfg->strategy) {
    case OS:
    case WS: m->num_ps_buffers = pe_used; break;
    case IS: m->num_ps_buffers = (long)td->TH_eff * td->TW_eff * td->TK_eff; break;
    }

    switch (cfg->strategy) {

    case OS:
        for (int i = 0; i < act_th; i++) {
            for (int j = 0; j < act_tw; j++) {
                for (int r = 0; r < cfg->R; r++) {
                    int si = i * cfg->stride + r;
                    for (int s = 0; s < cfg->S; s++) {
                        int sj = j * cfg->stride + s;
                        for (int c = 0; c < act_tc; c++) {
                            float in_val = sram_in[(si * td->rf_w + sj) * td->TC_eff + c];
                            for (int ki = 0; ki < act_tk; ki++) {
                                float wt_val = sram_w[((ki * cfg->R + r) * cfg->S + s) * td->TC_eff + c];
                                ps_buf[(i * td->TW_eff + j) * td->TK_eff + ki] += in_val * wt_val;
                            }
                            m->sram_input_reads  += 1;
                            m->sram_weight_reads += act_tk;
                            m->total_cycles      += n_pe_batches;
                        }
                    }
                }
            }
        }
        break;

    case WS:
        for (int r = 0; r < cfg->R; r++) {
            for (int s = 0; s < cfg->S; s++) {
                for (int c = 0; c < act_tc; c++) {
                    m->sram_weight_reads += act_tk;
                    for (int i = 0; i < act_th; i++) {
                        int si = i * cfg->stride + r;
                        for (int j = 0; j < act_tw; j++) {
                            int sj = j * cfg->stride + s;
                            float in_val = sram_in[(si * td->rf_w + sj) * td->TC_eff + c];
                            for (int ki = 0; ki < act_tk; ki++) {
                                float wt_val = sram_w[((ki * cfg->R + r) * cfg->S + s) * td->TC_eff + c];
                                ps_buf[(i * td->TW_eff + j) * td->TK_eff + ki] += in_val * wt_val;
                            }
                            m->sram_input_reads += 1;
                            m->total_cycles     += n_pe_batches;
                        }
                    }
                }
            }
        }
        break;

    case IS:
        for (int i = 0; i < act_th; i++) {
            for (int j = 0; j < act_tw; j++) {
                for (int c = 0; c < act_tc; c++) {
                    m->sram_input_reads += 1;
                    for (int r = 0; r < cfg->R; r++) {
                        int si = i * cfg->stride + r;
                        for (int s = 0; s < cfg->S; s++) {
                            int sj = j * cfg->stride + s;
                            float in_val = sram_in[(si * td->rf_w + sj) * td->TC_eff + c];
                            for (int ki = 0; ki < act_tk; ki++) {
                                float wt_val = sram_w[((ki * cfg->R + r) * cfg->S + s) * td->TC_eff + c];
                                ps_buf[(i * td->TW_eff + j) * td->TK_eff + ki] += in_val * wt_val;
                            }
                            m->sram_weight_reads += act_tk;
                            m->total_cycles      += n_pe_batches;
                        }
                    }
                }
            }
        }
        break;
    }
}