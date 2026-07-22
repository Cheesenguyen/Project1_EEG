#include "spatial_thw.h"
#include "utils.h"

/*
 * đọc 16 input khác nhau, 1 weight
 * pe_compute_thw — unroll THW, PE_MAX cố định
 * Dùng td->act_th, act_tw, act_tk, act_tc (kích thước thực tế của tile hiện tại)
 * để đảm bảo metrics đúng cho partial tile.
 *
 * pe_used      = min(act_th*act_tw, PE_MAX)
 * n_pe_batches = ceil(act_th*act_tw / PE_MAX)
 */
void pe_compute_thw(const float *sram_in, const float *sram_w,
                    float *ps_buf,
                    const Config *cfg, const TileDesc *td,
                    Metrics *m)
{
    int act_th = td->act_th, act_tw = td->act_tw;
    int act_tk = td->act_tk, act_tc = td->act_tc;

    long thw          = (long)act_th * act_tw;
    int  pe_used      = (int)MIN(thw, PE_MAX);
    int  n_pe_batches = (int)CEIL_DIV(thw, PE_MAX);
    if (pe_used > m->max_PE) m->max_PE = pe_used;

    switch (cfg->strategy) {
    case OS: m->num_ps_buffers = pe_used; break;
    case WS:
    case IS: m->num_ps_buffers = (long)td->TH_eff * td->TW_eff * td->TK_eff; break;
    }

    switch (cfg->strategy) {

    case OS:
        for (int ki = 0; ki < act_tk; ki++) {
            for (int r = 0; r < cfg->R; r++) {
                for (int s = 0; s < cfg->S; s++) {
                    for (int c = 0; c < act_tc; c++) {
                        float wt_val = sram_w[((ki * cfg->R + r) * cfg->S + s) * td->TC_eff + c];
                        for (int i = 0; i < act_th; i++) {
                            int si = i * cfg->stride + r;
                            for (int j = 0; j < act_tw; j++) {
                                int sj = j * cfg->stride + s;
                                float in_val = sram_in[(si * td->rf_w + sj) * td->TC_eff + c];
                                ps_buf[(i * td->TW_eff + j) * td->TK_eff + ki] += in_val * wt_val;
                            }
                        }
                        m->sram_weight_reads += 1;
                        m->sram_input_reads  += thw;
                        m->total_cycles      += n_pe_batches;
                    }
                }
            }
        }
        break;

    case WS:
        for (int ki = 0; ki < act_tk; ki++) {
            for (int r = 0; r < cfg->R; r++) {
                for (int s = 0; s < cfg->S; s++) {
                    for (int c = 0; c < act_tc; c++) {
                        float wt_val = sram_w[((ki * cfg->R + r) * cfg->S + s) * td->TC_eff + c];
                        for (int i = 0; i < act_th; i++) {
                            int si = i * cfg->stride + r;
                            for (int j = 0; j < act_tw; j++) {
                                int sj = j * cfg->stride + s;
                                float in_val = sram_in[(si * td->rf_w + sj) * td->TC_eff + c];
                                ps_buf[(i * td->TW_eff + j) * td->TK_eff + ki] += in_val * wt_val;
                            }
                        }
                        m->sram_weight_reads += 1;
                        m->sram_input_reads  += thw;
                        m->total_cycles      += n_pe_batches;
                    }
                }
            }
        }
        break;

    case IS:
        for (int c = 0; c < act_tc; c++) {
            m->sram_input_reads += thw;
            for (int r = 0; r < cfg->R; r++) {
                for (int s = 0; s < cfg->S; s++) {
                    for (int ki = 0; ki < act_tk; ki++) {
                        float wt_val = sram_w[((ki * cfg->R + r) * cfg->S + s) * td->TC_eff + c];
                        for (int i = 0; i < act_th; i++) {
                            int si = i * cfg->stride + r;
                            for (int j = 0; j < act_tw; j++) {
                                int sj = j * cfg->stride + s;
                                float in_val = sram_in[(si * td->rf_w + sj) * td->TC_eff + c];
                                ps_buf[(i * td->TW_eff + j) * td->TK_eff + ki] += in_val * wt_val;
                            }
                        }
                        m->sram_weight_reads += 1;
                        m->total_cycles      += n_pe_batches;
                    }
                }
            }
        }
        break;
    }
}