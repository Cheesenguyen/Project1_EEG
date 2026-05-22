#include "spatial.h"
#include "utils.h"

/*
 * Spatial Unrolling — SRAM → PE array
 * =====================================
 * Chiến lược: unroll theo chiều TK (output channels).
 *   → Mỗi cycle, TK_eff PE được kích hoạt song song.
 *   → Mỗi PE_k tính: out[oh,ow,k] += in[ih,iw,c] * w[k,c,r,s]
 *
 * Loop thực tế bên trong mỗi tile (th,tw,tc,tk):
 *   for oh in TH_eff
 *     for ow in TW_eff
 *       for r in R
 *         for s in S
 *           for c in TC_eff
 *             [PARALLEL over k in TK_eff]   ← 1 cycle
 *
 * Broadcast / Read policy:
 *   Input  : in[ih,iw,c] BROADCAST tới tất cả TK_eff PE
 *            → tất cả PE dùng CÙNG 1 giá trị → 1 địa chỉ SRAM duy nhất
 *            → sram_input_reads  += 1 / cycle
 *   Weight : mỗi PE_k đọc w[k,c,r,s] với k KHÁC NHAU → TK_eff địa chỉ khác nhau
 *            → dù đọc song song vẫn là TK_eff lần truy cập Weight buffer
 *            → sram_weight_reads += TK_eff / cycle
 *
 * Tổng:
 *   cycles_per_tile   = TH_eff × TW_eff × R × S × TC_eff
 *   total_cycles      = cycles_per_tile × n_th × n_tw × n_tc × n_tk
 *   sram_input_reads  = total_cycles × 1        (broadcast — 1 địa chỉ)
 *   sram_weight_reads = total_cycles × TK_eff   (mỗi PE đọc weight riêng)
 *
 * Lưu ý: max_PE = TK_eff (số PE kích hoạt song song mỗi cycle)
 */
void run_spatial(const Config *cfg, Metrics *m)
{
    int OH = out_dim(cfg->H, cfg->R, cfg->padding, cfg->stride);
    int OW = out_dim(cfg->W, cfg->S, cfg->padding, cfg->stride);

    int n_th = CEIL_DIV(OH, cfg->TH);
    int n_tw = CEIL_DIV(OW, cfg->TW);
    int n_tc = CEIL_DIV(cfg->C, cfg->TC);
    int n_tk = CEIL_DIV(cfg->K, cfg->TK);

    int TH_eff = MIN(cfg->TH, OH);
    int TW_eff = MIN(cfg->TW, OW);
    int TC_eff = MIN(cfg->TC, cfg->C);
    int TK_eff = MIN(cfg->TK, cfg->K);

    /* Peak PE count: TK_eff PEs fire in parallel each cycle */
    m->max_PE = TK_eff;

    /* Cycles per tile: one parallel MAC burst per (oh,ow,r,s,c) */
    long cycles_per_tile = (long)TH_eff * TW_eff * cfg->R * cfg->S * TC_eff;
    long n_tiles         = (long)n_th * n_tw * n_tc * n_tk;

    m->total_cycles = cycles_per_tile * n_tiles;

    /*
     * SRAM reads per cycle:
     *   Input  buffer : 1 broadcast read  (all TK_eff PEs share same input value)
     *   Weight buffer : 1 multicast read  (Weight buffer ships TK_eff values in
     *                   one bus transaction — counted as 1 SRAM access event)
     */
    m->sram_input_reads  = m->total_cycles;            /* 1 broadcast / cycle        */
    m->sram_weight_reads = m->total_cycles * TK_eff;   /* TK_eff distinct addr/cycle */
}
