#include "spatial_thw.h"
#include "utils.h"

/*
 * Spatial Unrolling — unroll theo TH × TW (vị trí không gian)
 * =============================================================
 * Chiến lược: unroll đồng thời theo cả 2 chiều TH và TW.
 *   → Mỗi cycle, TH_eff × TW_eff PE được kích hoạt song song.
 *   → Mỗi PE_(oh,ow) tính: out[oh,ow,k] += in[ih,iw,c] * w[k,c,r,s]
 *     với (ih,iw) phụ thuộc vào vị trí (oh,ow) của PE đó.
 *
 * Loop thực tế bên trong mỗi tile (th,tw,tc,tk):
 *   for k  in TK_eff
 *     for r  in R
 *       for s  in S
 *         for c  in TC_eff
 *           [PARALLEL over (oh,ow) in TH_eff × TW_eff]  ← 1 cycle
 *
 * Broadcast / Read policy:
 *   Weight : w[k,c,r,s] BROADCAST tới tất cả TH_eff×TW_eff PE
 *            → tất cả PE dùng CÙNG 1 weight → 1 địa chỉ SRAM duy nhất
 *            → sram_weight_reads += 1 / cycle
 *   Input  : mỗi PE_(oh,ow) đọc in[oh*stride+r-pad, ow*stride+s-pad, c]
 *            → mỗi PE đọc địa chỉ input KHÁC NHAU
 *            → sram_input_reads  += TH_eff × TW_eff / cycle
 *
 * Tổng:
 *   cycles_per_tile   = TK_eff × R × S × TC_eff
 *   total_cycles      = cycles_per_tile × n_th × n_tw × n_tc × n_tk
 *   sram_weight_reads = total_cycles × 1                 (broadcast)
 *   sram_input_reads  = total_cycles × TH_eff × TW_eff  (mỗi PE đọc riêng)
 *
 * Lưu ý: max_PE = TH_eff × TW_eff
 */
void run_spatial_thw(const Config *cfg, Metrics *m)
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

    /* Peak PE count: TH_eff × TW_eff PEs fire in parallel each cycle */
    m->max_PE = (long)TH_eff * TW_eff;

    /*
     * Cycles per tile:
     *   TH và TW đã được unroll ra ngoài (chạy song song trong 1 cycle)
     *   nên chỉ còn TK_eff × R × S × TC_eff bước tuần tự
     */
    long cycles_per_tile = (long)TK_eff * cfg->R * cfg->S * TC_eff;
    long n_tiles         = (long)n_th * n_tw * n_tc * n_tk;

    m->total_cycles = cycles_per_tile * n_tiles;

    /*
     * SRAM reads per cycle:
     *   Weight buffer : 1 broadcast read  (tất cả PE dùng cùng w[k,c,r,s])
     *   Input  buffer : TH_eff×TW_eff reads (mỗi PE đọc ô input khác nhau)
     */
    m->sram_weight_reads = m->total_cycles;                          /* 1 broadcast / cycle       */
    m->sram_input_reads  = m->total_cycles * (long)TH_eff * TW_eff; /* mỗi PE đọc input riêng    */
}
