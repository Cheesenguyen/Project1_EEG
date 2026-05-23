#include "stationary.h"
#include "utils.h"

/*
 * Stationary Strategies — Temporal Reuse Analysis
 * 
 * Mỗi chiến lược quyết định biến nào "đứng yên" trong PE accumulator
 * và biến nào phải được đẩy ra PS buffer (SRAM) giữa các cycle.
 *
 * Ký hiệu:
 *   out_tile = TH_eff × TW_eff × TK_eff  (số output neuron trong 1 tile)
 *   n_tiles  = n_th × n_tw × n_tc × n_tk (tổng số tile)
 *
 * OS — Output Stationary
 * 
 * Psum của mỗi output neuron (oh,ow,k) LUÔN nằm trong PE accumulator
 * xuyên suốt toàn bộ vòng lặp (r,s,c) BÊN TRONG một tile (th,tw,tc,tk).
 *
 * Nhưng khi C bị cắt thành nhiều tc-tile (n_tc > 1), output chưa hoàn chỉnh
 * sau tc-tile đầu tiên → Psum phải được spill ra PS buffer giữa các tc-tile.
 *
 * Spill xảy ra:
 *   - Cuối mỗi tc-tile (trừ tc-tile cuối cùng) → ghi Psum ra PS buffer
 *   - Đầu tc-tile tiếp theo → đọc Psum từ PS buffer để cộng tiếp
 *
 * Số lần spill / output neuron = (n_tc - 1)
 * Tổng output neurons = out_tile × n_th × n_tw × n_tk
 *
 * PS writes = PS reads = out_tile × n_th × n_tw × n_tk × (n_tc - 1)
 *
 * 
 * WS — Weight Stationary
 * 
 * Weight w[k,c,r,s] cố định trong PE. PE lần lượt xử lý các output
 * position (oh,ow). Với spatial unroll theo TK_eff:
 *
 * Loop order bên trong tile (WS-friendly):
 *   for c in TC_eff
 *     for r in R
 *       for s in S           ← weight (k,c,r,s) cố định trong PE
 *         for oh in TH_eff
 *           for ow in TW_eff
 *             [parallel k in TK_eff]  ← 1 cycle
 *
 * Mỗi cycle cho output neuron (oh,ow,k) chỉ nhận 1 đóng góp MAC
 * (từ một (c,r,s) cụ thể). Để hoàn thành output cần R×S×TC_eff cycle.
 * Vì weight thay đổi nhanh (inner loop), Psum của (oh,ow,k) phải được
 * lưu ra PS buffer sau mỗi (c,r,s) iteration để cộng dồn.
 *
 * Iteration count per tile = TC_eff × R × S
 * Mỗi iteration: out_tile Psum bị spill ra (write) và đọc lại (read)
 * Iteration đầu tiên: chỉ write (chưa có Psum cũ để đọc)
 *
 * PS writes = out_tile × (TC_eff × R × S)       per tile group
 * PS reads  = out_tile × (TC_eff × R × S - 1)   per tile group
 * (nhân với n_th × n_tw × n_tc × n_tk)
 *
 * 
 * IS — Input Stationary
 * 
 * Input in[ih,iw,c] cố định trong PE. PE quét tất cả weight dùng input đó.
 *
 * Loop order bên trong tile (IS-friendly):
 *   for oh in TH_eff
 *     for ow in TW_eff
 *       for c in TC_eff        ← input in[ih,iw,c] cố định trong PE
 *         for r in R
 *           for s in S
 *             [parallel k in TK_eff]  ← 1 cycle
 *
 * Với spatial unroll theo k: input broadcast miễn phí trong 1 cycle.
 * Psum của (oh,ow,k) tích lũy qua (c,r,s). Khi c thay đổi, PE chuyển
 * sang input mới → Psum dở dang của mỗi k phải spill ra PS buffer.
 *
 * Spill xảy ra sau mỗi c-step (trừ c-step cuối):
 *   Số c-steps per tile = TC_eff
 *   Psum spills / tile = out_tile × (TC_eff - 1)
 *
 * PS writes = PS reads = out_tile × (TC_eff - 1) × n_th × n_tw × n_tc × n_tk
 */

/*   helpers    */
static void get_dims(const Config *cfg,
                     int *OH, int *OW,
                     int *n_th, int *n_tw, int *n_tc, int *n_tk,
                     int *TH_e, int *TW_e, int *TC_e, int *TK_e)
{
    *OH   = out_dim(cfg->H, cfg->R, cfg->padding, cfg->stride);
    *OW   = out_dim(cfg->W, cfg->S, cfg->padding, cfg->stride);
    *n_th = CEIL_DIV(*OH, cfg->TH);
    *n_tw = CEIL_DIV(*OW, cfg->TW);
    *n_tc = CEIL_DIV(cfg->C, cfg->TC);
    *n_tk = CEIL_DIV(cfg->K, cfg->TK);
    *TH_e = MIN(cfg->TH, *OH);
    *TW_e = MIN(cfg->TW, *OW);
    *TC_e = MIN(cfg->TC, cfg->C);
    *TK_e = MIN(cfg->TK, cfg->K);
}

/*   OS    */
static void os_stat(const Config *cfg, Metrics *m)
{
    int OH,OW,n_th,n_tw,n_tc,n_tk,TH_e,TW_e,TC_e,TK_e;
    get_dims(cfg,&OH,&OW,&n_th,&n_tw,&n_tc,&n_tk,&TH_e,&TW_e,&TC_e,&TK_e);

    long out_tile  = (long)TH_e * TW_e * TK_e;
    long n_out_set = (long)n_th * n_tw * n_tk;   /* number of (th,tw,tk) combos */

    /* spill between tc-tiles only */
    long spills = out_tile * n_out_set * (n_tc - 1);
    m->ps_buffer_writes = spills;
    m->ps_buffer_reads  = spills;
}

/*   WS    */
static void ws_stat(const Config *cfg, Metrics *m)
{
    int OH,OW,n_th,n_tw,n_tc,n_tk,TH_e,TW_e,TC_e,TK_e;
    get_dims(cfg,&OH,&OW,&n_th,&n_tw,&n_tc,&n_tk,&TH_e,&TW_e,&TC_e,&TK_e);

    long n_tiles     = (long)n_th * n_tw * n_tc * n_tk;
    long out_tile    = (long)TH_e * TW_e * TK_e;
    long rsc_steps   = (long)TC_e * cfg->R * cfg->S;

    /* every (c,r,s) step spills all output psums; first step has no prior psum */
    m->ps_buffer_writes = out_tile * rsc_steps       * n_tiles;
    m->ps_buffer_reads  = out_tile * (rsc_steps - 1) * n_tiles;
}

/*   IS    */
static void is_stat(const Config *cfg, Metrics *m)
{
    int OH,OW,n_th,n_tw,n_tc,n_tk,TH_e,TW_e,TC_e,TK_e;
    get_dims(cfg,&OH,&OW,&n_th,&n_tw,&n_tc,&n_tk,&TH_e,&TW_e,&TC_e,&TK_e);

    long n_out_set = (long)n_th * n_tw * n_tk;   /* (th,tw,tk) combos — n_tc excluded */
    long out_tile  = (long)TH_e * TW_e * TK_e;

    /*
     * IS: spill xay ra moi khi c thay doi, ke ca qua ranh gioi tc-tile.
     *
     * Tong so c-step toan cuc = TC_e * n_tc  (= C)
     * So c-transition         = TC_e * n_tc - 1  (= C - 1)
     *
     * Code cu (sai): spills = out_tile * (TC_e-1) * n_tiles
     *   -> bo sot (n_tc-1) boundary transitions giua cac tc-tile
     *
     * Code moi (dung): dem tat ca C-1 transitions toan cuc,
     * ap dung cho moi (th,tw,tk) set -> nhan voi n_out_set.
     */
    long c_transitions  = (long)TC_e * n_tc - 1;   /* = C - 1 toan cuc */
    long spills         = out_tile * c_transitions;
    m->ps_buffer_writes = spills * n_out_set;
    m->ps_buffer_reads  = spills * n_out_set;
}

/* ── dispatcher ── */
void run_stationary(const Config *cfg, Metrics *m)
{
    switch (cfg->strategy) {
        case OS: os_stat(cfg, m); break;
        case WS: ws_stat(cfg, m); break;
        case IS: is_stat(cfg, m); break;
    }
}
