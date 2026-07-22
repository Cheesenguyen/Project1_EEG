"""
verify.py
=========
So sánh kết quả conv từ chương trình C với PyTorch ground truth.

Flow:
  1. Đọc configs.txt  -> lấy shape (P,Q,C,K,R,S,stride,padding) của từng label
  2. Tái tạo input/weight bằng lcg_rand (seed=42, giống hệt conv.c)
  3. Tính ground truth bằng PyTorch conv2d
  4. Đọc conv_stats.csv -> lấy min/max/mean/sum từng kênh do C tính
  5. So sánh từng chỉ số, in báo cáo

Cách dùng:
  pip install torch numpy
  python verify.py
"""

import sys, os, csv
import numpy as np
import torch
import torch.nn.functional as F

# CẤU HÌNH

CONFIGS_FILE   = "configs.txt"
STATS_FILE     = "conv_stats.csv"
TOLERANCE_MEAN = 1e-1   # ngưỡng sai số mean (float32 tích lũy lớn)
TOLERANCE_MINMAX = 5e-1 # ngưỡng sai số min/max


# TÁI TẠO lcg_rand — giống hệt conv.c

def lcg_rand_array(n_input, n_weight):
    """
    Tái tạo đúng dãy số lcg_rand trong conv.c:
        seed = 42u
        input [i] = lcg_rand(&seed)   (n_input  lần)
        weight[i] = lcg_rand(&seed)   (n_weight lần)

    Công thức:
        seed = seed * 1664525 + 1013904223  (mod 2^32, unsigned)
        value = (seed & 0xFFFF) / 65535.0 - 0.5
    """
    seed = np.uint32(42)
    total = n_input + n_weight
    values = np.zeros(total, dtype=np.float32)

    for i in range(total):
        seed = np.uint32(seed * np.uint32(1664525) + np.uint32(1013904223))
        values[i] = float(seed & np.uint32(0xFFFF)) / 65535.0 - 0.5

    inp = values[:n_input]
    wt  = values[n_input:]
    return inp, wt

# ĐỌC CONFIGS.TXT

def parse_configs(filepath):
    """
    Đọc configs.txt, trả về dict:
        { label_string : { P, Q, C, K, R, S, stride, padding } }
    """
    configs = {}
    current = {}

    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                # Dòng trống = kết thúc 1 block
                if "label" in current:
                    configs[current["label"]] = current.copy()
                    current = {}
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip(); val = val.strip()
            if key == "label":
                current["label"] = val
            elif key in ("P","Q","C","K","R","S","stride","padding"):
                current[key] = int(val)

    # Block cuối file (không có dòng trống sau)
    if "label" in current:
        configs[current["label"]] = current.copy()

    return configs

# ĐỌC CONV_STATS.CSV

def read_conv_stats(filepath):
    """
    Đọc conv_stats.csv, trả về dict:
        { label : [ {channel_k, min, max, mean, sum}, ... ] }
    """
    stats = {}
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip().strip('"')
            entry = {
                "channel_k" : int(row["channel_k"]),
                "min"        : float(row["min"]),
                "max"        : float(row["max"]),
                "mean"       : float(row["mean"]),
                "sum"        : float(row["sum"]),
            }
            stats.setdefault(label, []).append(entry)
    return stats


# GROUND TRUTH BẰNG PYTORCH

def pytorch_conv_stats(inp_flat, wt_flat, P, Q, C, K, R, S, stride, padding):
    """
    Tính conv2d bằng PyTorch, trả về list thống kê từng kênh:
        [ {channel_k, min, max, mean, sum}, ... ]

    Layout C: input [P*Q*C] lưu theo HWC, weight [K*R*S*C] lưu theo KRSC
    """
    # Reshape về đúng layout
    inp_hwc  = inp_flat.reshape(P, Q, C)                # HWC  (N=1)
    wt_krsc  = wt_flat.reshape(K, R, S, C)              # KRSC

    # Chuyển sang PyTorch layout: NCHW và KCHW
    inp_t = torch.tensor(inp_hwc,  dtype=torch.float32).permute(2,0,1).unsqueeze(0)  # 1CHW
    wt_t  = torch.tensor(wt_krsc,  dtype=torch.float32).permute(0,3,1,2)             # KCHW

    out_t = F.conv2d(inp_t, wt_t, bias=None, stride=stride, padding=padding)
    # out_t shape: [1, K, OH, OW]

    result = []
    for k in range(K):
        ch = out_t[0, k].detach().numpy()   # [OH, OW]
        result.append({
            "channel_k" : k,
            "min"        : float(ch.min()),
            "max"        : float(ch.max()),
            "mean"       : float(ch.mean()),
            "sum"        : float(ch.sum()),
        })
    return result

# SO SÁNH VÀ BÁO CÁO


def compare_label(label, c_stats, ref_stats):
    """
    So sánh từng kênh giữa C và PyTorch.
    Trả về (n_pass, n_total).
    """
    n_pass = 0; n_total = 0

    # Tạo dict để tra nhanh theo channel_k
    ref_dict = {r["channel_k"]: r for r in ref_stats}

    fail_channels = []

    for entry in c_stats:
        k   = entry["channel_k"]
        ref = ref_dict.get(k)
        if ref is None:
            continue

        err_mean   = abs(entry["mean"] - ref["mean"])
        err_min    = abs(entry["min"]  - ref["min"])
        err_max    = abs(entry["max"]  - ref["max"])

        ok = (err_mean <= TOLERANCE_MEAN and
              err_min  <= TOLERANCE_MINMAX and
              err_max  <= TOLERANCE_MINMAX)

        n_total += 1
        if ok:
            n_pass += 1
        else:
            fail_channels.append((k, err_mean, err_min, err_max, entry, ref))

    if fail_channels:
        print(f" {len(fail_channels)} kênh vượt ngưỡng:")
        for k, em, ei, ex, c, r in fail_channels[:3]:  # in tối đa 3 kênh
            print(f"      k={k:3d}  "
                  f"mean: C={c['mean']:+.4f} ref={r['mean']:+.4f} err={em:.4f} | "
                  f"min err={ei:.4f} | max err={ex:.4f}")
        if len(fail_channels) > 3:
            print(f"      ... và {len(fail_channels)-3} kênh khác")

    return n_pass, n_total


# MAIN


def main():
    # Kiểm tra file tồn tại
    for f in [CONFIGS_FILE, STATS_FILE]:
        if not os.path.exists(f):
            print(f" Không tìm thấy '{f}' — hãy chạy 'make run' trước.")
            sys.exit(1)

    configs    = parse_configs(CONFIGS_FILE)
    conv_stats = read_conv_stats(STATS_FILE)

    print("=" * 65)
    print("  CONV VERIFY  —  C stats vs PyTorch ground truth")
    print("=" * 65)

    grand_pass = 0; grand_total = 0

    # Lấy danh sách label từ conv_stats (giữ đúng thứ tự xuất hiện)
    seen = []; labels_ordered = []
    for label in conv_stats:
        if label not in seen:
            seen.append(label)
            labels_ordered.append(label)

    for label in labels_ordered:
        c_stats = conv_stats[label]

        # Tìm config khớp label (so sánh bằng startswith vì label có thể bị cắt)
        cfg = None
        for key, val in configs.items():
            if label.startswith(key) or key.startswith(label) or key == label:
                cfg = val
                break

        if cfg is None:
            print(f"\n  [{label}] — không tìm thấy config tương ứng, bỏ qua.")
            continue

        P=cfg["P"]; Q=cfg["Q"]; C=cfg["C"]; K=cfg["K"]
        R=cfg["R"]; S=cfg["S"]
        stride=cfg["stride"]; padding=cfg["padding"]

        print(f"\n{'─'*65}")
        print(f"  [{label}]")
        print(f"  Input: P={P} Q={Q} C={C}  "
              f"Filter: K={K} R={R} S={S}  "
              f"stride={stride} pad={padding}")

        # Tái tạo data giống conv.c
        inp_flat, wt_flat = lcg_rand_array(P*Q*C, K*R*S*C)

        # Ground truth
        ref_stats = pytorch_conv_stats(inp_flat, wt_flat,
                                       P, Q, C, K, R, S, stride, padding)

        # So sánh
        n_pass, n_total = compare_label(label, c_stats, ref_stats)
        grand_pass  += n_pass
        grand_total += n_total

        status = " PASS" if n_pass == n_total else " FAIL"
        print(f"  {status}  {n_pass}/{n_total} kênh đúng")

    # Tổng kết
    print(f"\n{'='*65}")
    print(f"  KẾT QUẢ: {grand_pass}/{grand_total} kênh PASS")
    if grand_pass == grand_total and grand_total > 0:
        print(" Tất cả đều đúng!")
    elif grand_total == 0:
        print(" Không có dữ liệu để kiểm tra.")
    else:
        print(f" {grand_total - grand_pass} kênh thất bại.")
    print("=" * 65)

    return 0 if grand_pass == grand_total else 1

if __name__ == "__main__":
    sys.exit(main())