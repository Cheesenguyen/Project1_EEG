#ifndef UTILS_H
#define UTILS_H

#include <string.h>
#include "config.h"

#define MIN(a,b)       ((a)<(b)?(a):(b))
#define MAX(a,b)       ((a)>(b)?(a):(b))
#define CEIL_DIV(a,b)  (((a)+(b)-1)/(b))        // làm tròn lên -> tính số lượng vòng lặp cần thiết

// tính kích thước đầu ra
static inline int out_dim(int in, int filt, int pad, int str){
    return (in + 2*pad - filt) / str + 1;
}

// chuyển ký tự trong file txt 
static inline Strategy parse_strategy(const char *s){
    if (strncmp(s,"WS",2)==0) return WS;
    if (strncmp(s,"IS",2)==0) return IS;
    return OS;
}

static inline const char *strat_str(Strategy s){
    switch(s){ case WS: return "WS"; case IS: return "IS"; default: return "OS"; }
}

#endif /* UTILS_H */
