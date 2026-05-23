#ifndef METRICS_H
#define METRICS_H
#include <stdio.h>
#include "config.h"

void metrics_reset(Metrics *m);

/* Terminal: liệt kê từng case theo dạng key: value */
void metrics_print_terminal(int n_cases, const Config *cfgs, const Metrics *ms);

/* File: xuất CSV với header row — unroll_mode là chuỗi "TK" hoặc "THW" */
void metrics_write_csv(FILE *fp, int n_cases, const Config *cfgs, const Metrics *ms,
                       const char *unroll_mode);

#endif