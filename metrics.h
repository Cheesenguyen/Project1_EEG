#ifndef METRICS_H
#define METRICS_H
#include <stdio.h>
#include "config.h"
void metrics_reset(Metrics *m);
void metrics_print_report(FILE *fp, int n_cases,
                           const Config *cfgs, const Metrics *ms);
#endif
