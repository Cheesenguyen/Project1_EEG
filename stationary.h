#ifndef STATIONARY_H
#define STATIONARY_H
#include "config.h"
#include "tiling.h"

void ps_buf_init(float *ps_buf, const TileDesc *td);
void ps_buf_load(float *ps_buf, const float *ps_store, const TileDesc *td);
void ps_buf_save(const float *ps_buf, float *ps_store, const TileDesc *td);

#endif