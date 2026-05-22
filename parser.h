#ifndef PARSER_H
#define PARSER_H
#include "config.h"

#define MAX_CASES 64

int parse_config(const char *filename, Config *cases);

#endif
