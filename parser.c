// READ FILE
#include "parser.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static char *trim(char *s){
    while(isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s) - 1;
    while(e > s && isspace((unsigned char)*e)) *e-- = '\0';
    return s;
}

static int config_valid(const Config *c){
    return c->H > 0 && c->W > 0 && c->C > 0 &&
           c->K > 0 && c->R > 0 && c->S > 0 &&
           c->TH > 0 && c->TW > 0 && c->TC > 0 && c->TK > 0 &&
           c->stride > 0;
}

static void config_init(Config *c, int idx){
    memset(c, 0, sizeof(*c));
    c->stride  = 1;
    c->padding = 0;
    c->strategy = OS;
    snprintf(c->label, sizeof(c->label), "Case %d", idx + 1);
}

int parse_config(const char *filename, Config *cases)
{
    FILE *fp = fopen(filename, "r");
    if(!fp){ fprintf(stderr, "ERROR Cannot open config file: %s\n", filename); return 0; }

    int n = 0, in_case = 0;
    Config cur;
    config_init(&cur, n);

    char line[256];
    while(fgets(line, sizeof(line), fp)){
        char *p = trim(line);
        if(p[0] == '#' || p[0] == '\0'){
            if(p[0] == '\0' && in_case){
                if(cur.P > 0) cur.H = out_dim(cur.P, cur.R, cur.padding, cur.stride);
                if(cur.Q > 0) cur.W = out_dim(cur.Q, cur.S, cur.padding, cur.stride);
                if(config_valid(&cur)){ if(n < MAX_CASES) cases[n++] = cur; }
                config_init(&cur, n);
                in_case = 0;
            }
            continue;
        }
        char *eq = strchr(p, '=');
        if(!eq) continue;
        *eq = '\0';
        char *key = trim(p);
        char *val = trim(eq + 1);
        in_case = 1;

        if      (!strcmp(key,"P"))        cur.P        = atoi(val);
        else if (!strcmp(key,"Q"))        cur.Q        = atoi(val);
        else if (!strcmp(key,"H"))        cur.H        = atoi(val);
        else if (!strcmp(key,"W"))        cur.W        = atoi(val);
        else if (!strcmp(key,"C"))        cur.C        = atoi(val);
        else if (!strcmp(key,"K"))        cur.K        = atoi(val);
        else if (!strcmp(key,"R"))        cur.R        = atoi(val);
        else if (!strcmp(key,"S"))        cur.S        = atoi(val);
        else if (!strcmp(key,"TH"))       cur.TH       = atoi(val);
        else if (!strcmp(key,"TW"))       cur.TW       = atoi(val);
        else if (!strcmp(key,"TC"))       cur.TC       = atoi(val);
        else if (!strcmp(key,"TK"))       cur.TK       = atoi(val);
        else if (!strcmp(key,"stride"))   cur.stride   = atoi(val);
        else if (!strcmp(key,"padding"))  cur.padding  = atoi(val);
        else if (!strcmp(key,"strategy")) cur.strategy = parse_strategy(val);
        else if (!strcmp(key,"label"))    strncpy(cur.label, val, sizeof(cur.label)-1);
        else fprintf(stderr,"WARN Unknown key '%s' — ignored.\n", key);
    }

    if(in_case){
        if(cur.P > 0) cur.H = out_dim(cur.P, cur.R, cur.padding, cur.stride);
        if(cur.Q > 0) cur.W = out_dim(cur.Q, cur.S, cur.padding, cur.stride);
        if(config_valid(&cur)){ if(n < MAX_CASES) cases[n++] = cur; }
    }

    fclose(fp);
    if(n == 0) fprintf(stderr,"WARN No valid cases found in '%s'.\n", filename);
    return n;
}