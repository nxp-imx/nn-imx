/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <time.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#define _BASETSD_H

#include "vsi_nn_pub.h"

/*-------------------------------------------
        Macros and Variables
-------------------------------------------*/

/*-------------------------------------------
                  Functions
-------------------------------------------*/
#define BILLION                                 1000000000
static vx_uint64 get_perf_count()
{
#if defined(__linux__) || defined(__ANDROID__) || defined(__QNX__) || defined(__CYGWIN__)
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (vx_uint64)((vx_uint64)ts.tv_nsec + (vx_uint64)ts.tv_sec * BILLION);
#elif defined(_WIN32) || defined(UNDER_CE)
    LARGE_INTEGER ln;

    QueryPerformanceCounter(&ln);

    return (vx_uint64)ln.QuadPart;
#endif
}

#include "GEMM_api.h"

void print_matrix(float* mat, uint32_t row_cnt, uint32_t col_cnt)
{
    uint32_t i,j;
    printf("************************\n");
    for (i = 0; i < row_cnt; ++i)
    {
        for(j = 0; j < col_cnt; ++j)
        {
            printf("%.4f\t", *(mat + i * col_cnt + j));
        }
        printf("\n");
    }
    printf("************************\n");
}

void init_matrix(float* mat, uint32_t sz)
{
    for (uint32_t i = 0; i < sz; ++i) {
        *(mat) = 1.f * i;
        mat++;
    }
}

extern void gemm_transpose(float* fp32, uint32_t cnt_row, uint32_t cnt_col);
/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main
    (
    int argc,
    char **argv
    )
{
    if (1) {
        vx_uint64 tmsStart, tmsEnd, msVal, usVal;
        const uint32_t M = 10 ;
        const uint32_t K = 12*1024 ;
        const uint32_t N = 1024 ;

        float* matrix_a = (float*)malloc(sizeof(float)*M*K);
        float* matrix_b = (float*)malloc(sizeof(float)*K*N);
        float* matrix_c = (float*)calloc(sizeof(float), M*N);

        init_matrix(matrix_a, M*K);
        init_matrix(matrix_b, K*N);
        //init_matrix(matrix_c, M*N);

        // print_matrix(matrix_a, M, K);
        // print_matrix(matrix_b, K, N);
        // print_matrix(matrix_c, M, N);

        tmsStart = get_perf_count();
        gemm_handle_t gemm_2 = GEMM_load(matrix_b, M, N, K);
        tmsEnd = get_perf_count();
        msVal = (tmsEnd - tmsStart)/1000000;
        usVal = (tmsEnd - tmsStart)/1000;
        VSILOGI("GEMM_load: %ldms or %ldus\n", msVal, usVal);

        tmsStart = get_perf_count();

        if (VSI_FAILURE == GEMM_run(gemm_2, matrix_a, matrix_c))
        {
            // Don't take matrix_c as final result, internal error happened.
        }

        tmsEnd = get_perf_count();
        msVal = (tmsEnd - tmsStart)/1000000;
        usVal = (tmsEnd - tmsStart)/1000;
        VSILOGI("GEMM_run: %ldms or %ldus\n", msVal, usVal);
        //print_matrix(matrix_c, M, N);

        GEMM_unload(&gemm_2);

        VSILOGI("GEMM sample done");
    }

    return 0;
}