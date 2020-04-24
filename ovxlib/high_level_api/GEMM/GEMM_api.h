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
#ifndef __GEMM_API_H__
#define __GEMM_API_H__

#include <stdio.h>
#include <stdlib.h>

#include "vsi_nn_pub.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _{
    const float* b_;    /* Matrix B */
    uint32_t m_;        /* Number of row for matrix A and C */
    uint32_t n_;        /* Number of col for matrix B and C */
    uint32_t k_;        /* Number of col for matrix A and Number of row for matrix B */

    vsi_nn_context_t ovx_ctx_;
    vsi_nn_graph_t* ovx_graph_;
    vsi_nn_tensor_id_t mat_a_tensor_id_;
    vsi_nn_tensor_id_t mat_c_tensor_id_;
    vsi_nn_tensor_id_t output_tensor_id_;
} gemm_t;

typedef gemm_t* gemm_handle_t;

gemm_handle_t GEMM_load
    (
    const float* b,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k
    );

vsi_bool GEMM_run(
    gemm_handle_t gemm,
    const float* a,
    float* c
    );

void GEMM_unload(gemm_handle_t* gemm_handle_addr);


#ifdef __cplusplus
}
#endif

#endif
