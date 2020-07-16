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
#include "GEMM_api.h"
#include <time.h>

#define GEMM_DEBUG_DUMP FALSE
#define GEMM_PERF_DBG FALSE
#define WORKAROUND_DRIVER_FC_ISSUE TRUE

#define GEMM_NEW_VXNODE(_graph, _node, _type, _in, _out, _uid) do {\
        _node = vsi_nn_AddNode( _graph, _type, _in, _out, NULL );\
        _node->uid = (uint32_t)_uid; \
        if( NULL == _node ) {\
            goto error;\
        }\
    } while(0)

#define GEMM_NEW_VIRTUAL_TENSOR(_graph, _id, _attr, _dtype) do {\
        memset( _attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));\
        _attr.dim_num = VSI_NN_DIM_AUTO;\
        _attr.vtl = !GEMM_DEBUG_DUMP;\
        _attr.is_const = FALSE;\
        _attr.dtype.vx_type = _dtype;\
        _id = vsi_nn_AddTensor( _graph, VSI_NN_TENSOR_ID_AUTO,\
                & _attr, NULL );\
        if( VSI_NN_TENSOR_ID_NA == _id ) {\
            goto error;\
        }\
    } while(0)

// Set const tensor dims out of this macro.
#define GEMM_NEW_CONST_TENSOR(_graph, _id, _attr, _dtype, _data) do {\
        _attr.vtl = FALSE;\
        _attr.is_const = TRUE;\
        _attr.dtype.vx_type = _dtype;\
        _id = vsi_nn_AddTensor( _graph, VSI_NN_TENSOR_ID_AUTO,\
                & _attr, _data );\
        if( VSI_NN_TENSOR_ID_NA == _id ) {\
            goto error;\
        }\
    } while(0)

// Set generic tensor dims out of this macro.
#define GEMM_NEW_NORM_TENSOR(_graph, _id, _attr, _dtype) do {\
        _attr.vtl = FALSE;\
        _attr.is_const = FALSE;\
        _attr.dtype.vx_type = _dtype;\
        _id = vsi_nn_AddTensor( _graph, VSI_NN_TENSOR_ID_AUTO,\
                & _attr, NULL );\
        if( VSI_NN_TENSOR_ID_NA == _id ) {\
            goto error;\
        }\
    } while(0)


#if GEMM_PERF_DBG == TRUE
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
#endif

static vsi_bool gemm_load_graph(gemm_handle_t gemm);

static uint32_t gemm_fp32Tofp16
    (
    const float* fp32,
    uint32_t row_cnt,
    uint32_t col_cnt,
    uint8_t** fp16_buf,
    vsi_nn_dtype_t* type_desc,
    vsi_bool transpose
    )
{
    uint32_t i,j;

    *fp16_buf = (uint8_t*)calloc(col_cnt*row_cnt, sizeof(float)/2);
    if (NULL == *fp16_buf)
    {
        VSILOGE("GEMM: fatal error-> OOM");
        return 0U;
    }

    for (i = 0; i < row_cnt; ++i)
    {
        for (j = 0; j < col_cnt; ++j)
        {
            uint32_t fp16_idx = transpose ? 2*(j*row_cnt + i) : 2*(i*col_cnt + j);
            if (VSI_FAILURE == vsi_nn_Float32ToDtype(*(fp32 + i*col_cnt + j), (*fp16_buf + fp16_idx), type_desc))
            {
                VSILOGE("GEMM: FP32->FP16 failed");
                return 0U;
            }
        }
    }
    return ((i * j) - 1);
}

static void gemm_feed_in_inputs
    (
    gemm_handle_t gemm,
    vsi_nn_tensor_id_t tensor_idx,
    const float* data,
    uint32_t row,
    uint32_t col
    )
{
    vsi_nn_tensor_t* tensor;
    vsi_bool status;
    tensor = vsi_nn_GetTensor(gemm->ovx_graph_, tensor_idx);
    if (NULL != tensor)
    {
        uint8_t* temp;
        (void)gemm_fp32Tofp16((const float*) data, row, col, &temp, &tensor->attr.dtype, FALSE);
        status = vsi_nn_CopyDataToTensor(gemm->ovx_graph_, tensor, temp);
        free(temp);
        if (VSI_FAILURE == status)
        {
            VSILOGE("GEMM Runtime error: copy Matrix A failed");
        }
    }
    else
    {
        VSILOGE("GEMM Runtime error: can not find tensor for matrix A");
    }
}

gemm_handle_t GEMM_load
    (
    const float* b,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k
    )
{
    gemm_handle_t gemm_handle = (gemm_handle_t)malloc(sizeof(gemm_t));
    if (NULL != gemm_handle)
    {
        gemm_handle->b_ = b;
        gemm_handle->m_ = m;
        gemm_handle->n_ = n;
        gemm_handle->k_ = k;
    }
    else
    {
        VSILOGE("GEMM: malloc error");
        gemm_handle = NULL;
    }

    gemm_load_graph(gemm_handle);

    // assert(gemm_handle != NULL);
    return gemm_handle;
}

vsi_bool GEMM_run
    (
    gemm_handle_t gemm,
    const float* a,
    float* c
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_tensor_t* tensor = NULL;

    if (NULL == gemm || NULL == gemm->ovx_graph_) return VSI_FAILURE;

    gemm_feed_in_inputs(gemm, gemm->mat_a_tensor_id_, a, gemm->m_, gemm->k_);
    gemm_feed_in_inputs(gemm, gemm->mat_c_tensor_id_, c, gemm->m_, gemm->n_);

    { //VSILOGD("GEMM run: execute graph");
        #if GEMM_PERF_DBG == TRUE
            vx_uint64 tmsStart, tmsEnd, msVal, usVal;
            tmsStart = get_perf_count();
        #endif

        status = vsi_nn_RunGraph(gemm->ovx_graph_);

        #if GEMM_PERF_DBG == TRUE
            tmsEnd = get_perf_count();
            msVal = (tmsEnd - tmsStart)/1000000;
            usVal = (tmsEnd - tmsStart)/1000;
            VSILOGD("Execute Graph: %ldms or %ldus\n", msVal, usVal);
        #endif

        if (VSI_FAILURE == status)
        {
            VSILOGE("GEMM run: execute graph failed");
            return VSI_FAILURE;
        }
    }

    {// restore graph output to accumulation buffer
        tensor = vsi_nn_GetTensor(gemm->ovx_graph_, gemm->output_tensor_id_);

        if (NULL != tensor)
        {
            float* output_fp32 = vsi_nn_ConvertTensorToFloat32Data(gemm->ovx_graph_, tensor);
            memcpy(c, output_fp32, (gemm->m_)*(gemm->n_)*sizeof(float));
            free(output_fp32);
        }
    }

    return VSI_SUCCESS;
}

void GEMM_unload(gemm_handle_t* gemm_handle)
{
    if (NULL != gemm_handle && NULL != *gemm_handle)
    {
        vsi_nn_ReleaseGraph(&(*gemm_handle)->ovx_graph_);
        vsi_nn_ReleaseContext(&(*gemm_handle)->ovx_ctx_);
        free(*gemm_handle);
        *gemm_handle = NULL;
    }
}

static vsi_bool gemm_load_graph(gemm_handle_t gemm)
{
    const uint32_t num_of_node = 2;
    #if WORKAROUND_DRIVER_FC_ISSUE == TRUE
    const uint32_t num_of_tensor = 6; //(num_of_norm_tensor + num_of_vir_tensor + num_of_const_tensor);
    #else
    const uint32_t num_of_tensor = 5; //(num_of_norm_tensor + num_of_vir_tensor + num_of_const_tensor);
    #endif

    {//create ovxlib graph container
        gemm->ovx_ctx_ = vsi_nn_CreateContext();
        gemm->ovx_graph_ = vsi_nn_CreateGraph(gemm->ovx_ctx_, num_of_tensor, num_of_node);

        if (NULL == gemm->ovx_graph_)
        {
            VSILOGE("GEMM: create graph failed");
            vsi_nn_ReleaseContext(&gemm->ovx_ctx_);
            goto error;
        }

        vsi_nn_SetGraphInputs ( gemm->ovx_graph_, NULL, 2 );
        vsi_nn_SetGraphOutputs( gemm->ovx_graph_, NULL, 1 );
    }

    {// Build graph: Add tensor/node to graph
        vsi_nn_tensor_attr_t attr;
        vsi_nn_tensor_id_t mat_a_input_tensor;
        vsi_nn_tensor_id_t mat_c_input_tensor;
        #if WORKAROUND_DRIVER_FC_ISSUE == TRUE
        vsi_nn_tensor_id_t fc_fake_bias;
        #endif
        vsi_nn_tensor_id_t out_tensor;

        vsi_nn_tensor_id_t fc_weight_tensor;
        vsi_nn_tensor_id_t fc_add_vir_tensor;

        vsi_nn_node_t* fc_node;
        vsi_nn_node_t* add_node;

        GEMM_NEW_VXNODE(gemm->ovx_graph_, fc_node, VSI_NN_OP_FCL, 3, 1, -1);
        fc_node->nn_param.fcl.weights = gemm->n_;

        GEMM_NEW_VXNODE(gemm->ovx_graph_, add_node, VSI_NN_OP_ADD, 2, 1, -2 );

        // --> create Tensor
        attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;

        /* Matrix A */
        attr.size[0] = gemm->k_;
        attr.size[1] = gemm->m_;
        attr.dim_num = 2;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        GEMM_NEW_NORM_TENSOR(gemm->ovx_graph_, mat_a_input_tensor, attr, VSI_NN_TYPE_FLOAT16);

        /* Matrix C as input tensor */
        attr.size[0] = gemm->n_;
        attr.size[1] = gemm->m_;
        attr.dim_num = 2;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        GEMM_NEW_NORM_TENSOR(gemm->ovx_graph_, mat_c_input_tensor, attr, VSI_NN_TYPE_FLOAT16);

        /* Output tensor */
        attr.size[0] = gemm->n_;
        attr.size[1] = gemm->m_;
        attr.dim_num = 2;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        GEMM_NEW_NORM_TENSOR(gemm->ovx_graph_, out_tensor, attr, VSI_NN_TYPE_FLOAT16);

        /* set Matrix as weight*/
        attr.size[0] = gemm->k_;
        attr.size[1] = gemm->n_;
        attr.dim_num = 2;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
        {
             uint8_t* fp16_buf;
             uint32_t fp16_cnt = gemm_fp32Tofp16(gemm->b_, (gemm->k_), (gemm->n_), &fp16_buf, &attr.dtype, TRUE);
             GEMM_NEW_CONST_TENSOR(gemm->ovx_graph_, fc_weight_tensor, attr, VSI_NN_TYPE_FLOAT16, fp16_buf);
             free(fp16_buf);
        }

        #if WORKAROUND_DRIVER_FC_ISSUE == TRUE
        attr.size[0] = gemm->n_;
        attr.dim_num = 1;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
        {
            float* bias_data = (float*)calloc(gemm->n_, sizeof(float));
            GEMM_NEW_CONST_TENSOR(gemm->ovx_graph_, fc_fake_bias, attr, VSI_NN_TYPE_FLOAT32, (uint8_t *)bias_data);
            free(bias_data);
        }
        #endif

        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        GEMM_NEW_VIRTUAL_TENSOR(gemm->ovx_graph_, fc_node->output.tensors[0], attr, VSI_NN_TYPE_FLOAT16);

        /* setup network connectivities */
        fc_node->input.tensors[0] = mat_a_input_tensor;
        fc_node->input.tensors[1] = fc_weight_tensor;
        #if WORKAROUND_DRIVER_FC_ISSUE == TRUE
            fc_node->input.tensors[2] = fc_fake_bias;
        #endif

        add_node->input.tensors[0] = mat_c_input_tensor;
        add_node->input.tensors[1] = fc_node->output.tensors[0];
        add_node->output.tensors[0] = out_tensor;

        gemm->ovx_graph_->input.tensors[0] = mat_a_input_tensor;
        gemm->ovx_graph_->input.tensors[1] = mat_c_input_tensor;
        gemm->ovx_graph_->output.tensors[0] = out_tensor;

        gemm->mat_a_tensor_id_ = mat_a_input_tensor;
        gemm->mat_c_tensor_id_ = mat_c_input_tensor;
        gemm->output_tensor_id_ = out_tensor;

        if ( VSI_FAILURE == vsi_nn_SetupGraph(gemm->ovx_graph_, FALSE) )
        {
            VSILOGE("GEMM_ setup graph failed");
            goto error;
        }

        if (VSI_FAILURE == vsi_nn_VerifyGraph(gemm->ovx_graph_)) {
            VSILOGE("GEMM_ verify graph failed");
            goto error;
        }
    }

    return VSI_SUCCESS;

error:
    VSILOGE("GEMM Fatal error while loading graph");
    return VSI_FAILURE;
}
