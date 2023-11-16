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
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "vsi_nn_test.h"
#include "vsi_nn_pub.h"
#include "vsi_nn_tensor_util_prv.h"
#include "vsi_nn_kernel_prv.h"
#include "lcov/vsi_nn_test_util.h"
#define BUF_SIZE (64)

static vsi_status vsi_nn_test_strncat( void )
{
    VSILOGI("vsi_nn_test_strncat");
    char dst[BUF_SIZE] = "test ";
    char golden[BUF_SIZE] = "test strncat";
    vsi_nn_strncat(dst, "strncat", BUF_SIZE - 1);
    if( 0 == strncmp(dst, golden, sizeof(golden)) )
    {
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

static vsi_status vsi_nn_test_get_vx_pad_mode(void)
{
    VSILOGI("vsi_nn_test_get_vx_pad_mode");
    int32_t mode = -1;
    int32_t pad_mode;
    pad_mode = vsi_nn_get_vx_pad_mode(mode);
    if (0 == pad_mode)
    {
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

static vsi_status vsi_nn_test_compute_filter_shape(void)
{
    VSILOGI("vsi_nn_test_compute_filter_shape");
    vsi_nn_pad_e padding_type = VSI_NN_PAD_VALID;
    vsi_size_t image_size = 3;
    vsi_size_t ksize = 2;
    uint32_t stride = 1;
    uint32_t dilation_rate = 1;
    vsi_size_t result;
    result = vsi_nn_compute_filter_shape
    (
        padding_type,
        image_size,
        ksize,
        stride,
        dilation_rate
    );
    if(2 == result){
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

uint32_t vsi_nn_partition_cmp(void* data, int32_t left, int32_t right)
{
    return left <= right || data == NULL;
}

static vsi_status vsi_nn_test_partition(void)
{
    VSILOGI("vsi_nn_test_partition");
    int32_t left = 0;
    int32_t right = 3;
    vsi_bool is_recursion = 1;
    uint32_t indices[4] = {1, 3, 2, 4};
    void* data = indices;
    int32_t result;
    result = vsi_nn_partition(data, left, right, vsi_nn_partition_cmp, is_recursion, indices);
    if(0 == result)
    {
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

static vsi_status vsi_nn_test_IsEVISFeatureAvaiable(void)
{
    VSILOGI("vsi_nn_test_IsEVISFeatureAvaiable");
    struct _vsi_nn_context_t temp;
    temp.config.evis.ver = 0;
    vsi_nn_context_t context;
    context = &temp;
    vsi_bool result;
    result = vsi_nn_IsEVISFeatureAvaiable(context);
    if(0 == result){
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

static vsi_status vsi_nn_test_print_size_array(void)
{
    VSILOGI("vsi_nn_test_print_size_array");
    vsi_size_t array[1] = {1};
    size_t size = 1;
    vsi_nn_print_size_array(array, size);
    return VSI_SUCCESS;
}

static vsi_status vsi_nn_test_activation(void)
{
    VSILOGI("vsi_nn_test_activation");
    float value = 8.f;
    vsi_nn_activation_e activation = VSI_NN_ACT_RELU6;
    float result;
    result = vsi_nn_activation(value, activation);
    if (result != 6.0) return VSI_FAILURE;

    activation =VSI_NN_ACT_HARD_SIGMOID;
    result = vsi_nn_activation(value, activation);
    if(1.f == result)
    {
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

static vsi_status vsi_nn_test_is_broadcast_operaton(void)
{
    VSILOGI("vsi_nn_test_is_broadcast_operaton");
    vsi_nn_tensor_t **inputs;
    inputs = (vsi_nn_tensor_t **)malloc(sizeof(vsi_nn_tensor_t *) * 1);
    if (inputs == NULL){
        VSILOGE("malloc for broadcast_operaton_inputs failed!");
    }
    inputs[0] = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    if(inputs[0] == NULL)
    {
        VSILOGE("malloc for broadcast_operaton_inputs_0 failed!");
    }
    inputs[0]->attr.dim_num = 1;
    inputs[0]->attr.size[0] = 1;
    size_t input_num = 1;
    vsi_nn_tensor_t *output;
    output = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    if(output == NULL)
    {
        VSILOGE("malloc for broadcast_operaton_output failed!");
    }
    output->attr.dim_num = 1;
    output->attr.size[0] = 1;

    vsi_bool result;
    result = vsi_nn_is_broadcast_operaton(inputs, input_num, output);

    free(inputs[0]);
    inputs[0] = NULL;
    free(inputs);
    inputs = NULL;
    free(output);
    output = NULL;
    if(0 == result){
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

static vsi_status vsi_nn_test_is_broadcast_axes_operaton(void)
{
    VSILOGI("vsi_nn_test_is_broadcast_axes_operaton");
    vsi_nn_tensor_t **inputs;
    inputs = (vsi_nn_tensor_t **)malloc(sizeof(vsi_nn_tensor_t *) * 2);
    if (inputs == NULL){
        VSILOGE("malloc for broadcast_axes_operaton_inputs failed!");
    }
    vsi_size_t i;
    for (i = 0; i < 2; i++)
    {
        inputs[i] = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
        if(inputs[i] == NULL)
        {
            VSILOGE("malloc for broadcast_operaton_inputs_i failed!");
        }
        inputs[i]->attr.dim_num = 1;
        inputs[i]->attr.size[0] = 1;
    }
    size_t input_num = 2;
    vsi_nn_tensor_t *output;
    output = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    output->attr.dim_num = 1;
    output->attr.size[0] = 2;
    int32_t axis_num = 1;
    int32_t axis[1] = {0};

    vsi_bool result;
    result = vsi_nn_is_broadcast_axes_operaton(inputs, input_num, output, axis, axis_num);
    if(result != 0) return VSI_FAILURE;

    axis[0] = 1;
    result = vsi_nn_is_broadcast_axes_operaton(inputs, input_num, output, axis, axis_num);

    for(i = 0; i < 2; i++)
    {
        free(inputs[i]);
        inputs[i] = NULL;
    }
    free(inputs);
    inputs = NULL;
    free(output);
    output = NULL;
    if(1 == result){
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

static vsi_status vsi_nn_test_get_tensor_clamp_min_max(void)
{
    VSILOGI("vsi_nn_test_get_tensor_clamp_min_max");
    vsi_nn_tensor_t * input;
    input = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    input->attr.dtype.qnt_type = VSI_NN_QNT_TYPE_SYMMETRIC_FLOAT8;
    input->attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    float clampMin = 0;
    float clampMax = 255;

    vsi_nn_get_tensor_clamp_min_max(input, &clampMin, &clampMax);
    input->attr.dtype.vx_type = VSI_NN_TYPE_INT8;
    vsi_nn_get_tensor_clamp_min_max(input, &clampMin, &clampMax);
    input->attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC;
    vsi_nn_get_tensor_clamp_min_max(input, &clampMin, &clampMax);
    input->attr.dtype.vx_type = VSI_NN_TYPE_INT16;
    vsi_nn_get_tensor_clamp_min_max(input, &clampMin, &clampMax);
    input->attr.dtype.vx_type = VSI_NN_TYPE_UINT16;
    vsi_nn_get_tensor_clamp_min_max(input, &clampMin, &clampMax);
    input->attr.dtype.vx_type = VSI_NN_TYPE_FLOAT8_E4M3;
    vsi_nn_get_tensor_clamp_min_max(input, &clampMin, &clampMax);
    input->attr.dtype.vx_type = VSI_NN_TYPE_FLOAT8_E5M2;
    vsi_nn_get_tensor_clamp_min_max(input, &clampMin, &clampMax);
    input->attr.dtype.vx_type = VSI_NN_TYPE_FLOAT8_E5M2;
    vsi_nn_get_tensor_clamp_min_max(input, &clampMin, &clampMax);
    input->attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    vsi_nn_get_tensor_clamp_min_max(input, &clampMin, &clampMax);

    free(input);
    input = NULL;
    return VSI_SUCCESS;
}

static vsi_status vsi_nn_test_is_stream_process_supported_types(void)
{
    VSILOGI("vsi_nn_test_is_stream_process_supported_types");
    vsi_nn_context_t ctx;
    ctx = vsi_nn_CreateContext();
    if(!ctx)
    {
        VSILOGE("Create ovxlib context fail");
        assert(FALSE);
    }
    vsi_nn_graph_t *graph;
    graph = vsi_nn_CreateGraph(ctx, 0, 0);
    if(!graph)
    {
        VSILOGE("Create ovxlib graph fail");
        assert(FALSE);
    }
    graph->ctx->config.support_stream_processor = 1;
    graph->ctx->config.sp_exec_count = 0;
    vsi_nn_tensor_t** inputs;
    inputs = (vsi_nn_tensor_t **)malloc(sizeof(vsi_nn_tensor_t *));
    inputs[0] = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    inputs[0]->attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    size_t input_num = 1;

    vsi_bool result;
    result = vsi_nn_is_stream_process_supported_types(graph, inputs, input_num);
    if(result != 0) return VSI_FAILURE;

    graph->ctx->config.sp_exec_count = 1;
    result = vsi_nn_is_stream_process_supported_types(graph, inputs, input_num);

    if(NULL != graph)
    {
        vsi_nn_ReleaseGraph(&graph);
    }
    if(NULL != ctx)
    {
        vsi_nn_ReleaseContext(&ctx);
    }
    free(inputs[0]);
    inputs[0] = NULL;
    free(inputs);
    inputs = NULL;
    if(result == 1)
    {
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

static vsi_status vsi_nn_test_is_sp_supported_broadcast(void)
{
    VSILOGI("vsi_nn_test_is_sp_supported_broadcast");
    vsi_nn_context_t ctx;
    ctx = vsi_nn_CreateContext();
    if(!ctx)
    {
        VSILOGE("Create ovxlib context fail");
        assert(FALSE);
    }
    vsi_nn_graph_t *graph;
    graph = vsi_nn_CreateGraph(ctx, 0, 0);
    if(!graph)
    {
        VSILOGE("Create ovxlib graph fail");
        assert(FALSE);
    }
    graph->ctx->config.support_stream_processor = 1;
    graph->ctx->config.sp_exec_count = 1;
    vsi_nn_tensor_t** inputs;
    inputs = (vsi_nn_tensor_t **)malloc(sizeof(vsi_nn_tensor_t *) * 2);
    vsi_size_t i;
    for(i = 0; i < 2; i++){
        inputs[i] = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    }
    inputs[0]->attr.size[0] = 1;
    inputs[0]->attr.size[1] = 2;
    inputs[0]->attr.size[2] = 2;
    inputs[0]->attr.dim_num = 3;
    inputs[1]->attr.size[0] = 1;
    inputs[1]->attr.size[1] = 2;
    inputs[1]->attr.dim_num = 2;
    size_t input_num = 2;
    vsi_nn_tensor_t *output;
    output = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    output->attr.dim_num = 3;

    vsi_bool result;
    result = vsi_nn_is_sp_supported_broadcast(graph, inputs, input_num, output);

    if(NULL != graph)
    {
        vsi_nn_ReleaseGraph(&graph);
    }
    if(NULL != ctx)
    {
        vsi_nn_ReleaseContext(&ctx);
    }
    for(i = 0; i < 2; i++)
    {
        free(inputs[i]);
        inputs[i] = NULL;
    }
    free(inputs);
    inputs = NULL;
    free(output);
    output = NULL;

    if(result == 1)
    {
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

vsi_status vsi_nn_test_util( void )
{
    vsi_status status = VSI_FAILURE;

    status = vsi_nn_test_strncat();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_get_vx_pad_mode();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_compute_filter_shape();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_partition();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_activation();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_IsEVISFeatureAvaiable();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_print_size_array();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_is_broadcast_operaton();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_is_broadcast_axes_operaton();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_get_tensor_clamp_min_max();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_is_stream_process_supported_types();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_is_sp_supported_broadcast();
    TEST_CHECK_STATUS(status, final);

final:
    return status;
}