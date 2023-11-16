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
#include "lcov/vsi_nn_test_tensor_op.h"
#define BUF_SIZE (64)

static vsi_status vsi_nn_test_Concat(void)
{
    VSILOGI("vsi_nn_test_Tensor_op");
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
    vsi_nn_tensor_t** tensors;
    tensors = (vsi_nn_tensor_t **)malloc(sizeof(vsi_nn_tensor_t *) * 2);
    vsi_size_t i;
    for(i = 0; i < 2; i++){
        tensors[i] = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    }

    vsi_nn_tensor_attr_t *attr0;
    attr0 = (vsi_nn_tensor_attr_t *)malloc(sizeof(vsi_nn_tensor_attr_t));
    attr0->dtype.vx_type = VSI_NN_TYPE_UINT8;
    attr0->size[0] = 1;
    attr0->dim_num = 1;

    uint8_t *data0;
    uint32_t sz = vsi_nn_TypeGetBytes(attr0->dtype.vx_type);
    data0 = (uint8_t *)malloc(sizeof( uint8_t) * sz);
    memset(data0, 0, sizeof( uint8_t ) * sz);
    vsi_nn_tensor_id_t id;
    id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, attr0, data0 );
    tensors[0] = vsi_nn_GetTensor(graph, id);

    uint8_t *data1;
    data1 = (uint8_t *)malloc( sizeof( uint8_t ) * sz);
    memset(data1, 1, sizeof( uint8_t ) * sz);
    id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, attr0, data1 );
    tensors[1] = vsi_nn_GetTensor(graph, id);

    uint32_t axis = 0;
    uint32_t tensor_num = 2;
    vsi_nn_tensor_t *result;

    VSILOGI("vsi_nn_test_Concat");
    result = vsi_nn_Concat(graph, tensors, tensor_num, axis);
    if(result == NULL) return VSI_FAILURE;
    uint8_t *tmp;
    uint8_t golden[2] = {0, 1};
    tmp = (uint8_t *)malloc(sizeof(uint8_t) * 2);
    tmp = (uint8_t*)vsi_nn_ConvertTensorToData( graph, result );
    for(i = 0; i < 2; i++)
    {
        if(tmp[i] != golden[i]) return VSI_FAILURE;
    }
    free(tmp);
    tmp = NULL;
    vsi_nn_ReleaseTensor(&result);

    VSILOGI("vsi_nn_test_Concat");
    tensor_num = 1;
    result = vsi_nn_Concat(graph, tensors, tensor_num, axis);
    if(result != NULL) return VSI_FAILURE;
    vsi_nn_ReleaseTensor(&result);

    VSILOGI("vsi_nn_test_Concat");
    tensor_num = 2;
    vsi_nn_tensor_attr_t *attr1;
    attr1 = (vsi_nn_tensor_attr_t *)malloc(sizeof(vsi_nn_tensor_attr_t));
    attr1->dtype.vx_type = VSI_NN_TYPE_UINT8;
    attr1->size[0] = 1;
    attr1->size[1] = 1;
    attr1->dim_num = 2;
    uint8_t *data2;
    data2 = (uint8_t *)malloc( sizeof( uint8_t ) * sz * 2);
    memset(data2, 1, sizeof( uint8_t ) * sz * 2);
    tensors[1] = NULL;
    id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, attr1, data2 );
    tensors[1] = vsi_nn_GetTensor(graph, id);
    result = vsi_nn_Concat(graph, tensors, tensor_num, axis);
    if(result != NULL) return VSI_FAILURE;
    vsi_nn_ReleaseTensor(&result);

    VSILOGI("vsi_nn_test_Concat");
    attr1->size[0] = 2;
    attr1->size[1] = 0;
    attr1->dim_num = 1;
    axis = 1;
    tensors[1] = NULL;
    id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, attr1, data2 );
    tensors[1] = vsi_nn_GetTensor(graph, id);
    result = vsi_nn_Concat(graph, tensors, tensor_num, axis);
    if(result != NULL) return VSI_FAILURE;
    vsi_nn_ReleaseTensor(&result);

    VSILOGI("vsi_nn_test_ConvertTensorDtype");
    vsi_nn_tensor_attr_t *attr2;
    attr2 = (vsi_nn_tensor_attr_t *)malloc(sizeof(vsi_nn_tensor_attr_t));
    attr2->dtype.vx_type = VSI_NN_TYPE_UINT16;
    vsi_nn_ConvertTensorDtype(graph, tensors[0], &attr2->dtype);

    VSILOGI("vsi_nn_test_Concat");
    memset(attr0, 0, sizeof(vsi_nn_tensor_attr_t));
    tensors[0] = NULL;
    id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, attr0, data0 );
    tensors[0] = vsi_nn_GetTensor(graph, id);
    result = vsi_nn_Concat(graph, tensors, tensor_num, axis);
    if(result != NULL) return VSI_FAILURE;
    vsi_nn_ReleaseTensor(&result);

    free(attr0);
    attr0 = NULL;
    free(attr1);
    attr1 = NULL;
    free(attr2);
    attr2 = NULL;
    free(result);
    result = NULL;

    free(data0);
    data0 = NULL;
    free(data1);
    data1 = NULL;
    free(data2);
    data2 = NULL;
    if(NULL != graph)
    {
        vsi_nn_ReleaseGraph(&graph);
    }
    if(NULL != ctx)
    {
        vsi_nn_ReleaseContext(&ctx);
    }
    return VSI_SUCCESS;
}

static vsi_status vsi_nn_test_TensorAdd(void)
{
    VSILOGI("vsi_nn_test_TensorAdd");
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
    vsi_nn_tensor_t** tensors_add;
    tensors_add = (vsi_nn_tensor_t **)malloc(sizeof(vsi_nn_tensor_t *) * 2);
    vsi_size_t i;
    for(i = 0; i < 2; i++){
        tensors_add[i] = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    }
    vsi_nn_tensor_attr_t *attr0;
    attr0 = (vsi_nn_tensor_attr_t *)malloc(sizeof(vsi_nn_tensor_attr_t));
    attr0->dtype.vx_type = VSI_NN_TYPE_UINT8;
    attr0->dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr0->size[0] = 1;
    attr0->dim_num = 1;
    uint8_t *data0;
    uint32_t sz = vsi_nn_TypeGetBytes(attr0->dtype.vx_type);
    data0 = (uint8_t *)malloc(sizeof( uint8_t) * sz);
    memset(data0, 0, sizeof( uint8_t ) * sz);
    vsi_nn_tensor_id_t id;
    id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, attr0, data0 );
    tensors_add[0] = vsi_nn_GetTensor(graph, id);
    uint8_t data1[1] = {2};
    id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, attr0, data1 );
    tensors_add[1] = vsi_nn_GetTensor(graph, id);
    vsi_nn_tensor_attr_t output_attr;
    output_attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    output_attr.size[0] = 1;
    output_attr.dim_num = 1;
    output_attr.is_const = 0;
    output_attr.vtl = 0;
    output_attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    uint32_t tensor_num = 2;
    vsi_nn_tensor_t *result_add;
    VSILOGI("vsi_nn_test_TensorAdd");
    result_add = vsi_nn_TensorAdd(graph, tensors_add, tensor_num, output_attr);
    if(result_add == NULL) return VSI_FAILURE;
    uint32_t sz_out = vsi_nn_TypeGetBytes(output_attr.dtype.vx_type);
    uint8_t *ret;
    ret = (uint8_t *)malloc(sizeof(uint8_t) * sz_out);
    ret = (uint8_t *)vsi_nn_ConvertTensorToData( graph, result_add );
    uint8_t golden[1] = {2};
    if(ret[0] != golden[0]) return VSI_FAILURE;
    free(ret);
    ret = NULL;
    vsi_nn_ReleaseTensor(&result_add);

    VSILOGI("vsi_nn_test_TensorAdd");
    tensor_num = 1;
    result_add = vsi_nn_TensorAdd(graph, tensors_add, tensor_num, output_attr);
    if(result_add != NULL) return VSI_FAILURE;
    vsi_nn_ReleaseTensor(&result_add);

    VSILOGI("vsi_nn_test_TensorAdd");
    tensor_num = 2;
    vsi_nn_tensor_attr_t *attr1;
    attr1 = (vsi_nn_tensor_attr_t *)malloc(sizeof(vsi_nn_tensor_attr_t));
    attr1->dtype.vx_type = VSI_NN_TYPE_UINT8;
    attr1->size[0] = 1;
    attr1->size[1] = 1;
    attr1->dim_num = 2;
    uint8_t *data2;
    data2 = (uint8_t *)malloc( sizeof( uint8_t ) * sz * 2);
    memset(data2, 1, sizeof( uint8_t ) * sz * 2);
    tensors_add[1] = NULL;
    id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, attr1, data2 );
    tensors_add[1] = vsi_nn_GetTensor(graph, id);
    result_add = vsi_nn_TensorAdd(graph, tensors_add, tensor_num, output_attr);
    if(result_add != NULL) return VSI_FAILURE;
    vsi_nn_ReleaseTensor(&result_add);

    VSILOGI("vsi_nn_test_TensorAdd");
    attr1->size[0] = 2;
    attr1->size[1] = 0;
    attr1->dim_num = 1;
    tensors_add[1] = NULL;
    id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, attr1, data2 );
    tensors_add[1] = vsi_nn_GetTensor(graph, id);
    result_add = vsi_nn_TensorAdd(graph, tensors_add, tensor_num, output_attr);
    if(result_add != NULL) return VSI_FAILURE;
    vsi_nn_ReleaseTensor(&result_add);

    VSILOGI("vsi_nn_test_TensorAdd");
    memset(attr0, 0, sizeof(vsi_nn_tensor_attr_t));
    tensors_add[0] = NULL;
    id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, attr0, data0 );
    tensors_add[0] = vsi_nn_GetTensor(graph, id);
    result_add = vsi_nn_TensorAdd(graph, tensors_add, tensor_num, output_attr);
    if(result_add != NULL) return VSI_FAILURE;
    vsi_nn_ReleaseTensor(&result_add);

    free(attr0);
    attr0 = NULL;
    free(attr1);
    attr1 = NULL;
    free(result_add);
    result_add = NULL;

    free(data0);
    data0 = NULL;
    free(data2);
    data2 = NULL;
    if(NULL != graph)
    {
        vsi_nn_ReleaseGraph(&graph);
    }
    if(NULL != ctx)
    {
        vsi_nn_ReleaseContext(&ctx);
    }
    return VSI_SUCCESS;
}

static vsi_status vsi_nn_test_shape_get_size(void)
{
    VSILOGI("vsi_nn_test_shape_get_size");
    const vsi_size_t shape[1] = {0};
    vsi_size_t rank = 1;
    vsi_size_t result;
    result = vsi_nn_shape_get_size(shape, rank);
    if(0 == result)
    {
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

static vsi_status vsi_nn_test_TypeGetRange(void)
{
    VSILOGI("vsi_nn_test_TypeGetRange");
    vsi_nn_type_e type = VSI_NN_TYPE_INT32;
    double max = 64.0;
    double min = 0.0;
    double *max_range = &max;
    double *min_range = &min;
    vsi_nn_TypeGetRange(type, max_range, min_range);
    return VSI_SUCCESS;
}

vsi_status vsi_nn_test_tensor_op( void )
{
    vsi_status status = VSI_FAILURE;

    status = vsi_nn_test_TensorAdd();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_Concat();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_shape_get_size();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_TypeGetRange();
    TEST_CHECK_STATUS(status, final);

final:
    return status;
}