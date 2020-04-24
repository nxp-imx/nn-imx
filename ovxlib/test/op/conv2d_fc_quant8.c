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
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <VX/vx_khr_cnn.h>

#include "vsi_nn_pub.h"

#define _CHECK_PTR( ptr, lbl )      do {\
    if( NULL == ptr ) {\
        ret = 0; \
        printf("CHECK PTR %d", __LINE__);\
        goto lbl;\
    }\
} while(0)

#define _CHECK_STATUS( stat, lbl )  do {\
    if( VX_SUCCESS != stat ) {\
        printf("CHECK STATUS %d", __LINE__);\
        goto lbl;\
    }\
} while(0)

static int compare
    (
    uint8_t * a, uint8_t * b,
    size_t size, int32_t tolerance
    )
{
    uint32_t i;
    int ret = 1;
    for( i = 0; i < size; i ++ )
    {
        if(vsi_abs(a[i] - b[i]) > tolerance)
        {
            VSILOGE("Compare fail %d: abs(%u - %u) > %d ",
                    i, a[i], b[i], tolerance);
            ret &= 0;
        }
    }
    return ret;
}

static vsi_nn_context_t ctx = NULL;

static int run_test_fc(
        vsi_nn_tensor_attr_t * input_attr,
        vsi_nn_tensor_attr_t * kernel_attr,
        vsi_nn_tensor_attr_t * bias_attr,
        vsi_nn_tensor_attr_t * output_attr,
        uint8_t * input, uint8_t * kernel, int32_t * bias,
        uint8_t * golden )
{
    int ret = 1;
    vsi_nn_graph_t      * graph;
    vsi_nn_node_t       * node;
    vsi_nn_tensor_t     * tensor;
    uint8_t *out = NULL;
    vsi_status status;

    graph = vsi_nn_CreateGraph( ctx, 0, 0 );
    _CHECK_PTR( graph, final );

    vsi_nn_SetGraphInputs( graph, NULL, 1 );
    vsi_nn_SetGraphOutputs( graph, NULL, 1 );

    node = vsi_nn_AppendNode( graph, VSI_NN_OP_FCL, NULL );
    _CHECK_PTR( node, final );
    node->nn_param.fcl.weights = 1;
    node->vx_param.has_relu = vx_true_e;
    node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    node->vx_param.down_scale_size_rounding =
        VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    node->input.tensors[0] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, input_attr, input );
    node->input.tensors[1] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, kernel_attr, kernel );
    node->input.tensors[2] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, bias_attr, (uint8_t*)bias );

    node->output.tensors[0] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, output_attr, NULL );

    graph->input.tensors[0] = node->input.tensors[0];
    graph->output.tensors[0] = node->output.tensors[0];

    status = vsi_nn_SetupGraph( graph, TRUE );
    _CHECK_STATUS( status, final );
    status = vsi_nn_VerifyGraph( graph );
    _CHECK_STATUS( status, final );
    status = vsi_nn_RunGraph( graph );
    _CHECK_STATUS( status, final );

    tensor = vsi_nn_GetTensor( graph, node->output.tensors[0] );
    _CHECK_PTR( tensor, final );

    out = vsi_nn_ConvertTensorToData( graph, tensor );
    _CHECK_PTR( out, final );

    ret &= compare(golden, out, vsi_nn_GetElementNum( tensor ), 0);

final:

    if(out != NULL)
    {
        free(out);
        out = NULL;
    }
    vsi_nn_ReleaseGraph( &graph );
    //vsi_nn_ReleaseContext( &ctx );

    return ret;
}

static int run_test_conv2d(
        vsi_nn_tensor_attr_t * input_attr,
        vsi_nn_tensor_attr_t * kernel_attr,
        vsi_nn_tensor_attr_t * bias_attr,
        vsi_nn_tensor_attr_t * output_attr,
        uint8_t * input, uint8_t * kernel, int32_t * bias,
        uint8_t * golden )
{
    int ret = 1;
    vsi_nn_graph_t      * graph;
    vsi_nn_node_t       * node;
    vsi_nn_tensor_t     * tensor;
    uint8_t *out = NULL;
    vsi_status status;

    graph = vsi_nn_CreateGraph( ctx, 0, 0 );
    _CHECK_PTR( graph, final );

    vsi_nn_SetGraphInputs( graph, NULL, 1 );
    vsi_nn_SetGraphOutputs( graph, NULL, 1 );

    node = vsi_nn_AppendNode( graph, VSI_NN_OP_CONV2D, NULL );
    _CHECK_PTR( node, final );
    node->nn_param.conv2d.ksize[0] = 1;
    node->nn_param.conv2d.ksize[1] = 1;
    node->nn_param.conv2d.weights = 1;
    node->nn_param.conv2d.stride[0] = 1;
    node->nn_param.conv2d.stride[1] = 1;
    node->nn_param.conv2d.pad[0] = 0;
    node->nn_param.conv2d.pad[1] = 0;
    node->nn_param.conv2d.pad[2] = 0;
    node->nn_param.conv2d.pad[3] = 0;
    node->nn_param.conv2d.group = 1;
    node->nn_param.conv2d.dilation[0] = 1;
    node->nn_param.conv2d.dilation[1] = 1;

    node->input.tensors[0] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, input_attr, input );
    node->input.tensors[1] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, kernel_attr, kernel );
    node->input.tensors[2] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, bias_attr, (uint8_t*)bias );

    node->output.tensors[0] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, output_attr, NULL );

    graph->input.tensors[0] = node->input.tensors[0];
    graph->output.tensors[0] = node->output.tensors[0];

    status = vsi_nn_SetupGraph( graph, TRUE );
    _CHECK_STATUS( status, final );
    status = vsi_nn_VerifyGraph( graph );
    _CHECK_STATUS( status, final );
    status = vsi_nn_RunGraph( graph );
    _CHECK_STATUS( status, final );

    tensor = vsi_nn_GetTensor( graph, node->output.tensors[0] );
    _CHECK_PTR( tensor, final );

    out = vsi_nn_ConvertTensorToData( graph, tensor );
    _CHECK_PTR( out, final );

    ret &= compare(golden, out, vsi_nn_GetElementNum( tensor ), 0);

final:

    if(out != NULL)
    {
        free(out);
        out = NULL;
    }
    vsi_nn_ReleaseGraph( &graph );
    //vsi_nn_ReleaseContext( &ctx );

    return ret;
}

#define CASE_NUM    (3)
static int test_conv2d_fc()
{
    int ret = 1;
    float scale_kernel = 0.5f;
    float scale_input = 1.f;
    float scale_output = 1.f;
    uint8_t input[][10]  = {
        {129,129,129,129,129,129,129,129,129,129},
        {127,127,127,127,127,127,127,127,127,127}
        };
    uint8_t kernel[CASE_NUM][10] = {
        {0,1,2,3,4,5,6,7,8,9},
        {0,1,2,3,4,5,6,7,8,0},
        {0,1,2,3,4,5,4,0,0,0},
        };
    int32_t bias[CASE_NUM][1]   = {{0},{0},{0}};
    uint8_t output[][CASE_NUM][1] = {
        {
        {128 + 22},   //22.5
        {128 + 18},
        {128 + 10}    //9.5
        },{
        {128 - 22},   //22.5
        {128 - 18},
        {128 - 10}    //9.5
        }
        };
    uint32_t i,n;

    vsi_nn_tensor_attr_t input_attr;
    vsi_nn_tensor_attr_t kernel_attr;
    vsi_nn_tensor_attr_t bias_attr;
    vsi_nn_tensor_attr_t output_attr;

    if( !ctx )
    {
        ctx = vsi_nn_CreateContext();
    }

    memset( &input_attr, 0, sizeof( vsi_nn_tensor_attr_t ) );
    input_attr.size[0] = 1;
    input_attr.size[1] = 1;
    input_attr.size[2] = 10;
    input_attr.size[3] = 1;
    input_attr.dim_num = 4;
    input_attr.vtl = FALSE;
    input_attr.is_const = FALSE;
    input_attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    input_attr.dtype.scale = scale_input;
    input_attr.dtype.zero_point = 128;
    input_attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;

    memset( &kernel_attr, 0, sizeof( vsi_nn_tensor_attr_t ) );
    kernel_attr.size[0] = 1;
    kernel_attr.size[1] = 1;
    kernel_attr.size[2] = 10;
    kernel_attr.size[3] = 1;
    kernel_attr.dim_num = 4;
    kernel_attr.vtl = FALSE;
    kernel_attr.is_const = TRUE;
    kernel_attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    kernel_attr.dtype.scale = scale_kernel;
    kernel_attr.dtype.zero_point = 0;
    kernel_attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;

    memset( &bias_attr, 0, sizeof( vsi_nn_tensor_attr_t ) );
    bias_attr.size[0] = 1;
    bias_attr.dim_num = 1;
    bias_attr.vtl = FALSE;
    bias_attr.is_const = TRUE;
    bias_attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    bias_attr.dtype.scale = scale_input * scale_kernel;
    bias_attr.dtype.zero_point = 0;
    bias_attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;

    memset( &output_attr, 0, sizeof( vsi_nn_tensor_attr_t ) );
    output_attr.size[0] = 1;
    output_attr.size[1] = 1;
    output_attr.size[2] = 1;
    output_attr.size[3] = 1;
    output_attr.dim_num = 4;
    output_attr.vtl = FALSE;
    output_attr.is_const = FALSE;
    output_attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    output_attr.dtype.scale = scale_output;
    output_attr.dtype.zero_point = 128;
    output_attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;

    for( n = 0; n < _cnt_of_array(input); n ++)
    {
        for( i = 0; i < CASE_NUM; i ++ )
        {
            int t = run_test_conv2d( &input_attr, &kernel_attr, &bias_attr, &output_attr,
                    input[n], kernel[i], bias[i], output[n][i] );
            VSILOGI("Conv2d case %d %s", i + (n*CASE_NUM), t == 0 ? "fail" : "pass");
            ret &= t;
        }
    }

    input_attr.size[0] = 10;
    input_attr.size[1] = 1;
    input_attr.dim_num = 2;
    kernel_attr.size[0] = 10;
    kernel_attr.size[1] = 1;
    kernel_attr.dim_num = 2;
    bias_attr.size[0] = 1;
    bias_attr.dim_num = 1;
    output_attr.size[0] = 1;
    output_attr.size[1] = 1;
    output_attr.dim_num = 2;
    for( n = 0; n < _cnt_of_array(input); n ++)
    {
        for( i = 0; i < CASE_NUM; i ++ )
        {
            int t = run_test_fc( &input_attr, &kernel_attr, &bias_attr, &output_attr,
                    input[n], kernel[i], bias[i], output[n][i] );
            VSILOGI("Fc case %d %s", i + (n*CASE_NUM), t == 0 ? "fail" : "pass");
            ret &= t;
        }
    }
    return ret;
}


int main( int argc, char *argv[] )
{
    int ret = test_conv2d_fc();
    if( ret )
    {
        VSILOGI("Pass");
    }
    else
    {
        VSILOGE("Fail");
    }
    return ret;
}

