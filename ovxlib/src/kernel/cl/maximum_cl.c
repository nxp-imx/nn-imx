/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define KERNEL_SOURCE_1    "maximum",
#define KERNEL_SOURCE_2    "maximum_fp16",
#define KERNEL_SOURCE_3    "maximum_i16"

#define HASH_MAXIMUM_KEY(_input0_type, _input1_type, _output_type, _image_2d) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_image_2d))

#define HASH_MAXIMUM_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("maximum_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE)

#define TENSOR_MAX_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MAXIMUM_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0), \
        HASH_MAXIMUM_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MAX_KERNELS_FLOAT(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MAXIMUM_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0), \
        HASH_MAXIMUM_SH_KERNEL_NAME(FP32, FP32, FP32), \
        SOURCE },


#define HASH_MAXIMUM_SH_KERNEL_2D_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("maximum_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE"_2D")

#define TENSOR_MAX_KERNELS_2D(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MAXIMUM_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 1), \
        HASH_MAXIMUM_SH_KERNEL_2D_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MAX_KERNELS_2D_FLOAT(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MAXIMUM_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 1), \
        HASH_MAXIMUM_SH_KERNEL_2D_NAME(FP32, FP32, FP32), \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } kernel_map[] =
{
    TENSOR_MAX_KERNELS_FLOAT(F32, F32, F32,  KERNEL_SOURCE_1)
    TENSOR_MAX_KERNELS_FLOAT(F16, F16, F16, KERNEL_SOURCE_1)
    TENSOR_MAX_KERNELS(U8, U8, U8, KERNEL_SOURCE_1)
    TENSOR_MAX_KERNELS(I32, I32, I32, KERNEL_SOURCE_1)

    TENSOR_MAX_KERNELS_2D_FLOAT(F32, F32, F32, KERNEL_SOURCE_1)
    TENSOR_MAX_KERNELS_2D_FLOAT(F16, F16, F16, KERNEL_SOURCE_1)
    TENSOR_MAX_KERNELS_2D(U8, U8, U8, KERNEL_SOURCE_1)
    TENSOR_MAX_KERNELS_2D(I32, I32, I32, KERNEL_SOURCE_1)
};

/*
 * Kernel params
 */
static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define SCALAR_INPUT0_SCALE          (3)
#define SCALAR_INPUT0_TAIL           (4)
#define SCALAR_INPUT1_SCALE          (5)
#define SCALAR_INPUT1_TAIL           (6)
#define SCALAR_OUTPUT_SCALE          (7)
#define SCALAR_OUTPUT_ZP             (8)
#define _CL_PARAM_NUM          _cnt_of_array(kernel_param_def)

/*
 * Kernel initializer
 */
static vx_status VX_CALLBACK _maximum_initializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status    status             = VX_SUCCESS;
    vx_tensor    input0             = (vx_tensor)paramObj[0];
    vx_tensor    input1             = (vx_tensor)paramObj[1];
    vx_tensor    output             = (vx_tensor)paramObj[2];

    vx_uint32 output_size[4] = {1, 1, 1, 1};

    vsi_nn_tensor_attr_t attr[3];
    uint32_t i, output_dim;

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[2], 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input0, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(input1, &attr[1]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[2]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
        return status;
    }

    output_dim  = attr[2].dim_num;
    for (i = 0; i < output_dim; i++)
    {
        output_size[i] = attr[2].size[i];
    }

    shaderParam.globalWorkScale[0]  = 1;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;

    shaderParam.globalWorkSize[0]   = gpu_align_p2((output_size[0] + shaderParam.globalWorkScale[0] - 1)
                                        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (output_size[1] + shaderParam.globalWorkScale[1] - 1)
                                        / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = output_dim > 2 ? output_size[2] : 1;

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return status;
} /* _maximum_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_bool image_2d,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input0_dtype;
    vsi_nn_kernel_dtype_e input1_dtype;
    vsi_nn_kernel_dtype_e output_dtype;
    vsi_status status = VSI_FAILURE;
    uint32_t key;
    int i;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    key = HASH_MAXIMUM_KEY( input0_dtype, input1_dtype, output_dtype, image_2d );

    for( i = 0; i < _cnt_of_array(kernel_map); i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _maximum_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 1,
                kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                kernel_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */

static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_CL_PARAM_NUM];
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;

    float input0Scale = inputs[0]->attr.dtype.scale;
    float input0Tail = (float)inputs[0]->attr.dtype.zero_point * input0Scale;
    float input1Scale = inputs[1]->attr.dtype.scale;
    float input1Tail = (float)inputs[1]->attr.dtype.zero_point * input1Scale;
    float outputScale = outputs[0]->attr.dtype.scale;
    float outputZP = (float)outputs[0]->attr.dtype.zero_point + 0.5f;

    outputScale = outputScale == 0.0 ? 0.0 : 1.0 / outputScale;

    if( !vsi_nn_kernel_gpu_check_shape( (int32_t*)outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    image_2d = (outputs[0]->attr.dim_num == 2);
    status = _query_kernel( inputs, outputs, image_2d, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );

        if( node )
        {
            vsi_nn_kernel_node_pack_io( node_params, _CL_PARAM_NUM,
                    inputs, 2, outputs, 1 );
            node_params[SCALAR_INPUT0_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &input0Scale );
            node_params[SCALAR_INPUT0_TAIL] = vsi_nn_kernel_scalar_create(
                    graph, F32, &input0Tail );
            node_params[SCALAR_INPUT1_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &input1Scale );
            node_params[SCALAR_INPUT1_TAIL] = vsi_nn_kernel_scalar_create(
                    graph, F32, &input1Tail );
            node_params[SCALAR_OUTPUT_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &outputScale );
            node_params[SCALAR_OUTPUT_ZP] = vsi_nn_kernel_scalar_create(
                    graph, F32, &outputZP );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CL_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT0_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT0_TAIL] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT1_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT1_TAIL] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_ZP] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( maximum, _setup )

