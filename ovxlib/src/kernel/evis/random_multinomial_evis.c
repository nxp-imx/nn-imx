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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS

typedef enum
{
    INTERNAL_KERNEL_SEED,
    INTERNAL_KERNEL_CDF,
    INTERNAL_KERNEL_MULTINOMIAL,
} _internal_kernel_e;

/*
 * Define kernel meta.
 */
#define _MULTINOMIAL_KERNEL_SOURCE  "random_multinomial"
#define _MULTINOMIAL_KERNEL_NAME    CVIVANTE_NAMESPACE("evis.random_multinomial")
#define _CDF_KERNEL_SOURCE          "random_multinomial"
#define _SEED_KERNEL_SOURCE         "random_multinomial"
#define _SEED_KERNEL_NAME           CVIVANTE_NAMESPACE("evis.random_seed")

// Add kernel hashtable here
#define MULTINOMIAL_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        ((IN0_DTYPE << 16) | ( IN1_DTYPE << 8 ) | ( OUT_DTYPE ))
#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, SOURCE ) \
        { MULTINOMIAL_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ), \
            _MULTINOMIAL_KERNEL_NAME, SOURCE }

#define CDF_HASH_KEY( IN_DTYPE, OUT_DTYPE ) \
        (( IN_DTYPE << 8 ) | ( OUT_DTYPE ))
#define CDF_PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE, SOURCE ) \
        { CDF_HASH_KEY( IN_DTYPE, OUT_DTYPE ), \
          CVIVANTE_NAMESPACE("evis.random_multinomial_cdf_"#IN_DTYPE), \
          SOURCE }

#define SEED_HASH_KEY( IN_DTYPE, OUT_DTYPE ) \
        (( IN_DTYPE << 8 ) | ( OUT_DTYPE ))
#define SEED_PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE, SOURCE ) \
        { SEED_HASH_KEY( IN_DTYPE, OUT_DTYPE ), _SEED_KERNEL_NAME, SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _seed_kernel_map[] =
{
    // Register kernel here
    SEED_PACK_KERNEL_MAP( I32, F32, _SEED_KERNEL_SOURCE ),
};

static const _kernel_map_type _cdf_kernel_map[] =
{
    // Register kernel here
    CDF_PACK_KERNEL_MAP( F16, F32, _CDF_KERNEL_SOURCE ),
    CDF_PACK_KERNEL_MAP( F32, F32, _CDF_KERNEL_SOURCE ),
};

static const _kernel_map_type _kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F32, F32, I32, _MULTINOMIAL_KERNEL_SOURCE ),
};

/*
 * Kernel params
 */
static vx_param_description_t _kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define SCALAR_CLASS_SIZE           (3)
#define SCALAR_CLASS_MAX_STRIDE     (4)
#define _PARAM_NUM  _cnt_of_array( _kernel_param_def )

static vx_param_description_t _cdf_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _CDF_PARAM_NUM  _cnt_of_array( _cdf_kernel_param_def )

static vx_param_description_t _seed_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _SEED_PARAM_NUM  _cnt_of_array( _seed_kernel_param_def )

/*
 * Kernel initializer
 */
static vx_status VX_CALLBACK _multinomial_initializer
    (
    vx_node              nodObj,
    const vx_reference * paramObj,
    uint32_t             paraNum
    )
{
    vx_status status = VX_SUCCESS;
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor     input           = (vx_tensor)paramObj[0];
    uint32_t      input_size[DIM_SIZE]   = {1, 1, 1, 1};
    uint32_t      sample_num = 0;
    uint32_t      batch = 0;
    vx_uint32 i = 0;
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status = vsi_nn_vxGetTensorAttr(input, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }
    for (i = 0; i < attr.dim_num; i++)
    {
        input_size[i] = attr.size[i];
    }

    sample_num = input_size[0];
    batch = input_size[1];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 4;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0]   = gpu_align_p2((sample_num + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = batch;
    shaderParam.globalWorkSize[2]   = 1;
    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }

    return status;
} /* _multinomial_initializer() */

static vx_status VX_CALLBACK _cdf_initializer
    (
    vx_node              nodObj,
    const vx_reference * paramObj,
    uint32_t             paraNum
    )
{
    vx_status status = VX_SUCCESS;
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor     input           = (vx_tensor)paramObj[0];
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    uint32_t      input_size[DIM_SIZE]   = {1, 1, 1, 1};
    uint32_t      class_size = 0, batch = 0;
    uint32_t      class_max_stride = 0;
    uint32_t      class_max_iter = 0;
    vx_uint32 i = 0;
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status = vsi_nn_vxGetTensorAttr(input, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }
    for (i = 0; i < attr.dim_num; i++)
    {
        input_size[i] = attr.size[i];
    }
    inputDataFormat = attr.dtype.vx_type;

    class_size = input_size[0];
    batch = input_size[1];
    if(inputDataFormat == VSI_NN_TYPE_FLOAT32)
    {
        class_max_iter = (class_size + 3) >> 2;
        class_max_stride = class_max_iter << 2;
    }
    else
    {
        class_max_iter = (class_size + 7) >> 3;
        class_max_stride = class_max_iter << 3;
    }

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 1;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0]   =  1;
    shaderParam.globalWorkSize[1]   = batch;
    shaderParam.globalWorkSize[2]   = 1;
    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }

    {
        vx_uint32 uniHorzSubMaxFp16_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };
        vx_uint32 uniConvertFstFp16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvertSecFp16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "class_max_iter", 1, &class_max_iter);
        status |= vxSetNodeUniform(nodObj, "class_max_stride", 1, &class_max_stride);
        status |= vxSetNodeUniform(nodObj, "uniConvertFstFp16Fp32_4x4", 1, uniConvertFstFp16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertSecFp16Fp32_4x4", 1, uniConvertSecFp16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniHorzSubMaxFp16_2x8", 1, uniHorzSubMaxFp16_2x8);
        if(status < 0)
        {
            VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        }
    }

    return status;
} /* _cdf_initializer() */

static vx_status VX_CALLBACK _seed_initializer
    (
    vx_node              nodObj,
    const vx_reference * paramObj,
    uint32_t             paraNum
    )
{
    vx_status status = VX_SUCCESS;
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor     output          = (vx_tensor)paramObj[1];
    uint32_t      output_size[DIM_SIZE]  = {0};
    uint32_t      stride = 0;
    uint32_t      iter = 8;
    uint32_t      w = 0;
    float         rand_max = (float)(pow(2.0,32));
    float         re_rand_max = 1 / rand_max;
    vx_uint32 i = 0;
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status = vsi_nn_vxGetTensorAttr(output, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }
    for (i = 0; i < attr.dim_num; i++)
    {
        output_size[i] = attr.size[i];
    }

    if(output_size[0] <= 4)
    {
        iter = 1;
        w = 1;
    }
    else if(output_size[0] <= 32)
    {
        iter = (output_size[0] + 3) / 4;
        w = 1;
    }
    else
    {
        w = (output_size[0] + 31) / 32;
    }
    stride = iter * 4;

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 1;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0]   = gpu_align_p2((w + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = output_size[1];
    shaderParam.globalWorkSize[2]   = 1;

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }

    status |= vxSetNodeUniform(nodObj, "stride", 1, &stride);
    status |= vxSetNodeUniform(nodObj, "iter", 1, &iter);
    status |= vxSetNodeUniform(nodObj, "re_rand_max", 1, &re_rand_max);
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }

    return status;
} /* _seed_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    const uint32_t hashkey,
    _internal_kernel_e kernel_id
    /* Add extra params */
    )
{
    vx_kernel_initialize_f  initializer = NULL;
    vx_param_description_t * param_def;
    vsi_status status = VSI_FAILURE;
    const _kernel_map_type* kernel_map;
    size_t kernel_map_size;
    size_t param_size;
    uint32_t i;

    switch( kernel_id )
    {
        case INTERNAL_KERNEL_SEED:
            initializer = _seed_initializer;
            kernel_map = _seed_kernel_map;
            kernel_map_size = _cnt_of_array( _seed_kernel_map );
            param_def = _seed_kernel_param_def;
            param_size = _SEED_PARAM_NUM;
            break;
        case INTERNAL_KERNEL_CDF:
            initializer = _cdf_initializer;
            kernel_map = _cdf_kernel_map;
            kernel_map_size = _cnt_of_array( _cdf_kernel_map );
            param_def = _cdf_kernel_param_def;
            param_size = _CDF_PARAM_NUM;
            break;
        case INTERNAL_KERNEL_MULTINOMIAL:
            initializer = _multinomial_initializer;
            kernel_map = _kernel_map;
            kernel_map_size = _cnt_of_array( _kernel_map );
            param_def = _kernel_param_def;
            param_size = _PARAM_NUM;
            break;
        default:
            VSI_ASSERT( FALSE );
            return VSI_FAILURE;
    }

    for( i = 0; i < kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == hashkey )
        {
            break;
        }
    }
    if( i < kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = param_size;
        kernel->info.initialize  = initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                kernel_map[i].source_name );
        // Register binary source
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
#define INTERNAL_KERNEL_SIZE    (2)
#define SEED_INDEX  (0)
#define CDF_INDEX   (1)
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_param_t cdf_node_params[_CDF_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_param_t seed_node_params[_SEED_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_kernel_t * ikernels[INTERNAL_KERNEL_SIZE] = { NULL };
    vsi_nn_tensor_t * tensors[INTERNAL_KERNEL_SIZE] = { NULL };
    int32_t class_max_stride = 0;
    int32_t class_size = 0;
    uint32_t hashkeys[INTERNAL_KERNEL_SIZE] = { 0 };
    uint32_t hashkey = 0;
    int32_t i;

    // Check if gpu can support the size
    if( !vsi_nn_kernel_gpu_check_shape(
        (int32_t*)outputs[0]->attr.size, outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    for( i = 0; i < INTERNAL_KERNEL_SIZE; i ++ )
    {
        ikernels[i] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
        // Assign unique_id
        ikernels[i]->unique_id = kernel->unique_id;
    }
    if( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 )
    {
        class_max_stride = (int32_t)(((inputs[0]->attr.size[0] + 3) >> 2) << 2);
    }
    else
    {
        class_max_stride = (int32_t)(((inputs[0]->attr.size[0] + 7) >> 3) << 3);
    }
    class_size = inputs[0]->attr.size[0];

    memcpy( &attr, &(outputs[0]->attr), sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    tensors[SEED_INDEX] = vsi_nn_CreateTensor( graph, &attr );

    if( attr.size[0] < GPU_TENSOR_MAX_WIDTH )
    {
        attr.size[0] = class_max_stride * inputs[0]->attr.size[1];
        attr.size[1] = 1;
    }
    else
    {
        attr.size[0] = class_max_stride;
        attr.size[1] = inputs[0]->attr.size[1];
    }
    attr.size[2] = 1;
    attr.size[3] = inputs[0]->attr.size[inputs[0]->attr.dim_num - 1];
    attr.dim_num = 4;
    tensors[CDF_INDEX] = vsi_nn_CreateTensor( graph, &attr );

    in0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    hashkeys[SEED_INDEX]= SEED_HASH_KEY( in1_dtype, F32 );
    hashkeys[CDF_INDEX] = CDF_HASH_KEY( in0_dtype, F32 );
    hashkey = MULTINOMIAL_HASH_KEY( F32, F32, out_dtype );

    status = _query_kernel( ikernels[SEED_INDEX], hashkeys[SEED_INDEX], INTERNAL_KERNEL_SEED );
    if( VSI_SUCCESS != status )
    {
        goto final;
    }
    status = _query_kernel( ikernels[CDF_INDEX], hashkeys[CDF_INDEX], INTERNAL_KERNEL_CDF );
    if( VSI_SUCCESS != status )
    {
        goto final;
    }
    status = _query_kernel( kernel, hashkey, INTERNAL_KERNEL_MULTINOMIAL );
    if( VSI_SUCCESS != status )
    {
        goto final;
    }

    // Seed
    node = vsi_nn_kernel_create_node( graph, ikernels[SEED_INDEX] );
    VSI_ASSERT( node != NULL );
    vsi_nn_kernel_node_pack_io( seed_node_params, _SEED_PARAM_NUM,
            &inputs[1], 1, &tensors[SEED_INDEX], 1 );
    status  = vsi_nn_kernel_node_pass_param( node, seed_node_params, _SEED_PARAM_NUM );
    VSI_ASSERT( status == VSI_SUCCESS );
    vsi_nn_kernel_node_release( &node );

    // CDF
    node = vsi_nn_kernel_create_node( graph, ikernels[CDF_INDEX] );
    VSI_ASSERT( node != NULL );
    vsi_nn_kernel_node_pack_io( cdf_node_params, _CDF_PARAM_NUM,
            &inputs[0], 1, &tensors[CDF_INDEX], 1 );
    status  = vsi_nn_kernel_node_pass_param( node, cdf_node_params, _CDF_PARAM_NUM );
    VSI_ASSERT( status == VSI_SUCCESS );
    vsi_nn_kernel_node_release( &node );

    // Multinomial
    node = vsi_nn_kernel_create_node( graph, kernel );
    VSI_ASSERT( node != NULL );
    vsi_nn_kernel_node_pack_io( node_params, _PARAM_NUM, tensors, 2, outputs, 1 );
    node_params[SCALAR_CLASS_SIZE] = vsi_nn_kernel_scalar_create( graph, I32, &class_size );
    node_params[SCALAR_CLASS_MAX_STRIDE] = vsi_nn_kernel_scalar_create(
            graph, I32, &class_max_stride );
    status  = vsi_nn_kernel_node_pass_param( node, node_params, _PARAM_NUM );
    VSI_ASSERT( status == VSI_SUCCESS );
    vsi_nn_kernel_scalar_release( &node_params[SCALAR_CLASS_SIZE] );
    vsi_nn_kernel_scalar_release( &node_params[SCALAR_CLASS_MAX_STRIDE] );

    /* Pass parameters to node. */
final:
    for( i = 0; i < INTERNAL_KERNEL_SIZE; i ++ )
    {
        if( ikernels[i] )
        {
            vsi_nn_kernel_release( &ikernels[i] );
        }
        if( tensors[i] )
        {
            vsi_nn_ReleaseTensor( &tensors[i] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( random_multinomial, _setup )

