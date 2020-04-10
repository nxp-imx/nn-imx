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
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
typedef enum
{
    INTERNAL_KERNEL_GRUCELL_ACTIVATION,
} _internal_kernel_e;

#define _GRUCELL_ACTIVATION_KERNEL_SOURCE      "grucell_activation"
#define _GRUCELL_ACTIVATION_KERNEL_NAME        CVIVANTE_NAMESPACE("evis.grucell_activation")

// Add kernel hashtable here
#define GRUCELL_ACTIVATION_HASH_KEY( IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, GATE_ACT, CAND_ACT ) \
        ((uint64_t)IN0_DTYPE | ( (uint64_t)IN1_DTYPE << 8 ) | ( (uint64_t)IN2_DTYPE << 16 ) \
        | ( (uint64_t)OUT_DTYPE << 24 ) | ( (uint64_t)GATE_ACT << 32 ) | ( (uint64_t)CAND_ACT << 40 ))


#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, GATE_ACT, CAND_ACT ) \
        { GRUCELL_ACTIVATION_HASH_KEY( IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, GATE_ACT, CAND_ACT ), \
          CVIVANTE_NAMESPACE("evis.grucell_activation_"#IN0_DTYPE"_"#IN1_DTYPE"_"#IN2_DTYPE"_to_"#OUT_DTYPE), \
          _GRUCELL_ACTIVATION_KERNEL_SOURCE }

typedef struct
{
    uint64_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _grucell_activation_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( U8, U8, U8, U8,     VSI_NN_ACT_SIGMOID, VSI_NN_ACT_TANH ),
    PACK_KERNEL_MAP( F16, F16, F16, F16, VSI_NN_ACT_SIGMOID, VSI_NN_ACT_TANH ),
    PACK_KERNEL_MAP( F16, F16, F16, U8,  VSI_NN_ACT_SIGMOID, VSI_NN_ACT_TANH ),
};


/*
 * Kernel params
 */
static vx_param_description_t _grucell_activation_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GRUCELL_ACTIVATION_PARAM_NUM  _cnt_of_array( _grucell_activation_kernel_param_def )
#define SCALAR_INPUT_GATE_ACT          (5)
#define SCALAR_INPUT_CANDIDATE_ACT     (6)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_grucell_activation_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        2,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    float    tensorScale[4]                 = {1.0f, 1.0f, 1.0f, 1.0f};
    float    tensorZP[4]                    = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t  i                             = 0;
    uint32_t  pack_key                      = 0;
    vsi_int_array_t * output_shape          = NULL;
    vsi_nn_kernel_tensor_attr_t * attr[4]   = { NULL, NULL, NULL, NULL };

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );
    attr[3] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[3], "Create tensor attr buffer fail.", final );

    for (i = 0; i < 4; i++)
    {
        if( attr[i]->quant == VSI_NN_KERNEL_QUANT_ASYMM
            || attr[i]->quant == VSI_NN_KERNEL_QUANT_SYMM)
        {
            tensorZP[i]     = (float)attr[i]->asymm.zero_point;
            tensorScale[i]  = attr[i]->asymm.scale;
        }
    }

    tensorZP[0] = tensorScale[0] * tensorZP[0];
    tensorZP[1] = tensorScale[1] * tensorZP[1];
    tensorZP[2] = tensorScale[2] * tensorZP[2];
    tensorScale[3] = 1.0f / tensorScale[3];

    output_shape  = attr[3]->shape;

    gpu_param.global_scale[0] = 4;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_size[0]  =
        gpu_align_p2((output_shape->data[0] + gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]  = output_shape->data[1];

#define _PACK_SELECT_KEY( IN0_TYPE, IN1_TYPE, IN2_TYPE, OUT_TYPE )    \
        (IN0_TYPE | (IN1_TYPE << 8) | (IN2_TYPE << 16) | ( OUT_TYPE << 24))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype,
                attr[2]->dtype, attr[3]->dtype );

    switch (pack_key)
    {
        case _PACK_SELECT_KEY( F16, F16, F16, F16 ):
        case _PACK_SELECT_KEY( F16, F16, F16, U8 ):
        case _PACK_SELECT_KEY( U8, U8, U8, U8 ):
            {
                gpu_dp_inst_t uniExtractHalf8_2x8 = {{
                    0x11111111, // TCfg
                    0x11110000, // ASelt
                    0x06040200, 0x06040200, // ABin
                    0x22222222, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000100, // AccumType, ConstantType, and PostShift
                    0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
                    0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
                }, GPU_DP_TYPE_16 };
                gpu_dp_inst_t uniConvDatatoFp32_4x4 = {{
                    0x01010101, // TCfg
                    0x00000000, // ASelt
                    0x00010000, 0x00030002, // ABin
                    0x02020202, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000100, // AccumType, ConstantType, and PostShift
                    0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                    0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
                }, GPU_DP_TYPE_16 };
                gpu_dp_inst_t uniExtractInteger_2x8 = {{
                    0x33333333, // TCfg
                    0x11110000, // ASelt
                    0x03020100, 0x03020100, // ABin
                    0x00000000, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00002400, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
                }, GPU_DP_TYPE_16 };

                if (attr[3]->dtype == F16)
                {
                    status = vsi_nn_kernel_gpu_add_param( node,
                            "uniExtract8Data_2x8", &uniExtractHalf8_2x8 );
                }
                else
                {
                    status = vsi_nn_kernel_gpu_add_param( node,
                            "uniExtract8Data_2x8", &uniExtractInteger_2x8 );
                }
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvDatatoFp32_4x4", &uniConvDatatoFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "tensorZP", &tensorZP );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "tensorScale", &tensorScale );
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
    default:
        break;
    }

    status |= vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    for (i = 0; i < 4; i++)
    {
        if (attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &attr[i] );
        }
    }

    return status;
} /* _grucell_activation_initializer() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t gate_activation,
    int32_t candidate_activation
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e in2_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _grucell_activation_kernel_map;
    vx_param_description_t * param_def  = _grucell_activation_kernel_param_def;
    vx_kernel_initialize_f  initializer = _grucell_activation_initializer;
    uint64_t key = 0;
    int32_t  i = 0;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    in2_dtype  = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = GRUCELL_ACTIVATION_HASH_KEY( in0_dtype, in1_dtype, in2_dtype, out_dtype,
                    gate_activation, candidate_activation );

    for( i = 0; i < _cnt_of_array( _grucell_activation_kernel_map ); i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array( _grucell_activation_kernel_map ) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = _cnt_of_array( _grucell_activation_kernel_param_def );;
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
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_GRUCELL_ACTIVATION_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t gate_activation = 0;
    int32_t candidate_activation = 0;

    gate_activation = vsi_nn_kernel_param_get_int32( params, "gate_activation" );
    candidate_activation = vsi_nn_kernel_param_get_int32( params, "candidate_activation" );

    status = _query_kernel( kernel, inputs, outputs, gate_activation, candidate_activation );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _GRUCELL_ACTIVATION_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            node_params[SCALAR_INPUT_GATE_ACT] = vsi_nn_kernel_scalar_create(
                graph, I32, &gate_activation );
            node_params[SCALAR_INPUT_CANDIDATE_ACT] = vsi_nn_kernel_scalar_create(
                graph, I32, &candidate_activation );
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _GRUCELL_ACTIVATION_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_GATE_ACT] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_CANDIDATE_ACT] );

        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( grucell_activation, _setup )

