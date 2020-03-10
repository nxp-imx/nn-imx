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
#include <float.h>
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_error.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.random_multinomial")


/*
 * Kernel params
 */
static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _PARAM_NUM  _cnt_of_array( kernel_param_def )


/*
 * Kernel function
 */
static int upper_bound(float* a, int n, float x) {
    int l = 0;
    int h = n;
    while (l < h) {
        int mid = (l + h) / 2;
        if (x >= a[mid]) {
            l = mid + 1;
        } else {
            h = mid;
        }
    }
    return l;
} /* upper_bound() */

static vx_status VX_CALLBACK _compute
    (
    vx_node              node,
    const vx_reference * param,
    uint32_t             param_size
    )
{
    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_tensor input[_INPUT_NUM] = {0};
    vx_tensor output[_OUTPUT_NUM] = {0};
    float *f32_in_buffer[_INPUT_NUM] = {0};
    int32_t* int32_in_buffer[_INPUT_NUM] = {0};
    int32_t *int32_out_buffer[_OUTPUT_NUM] = {0};
    vsi_nn_tensor_attr_t in_attr[_INPUT_NUM];
    vsi_nn_tensor_attr_t out_attr[_OUTPUT_NUM];
    uint32_t in_elements[_INPUT_NUM] = {0};
    uint32_t out_elements[_OUTPUT_NUM]= {0};

    int32_t sample_num;

    int32_t i;
    for(i = 0; i < _INPUT_NUM; i++)
    {
        memset(&in_attr[i], 0x0, sizeof(vsi_nn_tensor_attr_t));
    }
    for(i = 0; i < _OUTPUT_NUM; i++)
    {
        memset(&out_attr[i], 0x0, sizeof(vsi_nn_tensor_attr_t));
    }
    /* prepare data */
    context = vxGetContext((vx_reference)node);

    for(i = 0; i < _INPUT_NUM; i ++)
    {
        input[i] = (vx_tensor)param[i];
        status = vsi_nn_vxGetTensorAttr(input[i], &in_attr[i]);
        CHECK_STATUS_FAIL_GOTO(status, final);
        in_elements[i] = vsi_nn_vxGetTensorElementNum(&in_attr[i]);
        if (i == 1)
        {
            int32_in_buffer[i] = (int32_t *)vsi_nn_vxCopyTensorToData(context,
                input[i], &in_attr[i]);
        }
        else
        {
            f32_in_buffer[i] = (float *)malloc(in_elements[i] * sizeof(float));
            status = vsi_nn_vxConvertTensorToFloat32Data(
                context, input[i], &in_attr[i], f32_in_buffer[i],
                in_elements[i] * sizeof(float));
            CHECK_STATUS_FAIL_GOTO(status, final);
        }
    }
    for(i = 0; i < _OUTPUT_NUM; i ++)
    {
        output[i] = (vx_tensor)param[i + _INPUT_NUM];
        status = vsi_nn_vxGetTensorAttr(output[i], &out_attr[i]);
        CHECK_STATUS_FAIL_GOTO(status, final);
        out_elements[i] = vsi_nn_vxGetTensorElementNum(&out_attr[i]);
        int32_out_buffer[i]= (int32_t *)malloc(out_elements[i] * sizeof(int32_t));
        memset(int32_out_buffer[i], 0, out_elements[i] * sizeof(int32_t));
    }
    sample_num = out_attr[0].size[0];

    {
        uint32_t n, c;
        uint32_t batch = in_attr[0].size[1];
        uint32_t class_size = in_attr[0].size[0];
        float *cdf = (float *)malloc(class_size * sizeof(float));
        uint32_t *random_integer = (uint32_t *)malloc(out_elements[0] * sizeof(uint32_t));
        float *random_float = (float *)malloc(out_elements[0] * sizeof(float));
        vsi_nn_random_init_for_philox_4x32_10((uint32_t)(int32_in_buffer[1][0]),
            (uint32_t)(int32_in_buffer[1][1]));
        vsi_nn_random_generate_by_philox_4x32_10(random_integer, out_elements[0]);
        vsi_nn_random_uniform_transform(random_integer,
            random_float, out_elements[0]);
        for(n = 0; n < batch; n++)
        {
            float batch_max = -FLT_MAX;
            float total = 0;
            for(c = 0; c < class_size; c++)
            {
                uint32_t index = n * class_size + c;
                batch_max = vsi_nn_max(batch_max, f32_in_buffer[0][index]);
            }
            for(c = 0; c < class_size; c++)
            {
                uint32_t index = n * class_size + c;
                total += (float)(exp(f32_in_buffer[0][index] - batch_max));
                cdf[c] = total;
            }

            for(c = 0; c < (uint32_t)sample_num; c++)
            {
                uint32_t index = n * sample_num + c;
                float target = random_float[index] * total;
                uint32_t out_class = upper_bound(cdf, class_size, target);
                int32_out_buffer[0][index] = out_class;
            }
        }

        if (cdf) free(cdf);
        if (random_integer) free(random_integer);
        if (random_float) free(random_float);
    }

    /* save data */
    for(i = 0; i < _OUTPUT_NUM; i++)
    {
        vsi_nn_vxCopyDataToTensor(context, output[i], &out_attr[i],
            (uint8_t *)(int32_out_buffer[i]));
    }

final:
    for (i = 0; i < _INPUT_NUM; i++)
    {
        if (f32_in_buffer[i]) free(f32_in_buffer[i]);
        if (int32_in_buffer[i]) free(int32_in_buffer[i]);
    }
    for(i = 0; i < _OUTPUT_NUM; i++)
    {
        if (int32_out_buffer[i]) free(int32_out_buffer[i]);
    }
    return status;
} /* _compute() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( kernel_param_def );
    status = VSI_SUCCESS;
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
    vsi_nn_kernel_node_param_t node_params[_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _PARAM_NUM );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( random_multinomial, _setup )

