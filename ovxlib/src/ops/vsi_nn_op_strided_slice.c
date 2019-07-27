/****************************************************************************
*
*    Copyright 2012 - 2019 Vivante Corporation, Santa Clara, California.
*    All Rights Reserved.
*
*    Permission is hereby granted, free of charge, to any person obtaining
*    a copy of this software and associated documentation files (the
*    'Software'), to deal in the Software without restriction, including
*    without limitation the rights to use, copy, modify, merge, publish,
*    distribute, sub license, and/or sell copies of the Software, and to
*    permit persons to whom the Software is furnished to do so, subject
*    to the following conditions:
*
*    The above copyright notice and this permission notice (including the
*    next paragraph) shall be included in all copies or substantial
*    portions of the Software.
*
*    THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
*    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
*    IN NO EVENT SHALL VIVANTE AND/OR ITS SUPPLIERS BE LIABLE FOR ANY
*    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
*    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
*    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/


#include <string.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vx_nn_stride_slice_params_t param;
    vsi_nn_tensor_t *begin_dims_tensor = NULL;
    vsi_nn_tensor_t *end_dims_tensor = NULL;
    vsi_nn_tensor_t *stride_dims_tensor = NULL;
    vsi_nn_tensor_attr_t attr;

    memset(&param, 0, sizeof(vx_nn_stride_slice_params_t));

    memset(&attr, 0, sizeof(attr));
    attr.size[0] = self->nn_param.strided_slice.begin_dims_num;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    begin_dims_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)self->nn_param.strided_slice.begin_dims,
        &attr);
    if( NULL == begin_dims_tensor )
    {
        VSILOGE("Create begin_dims_tensor fail.(strided_slice)");
        return VSI_FAILURE;
    }

    self->nn_param.strided_slice.local.begin_dims_tensor = begin_dims_tensor;
    param.begin_dims = REQUIRED_IO(begin_dims_tensor);

    memset(&attr, 0, sizeof(attr));
    attr.size[0] = self->nn_param.strided_slice.end_dims_num;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    end_dims_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)self->nn_param.strided_slice.end_dims,
        &attr);
    if( NULL == end_dims_tensor )
    {
        VSILOGE("Create end_dims_tensor fail.(strided_slice)");
        return VSI_FAILURE;
    }

    self->nn_param.strided_slice.local.end_dims_tensor = end_dims_tensor;
    param.end_dims = REQUIRED_IO(end_dims_tensor);

    memset(&attr, 0, sizeof(attr));
    attr.size[0] = self->nn_param.strided_slice.stride_dims_num;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    stride_dims_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)self->nn_param.strided_slice.stride_dims,
        &attr);
    if( NULL == stride_dims_tensor )
    {
        VSILOGE("Create stride_dims_tensor fail.(strided_slice)");
        return VSI_FAILURE;
    }

    self->nn_param.strided_slice.local.stride_dims_tensor = stride_dims_tensor;
    param.stride_dims = REQUIRED_IO(stride_dims_tensor);

    param.begin_mask = self->nn_param.strided_slice.begin_mask;
    param.end_mask = self->nn_param.strided_slice.end_mask;
    param.shrink_axis_mask = self->nn_param.strided_slice.shrink_axis_mask;

    self->n = vxTensorStrideSliceNode(
        self->graph->g,
        inputs[0]->t,
        &param,
        sizeof(vx_nn_stride_slice_params_t),
        outputs[0]->t
        );

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    //TODO: Check tensor shapes.
    return TRUE;
} /* op_check() */

static vx_int32 get_slice_axis_value(vx_int32 value, vx_uint32 dimension_size)
{
    vx_int32 axis_vlaue = 0;
    if (value < 0)
        axis_vlaue = value + dimension_size;
    else
        axis_vlaue = value;
    return axis_vlaue;
}

static vx_int32 get_slice_mask_start_value(vx_int32 stride, vx_uint32 dimension_size)
{
    vx_int32 start_vlaue = 0;
    if (stride > 0)
        start_vlaue = 0;
    else
        start_vlaue = dimension_size - 1;
    return start_vlaue;
}

static vx_int32 get_slice_mask_stop_value(vx_int32 stride, vx_uint32 dimension_size)
{
    vx_int32 stop_vlaue = 0;
    if (stride > 0)
        stop_vlaue = dimension_size;
    else
        stop_vlaue = -1;
    return stop_vlaue;
}

static vx_int32 get_slice_clamp_stop(vx_int32 stride, vx_int32 stop, vx_uint32 dimension_size)
{
    vx_int32 stop_vlaue = 0;
    if (stride > 0)
    {
        stop_vlaue = vsi_nn_clamp(stop, 0, (vx_int32)dimension_size);
    }
    else
    {
        stop_vlaue = vsi_nn_clamp(stop, -1, (vx_int32)dimension_size - 1);
    }
    return stop_vlaue;
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* TODO: Add code to comput outputs' shape. */
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        vsi_nn_strided_slice_param *p = &(self->nn_param.strided_slice);
        vx_uint32 i;
        for (i = 0; i < inputs[0]->attr.dim_num; i++)
        {
            vx_int32 begin = 0, end = 1, stride = 1;
            vx_int32 input_size = inputs[0]->attr.size[i];
            vx_int32 output_size = 0;
            vx_int32 j;

            begin = get_slice_axis_value(p->begin_dims[i], input_size);
            end = get_slice_axis_value(p->end_dims[i], input_size);
            stride = p->stride_dims[i];
            if (p->begin_mask & (1 << i))
            {
                begin = get_slice_mask_start_value(stride, input_size);
            }
            begin = vsi_nn_clamp(begin, 0, (vx_int32)(input_size - 1));
            if (p->shrink_axis_mask & (1 << i))
            {
                end = begin + 1;
            }

            if (p->end_mask & (1 << i))
            {
                end = get_slice_mask_stop_value(stride, input_size);
            }
            end = get_slice_clamp_stop(stride, end, input_size);
            for (j = begin; !((stride > 0) ? (j >= end) : (j <= end)); j += stride)
            {
                output_size++;
            }
            outputs[0]->attr.size[i] = output_size;
        }
        outputs[0]->attr.dim_num = 0;
        for (i = 0; i < inputs[0]->attr.dim_num; i++)
        {
            if (p->shrink_axis_mask & (1 << i)) continue;
            outputs[0]->attr.size[outputs[0]->
                attr.dim_num] = outputs[0]->attr.size[i];
            outputs[0]->attr.dim_num++;
        }
    }
    return vx_true_e;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.strided_slice.local.begin_dims_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.strided_slice.local.begin_dims_tensor));
    }
    if (self->nn_param.strided_slice.local.end_dims_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.strided_slice.local.end_dims_tensor));
    }
    if (self->nn_param.strided_slice.local.stride_dims_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.strided_slice.local.stride_dims_tensor));
    }
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ STRIDED_SLICE,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

