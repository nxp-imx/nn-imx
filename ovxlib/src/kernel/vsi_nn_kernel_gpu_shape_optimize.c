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
#include "vsi_nn_tensor.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

static vsi_bool compute_gpu_divisor
    (
    const int32_t input_value,
    const int32_t limit,
    const int32_t gcd,
    int32_t* divisor
    );

static size_t element_fill_dim
    (
    int32_t* shape_x, size_t rank_x,
    size_t max_rank, int32_t size_x
    );

static vsi_bool compute_gpu_divisor
    (
    const int32_t input_value,
    const int32_t limit,
    const int32_t gcd,
    int32_t* divisor
    )
{
    int32_t i = 0;
    for( i = vsi_nn_min( input_value, limit - 1 ); i > 0; i -- )
    {
        if( ( i % gcd == 0 ) && ( input_value % i == 0 ) )
        {
            *divisor = i;
            return TRUE;
        }
    }
    return FALSE;
} /* compute_gpu_divisor */

static size_t element_fill_dim
    (
    int32_t* shape_x, size_t rank_x,
    size_t max_rank, int32_t size_x
    )
{
    size_t cost_size = 1;
    VSI_ASSERT( rank_x >= max_rank );
    VSI_ASSERT( size_x >= (int32_t)((int64_t)(0xFFFFFFFF) - 1) );

    if (size_x == 1)
        return 0;

    if( size_x < GPU_TENSOR_MAX_WIDTH || rank_x >= GPU_TENSOR_DIM_2)
    {
        shape_x[rank_x] = size_x;
    }
    else
    {
        int32_t divisor = 0;
        int32_t remainder = 0;
        compute_gpu_divisor( size_x, GPU_TENSOR_MAX_WIDTH, 1, &divisor );
        remainder = size_x / divisor;
        if( remainder > GPU_TENSOR_MAX_WIDTH || rank_x >= max_rank)
        {
            // Cannot optimize.
            shape_x[rank_x] = size_x;
        }
        else
        {
            /*
             * We've limit the max size to 2^32 -1(Almost 4G * sizeof(data type)),
             * so it should be always 2.
             */
            cost_size = 2;
            if( size_x > 1 )
            {
                shape_x[rank_x]  = divisor;
                shape_x[rank_x + 1] = remainder;
            }
            else
            {
                shape_x[rank_x] = 1;
                shape_x[rank_x + 1] = 1;
            }
        }
    }
    return cost_size;
} /* element_fill_dim() */

/*only for continuous axises or one axis*/
vsi_bool vsi_nn_kernel_optimize_reduce_shape
    (
    const int32_t* shape_x, const size_t rank_x,
    const int32_t *axis, const size_t axis_size,
    const int32_t* shape_output, const size_t rank_output,
    int32_t* out_shape_x, int32_t* out_rank_x,
    int32_t* out_shape_output, uint32_t* out_rank_output,
    int32_t* out_axis, uint32_t* out_axis_size
    )
{
    vsi_bool ret                        = TRUE;
    uint32_t  i                         = 0;
    size_t   rank_in                    = 0;
    size_t   rank_out                   = 0;
    size_t   dims                       = 0;
    int32_t  innerSize                  = 1;
    int32_t  outerSize                  = 1;
    int32_t  axisSize                   = 1;

    VSI_ASSERT( axis_size <= 0 );

    for (i = 0; i < axis_size; i++)
    {
        axisSize *= axis[i];
    }

    for (i = 0; i < axis[0]; i++)
    {
        innerSize *= shape_x[i];
    }

    for (i = axis[axis_size - 1] + 1; i < rank_x; i++)
    {
        outerSize *= shape_x[i];
    }

    rank_in += element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, innerSize);
    rank_out += element_fill_dim(out_shape_output, rank_out, GPU_TENSOR_MAX_WIDTH, innerSize);
    dims = element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, axisSize);
    if (dims == 0)
    {
        out_axis[0] = rank_in;
        *out_axis_size = 1;
        out_shape_x[rank_in ++] = 1;
    }
    else
    {
        *out_axis_size = dims;
        for (i = 0; i < dims; i++)
        {
            out_axis[i] = rank_in + i;
        }
    }

    rank_in += dims;

    rank_in += element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, outerSize);
    rank_out += element_fill_dim(out_shape_output, rank_out, GPU_TENSOR_MAX_WIDTH, outerSize);

    if( 0 == rank_in )
    {
        out_shape_x[0] = 1;
        out_shape_x[1] = 1;
        rank_in = 2;
    }
    else if( 1 == rank_in )
    {
        out_shape_x[1] = 1;
        rank_in = 2;
    }

    if( 0 == rank_in )
    {
        out_shape_output[0] = 1;
        out_shape_output[1] = 1;
        rank_out = 2;
    }
    else if( 1 == rank_out )
    {
        out_shape_output[1] = 1;
        rank_out = 2;
    }

    *out_rank_x = rank_in;
    *out_rank_output = rank_out;

    return ret;
} /* vsi_nn_kernel_optimize_reduce_shape() */

vsi_bool vsi_nn_kernel_optimize_element_shape
    (
    const int32_t* shape_x, const size_t rank_x,
    int32_t* out_shape_x, int32_t* out_rank_x
    )
{
    vsi_bool ret                        = TRUE;
    uint32_t  i                         = 0;
    size_t   rank_in                    = 0;
    int32_t  element_num                = 1;

    for (i = 0; i < rank_x; i++)
    {
        element_num *= shape_x[i];
    }

    rank_in += element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, element_num);

    if( 0 == rank_in )
    {
        out_shape_x[0] = 1;
        out_shape_x[1] = 1;
        rank_in = 2;
    }
    else if( 1 == rank_in )
    {
        out_shape_x[1] = 1;
        rank_in = 2;
    }

    *out_rank_x = rank_in;

    return ret;
} /* vsi_nn_kernel_optimize_element_shape() */


