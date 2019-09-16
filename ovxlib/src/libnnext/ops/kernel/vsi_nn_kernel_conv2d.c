/****************************************************************************
*
*    Copyright (c) 2018 Vivante Corporation
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
#include <stdlib.h>
#include <VX/vx_khr_cnn.h>
#include <VX/vx_helper.h>
#include <VX/vx.h>
#include <VX/vx_ext_program.h>

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_test.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"
#include "cpu_backend/cbee_interface.h"

#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_CONV2D)
#define _VX_KERNEL_VAR_CPU      (vx_client_kernel_CONV2D_CPU)
#define _VX_KERNEL_VAR_VX       (vx_client_kernel_CONV2D_VX)
#define _VX_KERNEL_NAME         ("com.vivantecorp.extension.Conv2DVXC")
#define _VX_KERNEL_FUNC_KERNEL  (vxConv2DKernel)

static vsi_status VX_CALLBACK vxConv2DKernel
    (
    vx_node node,
    const vx_reference* params,
    uint32_t param_num
    )
{
    int strides[2];
    int pad[4];
    int dilation[2];
    int multiplier;
    vx_context ctx;
    vx_status status;
    vx_tensor input;
    vx_tensor weight;
    vx_tensor bias;
    vx_tensor output;
    vsi_nn_tensor_attr_t input_attr;
    vsi_nn_tensor_attr_t weight_attr;
    vsi_nn_tensor_attr_t bias_attr;
    vsi_nn_tensor_attr_t output_attr;
    uint32_t output_size;
    uint8_t* input_buffer = NULL;
    uint8_t* weight_buffer = NULL;
    int32_t* bias_buffer = NULL;
    uint8_t* output_buffer = NULL;
    //uint8_t* col_buffer = NULL;
    //float float_multiplier;
    //int32_t quantized_multiplier = 0;
    //int32_t shift = 0;
    //int32_t col_h = 0;
    //int32_t col_w = 0;
    //int32_t batch;
    status = VX_SUCCESS;
    input  = (vx_tensor)params[0];
    weight  = (vx_tensor)params[1];
    bias  = (vx_tensor)params[2];
    output = (vx_tensor)params[3];
    vxCopyScalar( (vx_scalar)params[4], &strides[0],VX_READ_ONLY, VX_MEMORY_TYPE_HOST );
    vxCopyScalar( (vx_scalar)params[5], &strides[1],VX_READ_ONLY, VX_MEMORY_TYPE_HOST );
    vxCopyScalar( (vx_scalar)params[6], &pad[0],VX_READ_ONLY, VX_MEMORY_TYPE_HOST );
    vxCopyScalar( (vx_scalar)params[7], &pad[1],VX_READ_ONLY, VX_MEMORY_TYPE_HOST );
    vxCopyScalar( (vx_scalar)params[8], &pad[2],VX_READ_ONLY, VX_MEMORY_TYPE_HOST );
    vxCopyScalar( (vx_scalar)params[9], &pad[3],VX_READ_ONLY, VX_MEMORY_TYPE_HOST );
    vxCopyScalar( (vx_scalar)params[10], &dilation[0],VX_READ_ONLY, VX_MEMORY_TYPE_HOST );
    vxCopyScalar( (vx_scalar)params[11], &dilation[1],VX_READ_ONLY, VX_MEMORY_TYPE_HOST );
    vxCopyScalar( (vx_scalar)params[12], &multiplier,VX_READ_ONLY, VX_MEMORY_TYPE_HOST );
    ctx = vxGetContext((vx_reference)node);
    status = vsi_nn_vxGetTensorAttr( input, &input_attr );
    TEST_CHECK_STATUS( status, final );
    status = vsi_nn_vxGetTensorAttr( weight, &weight_attr );
    TEST_CHECK_STATUS( status, final );
    status = vsi_nn_vxGetTensorAttr( output, &output_attr );
    TEST_CHECK_STATUS( status, final );
    input_buffer = vsi_nn_vxCopyTensorToData( ctx, input, &input_attr);
    TEST_CHECK_PTR( input_buffer, final );
    weight_buffer = vsi_nn_vxCopyTensorToData( ctx, weight, &weight_attr);
    TEST_CHECK_PTR( weight_buffer, final );
    output_size = vsi_nn_vxGetTensorElementNum( &output_attr );
    output_buffer= (uint8_t*)malloc( output_size );
    memset( output_buffer, 0, output_size );
    //batch = output_attr.size[3];
    if( bias )
    {
        status = vsi_nn_vxGetTensorAttr( bias, &bias_attr );
        TEST_CHECK_STATUS( status, final );
        bias_buffer = (int32_t*)vsi_nn_vxCopyTensorToData( ctx, bias, &bias_attr);
        TEST_CHECK_PTR( bias_buffer, final );
    }
    // Compute
    cbee_interface_quant_conv2d(input_buffer, &input_attr,
            weight_buffer, &weight_attr, bias_buffer,
            pad, strides, dilation, &output_attr, output_buffer);

#if 0
    col_h = weight_attr.size[1] * weight_attr.size[0] * input_attr.size[2];
    col_w = output_attr.size[1] * output_attr.size[0];
    col_buffer = (uint8_t*)malloc( col_h * col_w);
    //VSILOGD("col: %d, %d",col_w,col_h);
    memset( col_buffer, 0, col_h * col_w );


    cbee_im2col( input_buffer, vsi_nn_GetTypeBytes(input_attr.dtype.vx_type),
            input_attr.size[2], &input_attr.size[0],
            &weight_attr.size[0], pad, strides, dilation, col_buffer );
    float_multiplier = (input_attr.dtype.scale * weight_attr.dtype.scale) / output_attr.dtype.scale;
    _compute_quantized_multiplier_and_shift( float_multiplier, &quantized_multiplier, &shift );
    //VSILOGD("multiplier: %f, %d, %d", float_multiplier, quantized_multiplier, shift);
    cbee_gemm_asymmetric_uint8( col_buffer, col_h, col_w,
            weight_buffer, (int32_t)weight_attr.size[3], col_h,
            bias_buffer, input_attr.dtype.zero_point,
            weight_attr.dtype.zero_point, output_attr.dtype.zero_point,
            quantized_multiplier, shift, output_buffer );
#endif
    vsi_nn_vxCopyDataToTensor( ctx, output, &output_attr, output_buffer );

final:
    if( input_buffer )
    {
        free( input_buffer );
    }
    if( weight_buffer )
    {
        free( weight_buffer );
    }
    if( bias_buffer )
    {
        free( bias_buffer );
    }
    if( output_buffer )
    {
        free( output_buffer );
    }
    //if( col_buffer )
    //{
    //    free( col_buffer );
    //}
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t s_params[] =
    {
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    };

static vx_status VX_CALLBACK vxConv2DInitializer
    (
    vx_node node,
    const vx_reference *params,
    vx_uint32 paraNum
    )
{
    return VX_SUCCESS;
}


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t _VX_KERNEL_VAR_CPU =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    s_params,
    _cnt_of_array( s_params ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t _VX_KERNEL_VAR_VX =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    NULL,
    s_params,
    _cnt_of_array( s_params ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxConv2DInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_CONV2D_list[] =
{
    &_VX_KERNEL_VAR_CPU,
    &_VX_KERNEL_VAR_VX,
    NULL
};
#ifdef __cplusplus
}
#endif
