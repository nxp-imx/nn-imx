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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_VAR          (vx_kernel_SPATIAL_TRANSFORMER)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_SPATIAL_TRANSFORMER)
#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_SPATIAL_TRANSFORMER)
#define _VX_KERNEL_FUNC_KERNEL  (vxSpatial_transformerKernel)

#if 1
static void transform_gemm_cpu(const void *src0_ptr, int M, int N, int K, const void *src1_ptr, const void *dst_ptr)
{
    //vx_uint32  elementCount      = 1;
    vx_uint32  i                 = 0;
    vx_uint32  j                 = 0;
    vx_uint32  k                 = 0;
    //vx_uint32  z                 = 0;
    /*vx_uint32  m                 = 0;
    vx_uint32  n                 = 0;
    vx_float32 dx                = 0;
    vx_float32 dy                = 0;*/
    float     *f32_src0 = (float *)src0_ptr;
    float     *f32_src1 = (float *)src1_ptr;
    float     *f32_dst  = (float *)dst_ptr;
    //vx_uint32  src0_depth        = src0_params_t.sizes[2];
    //vx_uint32  dst_width         = dst_params_t.sizes[0];
    //vx_uint32  dst_height        = dst_params_t.sizes[1];
    //vx_uint32  dst_depth         = dst_params_t.sizes[2];

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            //vx_uint32 idx = i * M + j;
            vx_float32 sum = 0;

            for (k = 0; k < K; k++)
            {
                vx_uint32 idx0 = j * K + k;
                vx_uint32 idx1 = i * K + k;

                vx_float32 src0_val     = f32_src0[idx0];
                vx_float32 src1_val     = f32_src1[idx1];

                sum += src0_val * src1_val;
            }
            f32_dst[j * N +  i] = sum;
            //vxnneSaveDataExt_(idx, sum, dst_ptr, dst_params_t);
        }
    }
}

static int st_process(float *data_in, float *pos, float *output, int in_w, int in_h, int out_w, int out_h,  int c)
{
    int i = 0;
    int j = 0;
    int k = 0;
    float px = 0.0;
    float py = 0.0;
    int idx = 0;
    int m = 0;
    int n = 0;
    float value = 0.0;

    for(i = 0; i < c; i++)
    {
        idx = 0;
        for(j = 0; j < out_h; j++)
        {
            for(k = 0; k < out_w; k++)
            {
                px = pos[idx];
                py = pos[idx + 1 ];
                idx = idx + 2;

                px = (px + 1)/ 2 * in_h;
                py = (py + 1)/ 2 * in_w;
                value = 0.0;

                for(m = floor(px); m <= ceil(px); m++)
                {
                    for(n = floor(py); n <= ceil(py); n++)
                    {
                        if(m >= 0 && m < in_h && n >= 0 && n < in_w)
                        {
                            value += (1 - fabs(px-m)) * (1 - fabs(py-n)) * data_in[ i * (in_w * in_h) + m * in_w + n];
                        }
                    }
                }
                output[i * out_h * out_w +  j * out_w + k] = value;
            }
        }
    }
    return 0;
}
#endif

static vsi_status VX_CALLBACK vxSpatial_transformerKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vx_tensor input_data_t;
    vx_tensor thre_data_t;
    vx_scalar flag_s;
    vx_tensor thre_proto_t;
    vx_tensor output_t;
    vx_context context;
    vx_tensor_addressing in_addr,out_addr, thre_data_addr,thre_proto_addr;
    vsi_nn_tensor_attr_t in_attr,out_attr, thre_data_attr,thre_proto_attr;
    uint32_t in_stride[6],out_stride[6], thre_data_stride[6],thre_proto_stride[6];
    vsi_status status = VX_SUCCESS;

    float *f32_in_buffer = NULL,*f32_out_buffer = NULL,*f32_thre_proto_buffer = NULL;
    uint8_t *in_buffer = NULL, *out_buffer = NULL, *thre_data_buffer;
    int dim = 0;
    int size = 0;
    int i = 0;
    int flag;
    float thre_value[6];
    int thre_num = 0;
    float *f32_grid = NULL;
    float *f32_out_grid = NULL;
    float tmp = 0;

    //VX_TYPE_FLOAT16 *in_buffer = NULL, *out_buffer = NULL;

    /* TODO: Add CPU kernel implement */
    if(paramNum != 5)
    {
        return VSI_FAILURE;
    }

    input_data_t = (vx_tensor)paramObj[0];
    thre_data_t =  (vx_tensor)paramObj[1];
    output_t =     (vx_tensor)paramObj[2];
    flag_s =       (vx_scalar)paramObj[3];
    thre_proto_t = (vx_tensor)paramObj[4];

    context = vxGetContext((vx_reference)node);

    in_buffer = (uint8_t *)vsi_nn_ConvertRawTensorToData2(context,
                                input_data_t,&in_attr,in_stride,&in_addr,VX_READ_ONLY);
    out_buffer = (uint8_t *)vsi_nn_ConvertRawTensorToData2(context,
                                output_t,&out_attr,out_stride,&out_addr,VX_WRITE_ONLY);
    thre_data_buffer = (uint8_t *)vsi_nn_ConvertRawTensorToData2(context,
                                thre_data_t,&thre_data_attr,thre_data_stride,&thre_data_addr,VX_READ_ONLY);
    f32_thre_proto_buffer = (float *)vsi_nn_ConvertRawTensorToData2(context,
                                thre_proto_t,&thre_proto_attr,thre_proto_stride,&thre_proto_addr,VX_READ_ONLY);

    f32_grid = (float *)malloc(out_attr.size[0] * out_attr.size[1] * 3 * sizeof(float));
    f32_out_grid = (float *)malloc(out_attr.size[0] * out_attr.size[1] * 2 * sizeof(float));
    if(f32_out_grid == NULL || f32_out_grid== NULL)
    {
        printf("Malloc space for grid failed \n");
        return VSI_FAILURE;
    }
    for(i = 0; i < out_attr.size[0] * out_attr.size[1]; i++)
    {
        tmp = ( i / out_attr.size[0] ) * 1.0 / out_attr.size[1] * 2 - 1;
        f32_grid[i * 3] = tmp;
        tmp = i;
        tmp = ( i % out_attr.size[0] ) * 1.0 / out_attr.size[0] * 2 - 1;
        f32_grid[i * 3 + 1] = tmp;
        f32_grid[i * 3 + 2] = 1;
    }

    dim = in_attr.dim_num;
    size = 1;
    for(i = 0; i < dim; i++)
        size *= in_attr.size[i];

    f32_in_buffer = (float*)malloc(size*sizeof(float));
    if(f32_in_buffer == NULL)
    {
        printf("Malloc space for input failed \n");
        return VSI_FAILURE;
    }

    for(i=0; i<size; i++)
        vsi_nn_DtypeToFloat32((uint8_t*)&in_buffer[in_stride[0] * i], &f32_in_buffer[i], &in_attr.dtype);

    vxCopyScalar(flag_s,&flag,VX_READ_ONLY,VX_MEMORY_TYPE_HOST);

    for(i = 0; i < 6; i++)
    {
        if( (flag & (1 << i)) != 0)
            thre_num ++;
    }
    dim = thre_data_attr.dim_num;
    size = 1;
    for(i = 0; i < dim; i++)
        size *= thre_data_attr.size[i];
    if(size + thre_num > 6)
    {
        printf("The dim of thre must 6\n");
        return VSI_FAILURE;
    }
    thre_num = 0;
    for(i = 0; i < size; i++)
    {
        vsi_nn_DtypeToFloat32((uint8_t*)&thre_data_buffer[thre_data_stride[0] * i],
                                &thre_value[i], &thre_data_attr.dtype);
    }
    for(i = 0; i < 6; i++)
    {
        if( (flag & (1 << i)) != 0)
        {
            thre_value[i] = f32_thre_proto_buffer[i];
        }
        else
        {
            vsi_nn_DtypeToFloat32((uint8_t*)&thre_data_buffer[thre_data_stride[0] * thre_num],
                                    &thre_value[i], &thre_data_attr.dtype);
            thre_num++;
        }
    }

    dim = out_attr.dim_num;
    size = 1;
    for(i = 0; i < dim; i++)
        size *= out_attr.size[i];
    f32_out_buffer = (float*)malloc(size*sizeof(float));
    if(f32_out_buffer == NULL)
    {
        printf("Malloc space for output failed \n");
        return VSI_FAILURE;
    }
    transform_gemm_cpu(f32_grid,out_attr.size[0] * out_attr.size[1],2,3,thre_value,f32_out_grid);

    st_process(f32_in_buffer,f32_out_grid,f32_out_buffer,
                in_attr.size[0],in_attr.size[1],out_attr.size[0],out_attr.size[1],in_attr.size[2]);

    for(i=0; i<size; i++)
        vsi_nn_Float32ToDtype(f32_out_buffer[i], (uint8_t*)&out_buffer[out_stride[0] * i], &out_attr.dtype);
    status = vxCopyTensorPatch(output_t,NULL,out_addr,out_buffer,VX_WRITE_ONLY,0);

    if (out_addr)
    {
        vxReleaseTensorAddressing(&out_addr);
    }

    free(f32_in_buffer);
    f32_in_buffer = NULL;

    free(f32_grid);
    f32_grid = NULL;

    free(f32_out_grid);
    f32_out_grid = NULL;

    free(f32_out_buffer);
    f32_out_buffer = NULL;

    free(in_buffer);
    in_buffer = NULL;

    free(out_buffer);
    out_buffer = NULL;

    free(thre_data_buffer);
    thre_data_buffer = NULL;

    free(f32_thre_proto_buffer);
    f32_thre_proto_buffer = NULL;

    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t s_params[] =
{
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
};

vx_status VX_CALLBACK vxTransform_GemmInputValidator(vx_node node, vx_uint32 index)
{
    return VX_SUCCESS;
}

vx_status VX_CALLBACK vxTransform_GemmOutputValidator(vx_node node, vx_uint32 index, vx_meta_format metaObj)
{
    return VX_SUCCESS;
}

vx_status VX_CALLBACK vxValidator(vx_node node, const vx_reference parameters[],
                                    vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_uint32 index = 0;
    for(index = 0; index < num; index++)
    {
        if(index < 2)
        {
            status |= vxTransform_GemmInputValidator(node,index);
        }
        else
        {
            status |= vxTransform_GemmOutputValidator(node,index,metas[index]);
        }
    }
    return status;
}

static vx_param_description_t vxTransform_GemmKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};

vx_status VX_CALLBACK vxTransform_GemmInitializer(vx_node nodObj, const vx_reference *paramObj, vx_uint32 paraNum)
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
#define gcmMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define gcmMAX(x, y)            (((x) >= (y)) ?  (x) :  (y))
#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status    status             = VX_SUCCESS;
    vx_tensor    input0             = (vx_tensor)paramObj[0];
    vx_tensor    input1             = (vx_tensor)paramObj[1];
    vx_tensor    output             = (vx_tensor)paramObj[2];
    vx_enum      src0Format         = VX_TYPE_FLOAT16;
    vx_enum      src1Format         = VX_TYPE_FLOAT16;
    vx_enum      dstFormat          = VX_TYPE_FLOAT16;
    vx_enum      src0QuantType      = 0;
    vx_int8      src0FixPointPos    = 0;
    vx_enum      src1QuantType      = 0;
    vx_int8      src1FixPointPos    = 0;
    vx_enum      dstQuantType       = 0;
    vx_int8      dstFixPointPos     = 0;

    vx_uint32    coord_size[4]      = {0, 0, 0, 0};
    vx_uint32    input_size[4]      = {0, 0, 0, 0};
    vx_uint32    output_size[4]     = {0, 0, 0, 0};

    status = vxQueryTensor(input0, VX_TENSOR_DATA_TYPE, &src0Format, sizeof(src0Format));
    status |= vxQueryTensor(input0, VX_TENSOR_QUANT_FORMAT, &src0QuantType, sizeof(src0QuantType));
    status |= vxQueryTensor(input0, VX_TENSOR_FIXED_POINT_POSITION, &src0FixPointPos, sizeof(src0FixPointPos));
    status |= vxQueryTensor(input0, VX_TENSOR_DIMS, input_size, sizeof(input_size));

    status |= vxQueryTensor(input1, VX_TENSOR_DATA_TYPE, &src1Format, sizeof(src1Format));
    status |= vxQueryTensor(input1, VX_TENSOR_QUANT_FORMAT, &src1QuantType, sizeof(src1QuantType));
    status |= vxQueryTensor(input1, VX_TENSOR_FIXED_POINT_POSITION, &src1FixPointPos, sizeof(src1FixPointPos));
    status |= vxQueryTensor(input1, VX_TENSOR_DIMS, coord_size, sizeof(coord_size));
    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &dstFormat, sizeof(dstFormat));
    status |= vxQueryTensor(output, VX_TENSOR_QUANT_FORMAT, &dstQuantType, sizeof(dstQuantType));
    status |= vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POSITION, &dstFixPointPos, sizeof(dstFixPointPos));
    status |= vxQueryTensor(output, VX_TENSOR_DIMS, output_size, sizeof(output_size));

    if(status < 0)
        printf("error-%s,%d\n",__FILE__,__LINE__);

    if (src0Format == VX_TYPE_FLOAT16 && src1Format == VX_TYPE_FLOAT16 && dstFormat == VX_TYPE_FLOAT16)
    {
        shaderParam.globalWorkScale[0]  = 12;
        shaderParam.globalWorkScale[1]  = 1;
    }

    shaderParam.globalWorkSize[0]   =
                gcmALIGN((coord_size[0] + shaderParam.globalWorkScale[0] - 1) / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   =
                (coord_size[1] + shaderParam.globalWorkScale[1] - 1) / shaderParam.globalWorkScale[1];
    {
        vx_uint32 uniGemm3x3_4x4[16] = {
            0x15151515, // TCfg
            0x00000000, // ASelt
            0x02100210, 0x05430543, // ABin
            0x15151515, // BSelt
            0x05430210, 0x05430210, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vxSetNodeUniform(nodObj, "uniGemm3x3_4x4", 1, uniGemm3x3_4x4);
    }
    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                    &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return VX_SUCCESS;
}

static vx_param_description_t vxTransform_setupThresKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};

vx_status VX_CALLBACK vxTransform_setupThresInitializer(vx_node nodObj,
                                                        const vx_reference *paramObj, vx_uint32 paraNum)
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
#define gcmMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define gcmMAX(x, y)            (((x) >= (y)) ?  (x) :  (y))
#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status    status             = VX_SUCCESS;
    vx_scalar    thresFlag_s        = (vx_scalar)paramObj[2];
    vx_enum      src0Format         = VX_TYPE_FLOAT16;
    vx_enum      src1Format         = VX_TYPE_FLOAT16;

    vx_int32     thresFlag          = 0;
    vx_uint32    extract_packed[4]  = {0};

    vxCopyScalar(thresFlag_s, &thresFlag, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if(status < 0)
        printf("error-%s,%d\n",__FILE__,__LINE__);

    shaderParam.globalWorkScale[0]  = 1;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.localWorkSize[0]    = 1;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.globalWorkSize[0]   = 1;
    shaderParam.globalWorkSize[1]   = 1;

    if (src0Format == src1Format && src0Format == VX_TYPE_FLOAT16)
    {
        vx_uint32 i = 0;
        vx_uint32 j = 0;
        for (i = 0; i < 4; i++)
        {
            if (thresFlag & (1 << i))
            {
                extract_packed[0] |= ((i << 4) << (i * 8));
            }
            else
            {
                extract_packed[0] |= (((j << 4) + 128) << (i * 8));
                j ++;
            }
        }

        for (i = 4; i < 6; i++)
        {
            if (thresFlag & (1 << i))
            {
                extract_packed[1] |= ((i << 4) << (i * 8 - 32));
            }
            else
            {
                extract_packed[1] |= (((j << 4) + 128) << (i * 8 - 32));
                j ++;
            }
        }

        extract_packed[2] = extract_packed[3] = 0x10101010;
    }

    vxSetNodeUniform(nodObj, "extract_packed", 1, extract_packed);

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                                &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return VX_SUCCESS;
}


static vx_param_description_t vxTransform_InterPKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};


vx_status VX_CALLBACK vxTransform_InterPInitializer(vx_node nodObj, const vx_reference *paramObj, vx_uint32 paraNum)
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
#define gcmMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define gcmMAX(x, y)            (((x) >= (y)) ?  (x) :  (y))
#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status    status             = VX_SUCCESS;
    vx_tensor    input0             = (vx_tensor)paramObj[0];
    vx_tensor    input1             = (vx_tensor)paramObj[1];
    vx_tensor    output             = (vx_tensor)paramObj[2];
    vx_enum      src0Format         = VX_TYPE_FLOAT16;
    vx_enum      src1Format         = VX_TYPE_FLOAT16;
    vx_enum      dstFormat          = VX_TYPE_FLOAT16;
    vx_enum      src0QuantType      = 0;
    vx_int8      src0FixPointPos    = 0;
    vx_enum      src1QuantType      = 0;
    vx_int8      src1FixPointPos    = 0;
    vx_enum      dstQuantType       = 0;
    vx_int8      dstFixPointPos     = 0;

    vx_uint32    coord_size[4]      = {0, 0, 0, 0};
    vx_uint32    input_size[4]      = {0, 0, 0, 0};
    vx_uint32    output_size[4]     = {0, 0, 0, 0};

    status = vxQueryTensor(input0, VX_TENSOR_DATA_TYPE, &src0Format, sizeof(src0Format));
    status |= vxQueryTensor(input0, VX_TENSOR_QUANT_FORMAT, &src0QuantType, sizeof(src0QuantType));
    status |= vxQueryTensor(input0, VX_TENSOR_FIXED_POINT_POSITION, &src0FixPointPos, sizeof(src0FixPointPos));
    status |= vxQueryTensor(input0, VX_TENSOR_DIMS, input_size, sizeof(input_size));

    status |= vxQueryTensor(input1, VX_TENSOR_DATA_TYPE, &src1Format, sizeof(src1Format));
    status |= vxQueryTensor(input1, VX_TENSOR_QUANT_FORMAT, &src1QuantType, sizeof(src1QuantType));
    status |= vxQueryTensor(input1, VX_TENSOR_FIXED_POINT_POSITION, &src1FixPointPos, sizeof(src1FixPointPos));
    status |= vxQueryTensor(input1, VX_TENSOR_DIMS, coord_size, sizeof(coord_size));
    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &dstFormat, sizeof(dstFormat));
    status |= vxQueryTensor(output, VX_TENSOR_QUANT_FORMAT, &dstQuantType, sizeof(dstQuantType));
    status |= vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POSITION, &dstFixPointPos, sizeof(dstFixPointPos));
    status |= vxQueryTensor(output, VX_TENSOR_DIMS, output_size, sizeof(output_size));

    if(status < 0)
        printf("error-%s,%d\n",__FILE__,__LINE__);

    if ((src0Format == VX_TYPE_FLOAT16 && src1Format == VX_TYPE_FLOAT16 && dstFormat == VX_TYPE_FLOAT16)
     || (src0Format == VX_TYPE_INT16 && src1Format == VX_TYPE_INT16 && dstFormat == VX_TYPE_INT16))
    {
        shaderParam.globalWorkScale[0]  = 2;
        shaderParam.globalWorkScale[1]  = 1;
    }

    shaderParam.globalWorkSize[0]   =
                gcmALIGN((coord_size[0] + shaderParam.globalWorkScale[0] - 1) / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   =
                (coord_size[1] + shaderParam.globalWorkScale[1] - 1) / shaderParam.globalWorkScale[1];
    {
        vx_int32 packedWH2[2]   = {input_size[0], input_size[1]};
        vx_int32 packedWH       = (input_size[1] << 16) | (input_size[0] & 0xFFFF);
        vx_uint32 uniGetDXY_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00100001, 0x00010010, // ABin
            0x09090909, // BSelt
            0x00010000, 0x00000001, // BBin
            0x00000101, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x00000000, 0x3c000000, 0x00000000, 0x3c000000, 0x00000000, 0x3c000000, 0x00000000 // Constant
        };
        vx_uint32 uniConvertF16toF32_4x4[16] = {
            0x01010101, // TCfg
            0x01010000, // ASelt
            0x00010000, 0x00010000, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };

        vxSetNodeUniform(nodObj, "uniGetDXY_4x4", 1, uniGetDXY_4x4);
        vxSetNodeUniform(nodObj, "uniConvertF16toF32_4x4", 1, uniConvertF16toF32_4x4);

        //packedWH2[0]   = input_size[0];
        //packedWH2[1]   = input_size[1];
        //packedWH       = (input_size[1] << 16) | (input_size[0] & 0xFFFF);
        vxSetNodeUniform(nodObj, "packedWH2", 1, packedWH2);
        vxSetNodeUniform(nodObj, "packedWH", 1, &packedWH);
    }
    if (output_size[2] > 1)
    {
        vxSetNodeUniform(nodObj, "depth", 1, &output_size[2]);
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                                &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return VX_SUCCESS;
}

#ifdef __cpluplus
extern "C" {
#endif
vx_kernel_description_t vxSpatial_transformer_CPU =
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

vx_kernel_description_t vxTransform_GemmKernelInfo_F16toF16 =
{
    VX_KERNEL_ENUM_SPATIAL_TRANSFORMER,
    VX_KERNEL_NAME_SPATIAL_TRANSFORMER,
    NULL,
    vxTransform_GemmKernelParam,
    (sizeof(vxTransform_GemmKernelParam) / sizeof(vxTransform_GemmKernelParam[0])),
    vxValidator,
    NULL,
    NULL,
    vxTransform_GemmInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTransform_setupThresKernelInfo_F16toF16 =
{
    VX_KERNEL_ENUM_SPATIAL_TRANSFORMER,
    VX_KERNEL_NAME_TRANSFORM_SETUP_THRES_F16TOF16,
    NULL,
    vxTransform_setupThresKernelParam,
    (sizeof(vxTransform_setupThresKernelParam) / sizeof(vxTransform_setupThresKernelParam[0])),
    vxValidator,
    NULL,
    NULL,
    vxTransform_setupThresInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTransform_InterPKernelInfo_F16toF16_2D =
{
    VX_KERNEL_ENUM_SPATIAL_TRANSFORMER,
    VX_KERNEL_NAME_TRANSFORM_INTERP_F16TOF16_2D,
    NULL,
    vxTransform_InterPKernelParam,
    (sizeof(vxTransform_InterPKernelParam) / sizeof(vxTransform_InterPKernelParam[0])),
    vxValidator,
    NULL,
    NULL,
    vxTransform_InterPInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTransform_InterPKernelInfo_F16toF16 =
{
    VX_KERNEL_ENUM_SPATIAL_TRANSFORMER,
    VX_KERNEL_NAME_TRANSFORM_INTERP_F16TOF16,
    NULL,
    vxTransform_InterPKernelParam,
    (sizeof(vxTransform_InterPKernelParam) / sizeof(vxTransform_InterPKernelParam[0])),
    vxValidator,
    NULL,
    NULL,
    vxTransform_InterPInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_SPATIAL_TRANSFORMER_list[] =
{
    &vxSpatial_transformer_CPU,
    &vxTransform_setupThresKernelInfo_F16toF16,
    &vxTransform_GemmKernelInfo_F16toF16,
    &vxTransform_InterPKernelInfo_F16toF16_2D,
    &vxTransform_InterPKernelInfo_F16toF16,
    NULL
};
#ifdef __cpluplus
}
#endif
