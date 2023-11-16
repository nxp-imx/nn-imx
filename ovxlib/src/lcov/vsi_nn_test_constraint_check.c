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
#include "vsi_nn_test.h"
#include "vsi_nn_pub.h"
#include "lcov/vsi_nn_test_constraint_check.h"
#define BUF_SIZE (64)

static vsi_status vsi_nn_test_is_const_tensor(void)
{
    VSILOGI("vsi_nn_test_is_const_tensor");
    vsi_nn_tensor_t *temp;
    temp = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    temp->attr.is_const = FALSE;
    const vsi_nn_tensor_t *tensor = temp;
    vsi_bool result;
    result = is_const_tensor(tensor);
    free(temp);
    temp = NULL;
    if(result == FALSE)
    {
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

static vsi_status test_print_op_io_types(void)
{
    char name_tmp[BUF_SIZE] = "test_print_op_io_types";
    const char* name = name_tmp;
    op_constraint_reg_type *reg;
    reg = (op_constraint_reg_type *)malloc(sizeof(op_constraint_reg_type));
    reg->reg_input_num = 1;
    reg->reg_output_num = 1;
    reg->io_types_item_size = 1;
    reg->io_types_item_count = 1;
    uint8_t type[2] = {2, 1};
    const void* types_tmp = type;
    reg->types = types_tmp;
    const op_constraint_reg_type* op_constraint_reg = reg;
    print_op_io_types(name, op_constraint_reg);
    free(reg);
    reg = NULL;
    return VSI_SUCCESS;
}

static vsi_status test_generate_op_io_types_desc(void)
{
    VSILOGI("test_generate_op_io_types_desc");
    vsi_nn_tensor_t** inputs;
    inputs = (vsi_nn_tensor_t **)malloc(sizeof(vsi_nn_tensor_t *));
    inputs[0] = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    inputs[0]->attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    inputs[0]->attr.dtype.vx_type = D_I4;
    int inputs_num = 1;
    vsi_nn_tensor_t** outputs;
    outputs = (vsi_nn_tensor_t **)malloc(sizeof(vsi_nn_tensor_t *));
    outputs[0] = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    int outputs_num = 0;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.vx_type = D_U4;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.vx_type = D_I8;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.vx_type = D_U8;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.vx_type = D_I16;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.vx_type = D_U16;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.vx_type = D_I32;
    inputs[0]->attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.vx_type = D_U32;
    inputs[0]->attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.vx_type = D_I64;

    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.vx_type = D_U64;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.vx_type = D_F16;
    inputs[0]->attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.vx_type = D_F64;
    inputs[0]->attr.dtype.qnt_type = VSI_NN_QNT_TYPE_PERCHANNEL_SYMMETRIC_FLOAT8;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.vx_type = D_BF16;
    inputs[0]->attr.dtype.qnt_type = VSI_NN_QNT_TYPE_SYMMETRIC_FLOAT8;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.qnt_type = 9;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    inputs[0]->attr.dtype.qnt_type = VSI_NN_QNT_TYPE_SYMMETRIC_FLOAT8;
    inputs[0]->attr.dtype.vx_type = 30;
    generate_op_io_types_desc(inputs, inputs_num, outputs, outputs_num);

    free(inputs[0]);
    inputs[0] = NULL;
    free(inputs);
    inputs = NULL;
    free(outputs[0]);
    outputs[0] = NULL;
    free(outputs);
    outputs = NULL;
    return VSI_SUCCESS;
}

static vsi_status vsi_nn_test_QuantAffinePerchannelCalParam(void)
{
    VSILOGI("vsi_nn_test_QuantAffinePerchannelCalParam");
    vsi_nn_type_e dtype = VSI_NN_TYPE_INT8;
    float max_data = 4.0;
    float min_data = 0;
    float scale[1] = {0};
    vsi_status result;
    result = vsi_nn_QuantAffinePerchannelCalParam(dtype, max_data, min_data, scale);
    if(result != 0) return VSI_FAILURE;

    dtype = VSI_NN_TYPE_INT16;
    result = vsi_nn_QuantAffinePerchannelCalParam(dtype, max_data, min_data, scale);
    if(result != -1) return VSI_FAILURE;
    return VSI_SUCCESS;
}

static vsi_status vsi_nn_test_QuantAffinePerchannelCheck(void)
{
    VSILOGI("vsi_nn_test_QuantAffinePerchannelCheck");
    vsi_nn_tensor_t *input;
    input = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    input->attr.dtype.vx_type = VSI_NN_TYPE_INT8;
    input->attr.dtype.scale = 1;

    float scale[2] = {1.0};
    vsi_nn_tensor_t *weight;
    weight = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    weight->attr.dtype.vx_type = VSI_NN_TYPE_INT8;
    weight->attr.dtype.scales = scale;
    weight->attr.dtype.scale_dim = 1;
    vsi_nn_tensor_t *bias;
    bias = (vsi_nn_tensor_t *)malloc(sizeof(vsi_nn_tensor_t));
    bias->attr.dtype.vx_type = VSI_NN_TYPE_INT8;
    bias->attr.dtype.scales = scale;
    vsi_bool result;
    result = vsi_nn_QuantAffinePerchannelCheck(input, weight, bias);
    if(result != 1) return VSI_FAILURE;

    input->attr.dtype.vx_type = VSI_NN_TYPE_INT16;
    result = vsi_nn_QuantAffinePerchannelCheck(input, weight, bias);
    if(result != 0) return VSI_FAILURE;
    free(input);
    input = NULL;
    free(weight);
    weight = NULL;
    free(bias);
    bias = NULL;
    return VSI_SUCCESS;
}

vsi_status vsi_nn_test_constraint_check( void )
{
    vsi_status status = VSI_FAILURE;

    status = vsi_nn_test_is_const_tensor();
    TEST_CHECK_STATUS(status, final);

    status = test_print_op_io_types();
    TEST_CHECK_STATUS(status, final);

    status = test_generate_op_io_types_desc();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_QuantAffinePerchannelCalParam();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_QuantAffinePerchannelCheck();
    TEST_CHECK_STATUS(status, final);

final:
    return status;
}
