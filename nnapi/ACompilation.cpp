/****************************************************************************
*
*    Copyright (c) 2005 - 2019 by Vivante Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Vivante Corporation. This is proprietary information owned by
*    Vivante Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Vivante Corporation.
*
*****************************************************************************/


#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <stdio.h>
#include <algorithm>
#include <VX/vx.h>
#include "math.h"
#include <VX/vxu.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>
#include "ACompilation.h"
#include "AModel.h"
#include "NeuralNetworks.h"

using namespace std;

#define IMPLICIT_PADDING_MAX_INPUT  7

static vx_tensor creatVirtualTensorFromOperand(vx_graph graph, AnnOperand &operand, int fixed_point_pos);

ACompilation::ACompilation(AModel *model):
        m_context(model->getVXContext()),
        m_model(model), m_lockFlag(false),
        m_compilationRefence(ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER)
{
    m_graph = vxCreateGraph(m_context);

    /*create virtual tensor for intermediate tensor that link ops*/
    std::vector<AnnOperand> * allOperand = m_model->getAllOperand();
    for(auto &operand : *allOperand)
    {
        if(operand.tensor == NULL && operand.type > 2)
            operand.tensor = creatVirtualTensorFromOperand(m_graph, operand, 0);
    }

    /* register operantion*/
    PUSH_OP(ADD);
    PUSH_OP(CONV_2D);
    PUSH_OP(RELU);
    PUSH_OP(RELU1);
    PUSH_OP(RELU6);
    PUSH_OP(FULLY_CONNECTED);
    PUSH_OP(SOFTMAX);
    PUSH_OP(MAX_POOL_2D);
    PUSH_OP(AVERAGE_POOL_2D);
    PUSH_OP(MUL);
    PUSH_OP(RESHAPE);
    PUSH_OP(L2_NORMALIZATION);
    PUSH_OP(LOGISTIC);
    PUSH_OP(TANH);
    PUSH_OP(FLOOR);
    PUSH_OP(RESIZE_BILINEAR);
    PUSH_OP(SPACE_TO_DEPTH);
    PUSH_OP(DEPTH_TO_SPACE);
    PUSH_OP(EMBEDDING_LOOKUP);
    PUSH_OP(RNN);
    PUSH_OP(LOCAL_RESPONSE_NORMALIZATION);
    PUSH_OP(HASHTABLE_LOOKUP);
    PUSH_OP(LSTM);
    PUSH_OP(SVDF);
    PUSH_OP(LSH_PROJECTION);
    PUSH_OP(L2_POOL_2D);
    PUSH_OP(DEQUANTIZE);
    PUSH_OP(DEPTHWISE_CONV_2D);
    PUSH_OP(CONCATENATION);

    /*NN API 1.1*/
    PUSH_OP(DIV);
    PUSH_OP(SUB);
    PUSH_OP(TRANSPOSE);
    PUSH_OP(MEAN);
    PUSH_OP(SQUEEZE);
    PUSH_OP(STRIDED_SLICE);
    PUSH_OP(SPACE_TO_BATCH_ND);
    PUSH_OP(BATCH_TO_SPACE_ND);
    PUSH_OP(PAD);

#ifdef NN_DEBUG
    cout << "new compilation" << endl;
#endif
}

ACompilation::~ACompilation()
{
    for(auto &tensor: m_tempTensor)
        if(tensor != NULL)
            vxReleaseTensor( &tensor);

    for(auto &node : m_nodes)
        if(node != NULL)
            vxReleaseNode(&node);

    if (m_graph != NULL)
    {
        // TODO:
        vxReleaseGraph(&m_graph);
    }

}

int ACompilation::setPreference(int32_t preference)
{
    if(m_lockFlag)
    {
        error_log("ANeuralNetworksCompilation cannot be modified\n");
    }

    // TODO

    return ANEURALNETWORKS_NO_ERROR;
}
int ACompilation::run()
{
    if(m_lockFlag)
    {
        error_log("ANeuralNetworksCompilation_finish has been called more than once\n");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    const std::vector<AnnOperation>* operations = m_model->getAllOperation();
    std::vector<AnnOperand> * oprands = m_model->getAllOperand();

#ifndef DUMP_NEURALNETWORK

#ifdef NN_DEBUG
    fprintf(stderr, "start compilation\n");
#endif
    for(size_t idx = 0; idx < operations->size(); idx++)
    {
        const AnnOperation &op = (*operations)[idx];
        if(m_opContainer.find(op.type) == m_opContainer.end())
        {
            error_log("operator type error");
        }
        else{
            NN_ERR_CHECK( m_opContainer[op.type](m_graph, m_model, oprands, op.OpInputs, op.OpOutputs, m_nodes, &m_tempTensor));
#ifdef NN_DEBUG
            fprintf(stderr, "add feature operation: %d done\n", op.type);
#endif
        }

    }

#ifdef NN_DEBUG
    cout << "ann api start verify graph"<< endl;
#endif

    vx_status status = vxVerifyGraph(m_graph);
    if (status != VX_SUCCESS)
    {
        cout << "fail to verify graph, error code: " <<status<< endl;
        assert(0);
    }
#endif
    m_lockFlag = true;
    return ANEURALNETWORKS_NO_ERROR;
}

inline int checkFusedCode(const std::vector<AnnOperand> *operands, const std::vector<uint32_t> inputs)
{
    if( (*operands)[inputs[inputs.size() -1]].scalar.i32 != ANEURALNETWORKS_FUSED_NONE)
    {
        error_log("could not support fusedcode");
        return ANEURALNETWORKS_OP_FAILED;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

inline int enumConvertFusedcode2relu(int type)
{
    switch (type){
        case ANEURALNETWORKS_FUSED_RELU:
            return VX_NN_ACTIVATION_RELU;
        case ANEURALNETWORKS_FUSED_RELU1:
            return VX_NN_ACTIVATION_RELU1;
        case ANEURALNETWORKS_FUSED_RELU6:
            return VX_NN_ACTIVATION_RELU6;
        default:
            return VX_NN_ACTIVATION_RELU;
            break;
    }
    return VX_NN_ACTIVATION_RELU;
}

/*    out_size = (input + stride - 1) / stride;
     *    needed_input = (out_size - 1) * stride + filter_size
     *    total_padding = max(0, needed_input - output_size)
     */
static void computePadding(uint32_t input, uint32_t output,uint32_t filter, uint32_t stride, uint32_t &padlt, uint32_t &padrb, int32_t padschme)
{
    if (ANEURALNETWORKS_PADDING_SAME == padschme)
    {
        int total_pad = (output - 1) * stride + filter - input;
        total_pad = max(0, total_pad);
        padlt = total_pad / 2;
        padrb = (total_pad + 1) / 2;
    }
    else
    {
        padlt = 0;
        padrb = 0;
    }
}

static vx_tensor createVirtualTenosrByParam(vx_graph graph, vx_uint32 dimNum, vx_uint32 *dims, int32_t type,
                                            vx_float32 scale = 0, vx_uint32 zp = 0, vx_int8 fps = 0)
{
    vx_enum dataType = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM == type ? VX_TYPE_UINT8 :
        (ANEURALNETWORKS_TENSOR_INT32 == type ? VX_TYPE_INT32 :
#if HIGH_PRECISION_COMPUTE
        VX_TYPE_FLOAT32
#else
        VX_TYPE_FLOAT16
#endif
        );

    vx_enum quant_format = ( (dataType == VX_TYPE_UINT8) ||  (dataType == VX_TYPE_INT32) )? VX_QUANT_AFFINE_SCALE : 0;
    vx_tensor_create_params_t param = { dimNum, dims, dataType, quant_format, {{0}}};
    if(quant_format == VX_QUANT_AFFINE_SCALE)
    {
        param.quant_data.affine.scale     = scale;
        param.quant_data.affine.zeroPoint = zp;
    }
    else
    {
        param.quant_data.dfp.fixed_point_pos = fps;
    }
    vx_tensor tensor = vxCreateVirtualTensor2(graph, &param, sizeof(vx_tensor_create_params_t));

    vx_enum precision = VX_TENSOR_PRECISION_AUTO;
    VX_ERR_CHECK( vxSetTensorAttribute(tensor, VX_TENSOR_PRECISION, &precision, sizeof(precision)) );

    return tensor;
}

static vx_tensor creatVirtualTensorFromOperand(vx_graph graph, AnnOperand &operand, int fixed_point_pos = 0)
{
    vx_tensor tensor = NULL;
    vx_uint32 whcnDim[4] = {1,1,1,1};
    if(operand.dimensionCount == 2)
    {
        whcnDim[0] = operand.dimensions[1];
        whcnDim[1] = operand.dimensions[0];
    }
    else if(operand.dimensionCount == 3)
    {
        whcnDim[0] = operand.dimensions[2];
        whcnDim[1] = operand.dimensions[1];
        whcnDim[2] = operand.dimensions[0];
        whcnDim[3] = 1;
    }
    else if(operand.dimensionCount == 4)
    {
        whcnDim[0] = operand.dimensions[2];
        whcnDim[1] = operand.dimensions[1];
        whcnDim[2] = operand.dimensions[3];
        whcnDim[3] = operand.dimensions[0];
    }

    if(!operand.isEmpty)
    {
        if(2 < operand.type)
        {
            tensor = createVirtualTenosrByParam(graph, operand.dimensionCount, whcnDim, operand.type, operand.scale, operand.zeroPoint);
        }
        if (tensor == NULL)
        {
            error_log("creat tensor error");
        }
    }

    return tensor;
}

vx_tensor createTmpTensor(vx_graph graph,AnnOperand &output, int32_t fuseCode,std::vector<vx_tensor> *tmpTensors)
{
    //uint32_t fuseCode = (*operands)[inputs[inputs.size() - 1]].scalar.i32;
    vx_tensor convTensor = output.tensor;
    if( fuseCode != ANEURALNETWORKS_FUSED_NONE)
    {
        uint32_t size = output.dimensions[0];
        for(uint32_t i = 1; i < output.dimensionCount; i ++)
            size *= output.dimensions[i];
        convTensor = creatVirtualTensorFromOperand(graph, output);
        tmpTensors->push_back(convTensor);
    }

    return convTensor;
}

int addActiveOp(vx_graph graph, vx_tensor input, vx_tensor output, uint32_t fuseCode, std::vector<vx_node> &nodeContainer)
{
    if( fuseCode != ANEURALNETWORKS_FUSED_NONE)
    {
        VX_NODE_CHECK( vxActivationLayer( graph, input, enumConvertFusedcode2relu(fuseCode) , 0, 0, output) );
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int checkIndex(std::vector<uint32_t> inputIndx, std::vector<uint32_t> outputIndx,
               const size_t operandSize, const uint32_t inputSize, const uint32_t outputSize, const std::string functionName)
{
    if(inputIndx.size() != inputSize || outputSize != outputIndx.size())
    {
#ifdef NN_DEBUG
        printf("input size: %lu, output size: %lu, oprands : %lu\n", inputIndx.size(), outputIndx.size(), operandSize);
#endif
        error_log(functionName + ": parameter numbers error");
        return ANEURALNETWORKS_OP_FAILED;
    }
    for(auto idx : inputIndx)
    {
        if(idx > operandSize - 1)
        {
            error_log(functionName + ": inputs index out of range");
            return ANEURALNETWORKS_OP_FAILED;
        }
    }

    for(auto idx : outputIndx)
    {
        if(idx > operandSize - 1)
        {
            error_log(functionName + ": outputs index out of range");
            return ANEURALNETWORKS_OP_FAILED;
        }
    }

    return ANEURALNETWORKS_NO_ERROR;
}

static void convertTensor(AnnOperand& operand, vx_context vx_Context, vx_uint32 input_dimensionCount)
{
    vx_int32 axisDims_org[4] = { 0 };
    vx_int32 axisDims[4] = { 0 };

    vxcMemcpy(vx_Context, operand.tensor, axisDims_org, VX_READ_ONLY);

    vx_int32 converts[][4] = {
        { 0 },
        { 1, 0 },
        { 2, 1, 0 },
        { 2, 1, 3, 0 },
    };

    for (vx_uint32 i = 0; i < operand.dimensions[0]; i++)
    {
        axisDims[i] = axisDims_org[converts[input_dimensionCount - 1][i]];
    }

    vxcMemcpy(vx_Context, operand.tensor, axisDims, VX_WRITE_ONLY);
}

static void convertScalar(vx_int32 *scalar_org, vx_uint32 input_dimensionCount)
{
    vx_int32 scalar = *scalar_org;
    if (input_dimensionCount == 2)
    {
        scalar = ((*scalar_org & 1) << 1) | (*scalar_org >> 1);
    }
    else if (input_dimensionCount == 3)
    {
        scalar = ((*scalar_org & 1) << 2) | (*scalar_org & 2) | (*scalar_org >> 2);
    }
    else if (input_dimensionCount == 4)
    {
        scalar = ((*scalar_org & 4) << 1) | ((*scalar_org & 1) << 2) | (*scalar_org & 2) | (*scalar_org >> 3);
    }

    *scalar_org = scalar;
}

static void Convert_tensor_for_space2batch(int dimCount, vx_graph graph, vx_tensor tensor)
{
    vx_int32 tensor_value[4] = { 0 }, temp = 0;

    vxcMemcpy(vxGetContext((vx_reference)graph), tensor, tensor_value, VX_READ_ONLY);
    if (dimCount == 1)
    {
        temp = tensor_value[0];
        tensor_value[0] = tensor_value[1];
        tensor_value[1] = temp;
    }
    else if (dimCount == 2)
    {
        temp = tensor_value[0];
        tensor_value[0] = tensor_value[2];
        tensor_value[2] = temp;
        temp = tensor_value[1];
        tensor_value[1] = tensor_value[3];
        tensor_value[3] = temp;
    }

    vxcMemcpy(vxGetContext((vx_reference)graph), tensor, tensor_value, VX_WRITE_ONLY);
}

int ACompilation::addOperation_L2_NORMALIZATION(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                                std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 1, 1,"ANEURALNETWORKS_L2_NORMALIZATION") );

    AnnOperand& input =  (*operands)[inputs[0]];
    AnnOperand& output =  (*operands)[outputs[0]];

    CHECK_PARAMETER(input.dimensionCount == 4, INPUT, 0);
    CHECK_PARAMETER(output.dimensionCount == 4, OUTPUT, 0);

    VX_NODE_CHECK(vxL2NormalizeLayer( graph,  input.tensor, output.tensor));

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_RESHAPE(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                       std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 2, 1,"ANEURALNETWORKS_RESHAPE") );

    AnnOperand& input0 = (*operands)[inputs[0]];
    AnnOperand& input1 = (*operands)[inputs[1]];
    AnnOperand& output = (*operands)[outputs[0]];

    CHECK_PARAMETER(input0.type > 2,INPUT, 0);
    CHECK_PARAMETER(input1.type == ANEURALNETWORKS_TENSOR_INT32 && input1.dimensionCount == 1, INPUT, 1);

    /*viv:
        in driver stream, the tensor is whcn,
        so that it need to be converted to be cwhn ,flattened and coverted to whcn.
    */
    vx_uint32 perm_whcn2cwhn[] = {2, 0, 1, 3};
    vx_uint32 perm_cwhn2whcn[] = {1, 2, 0, 3};
    vx_tensor tfTensorIn = input0.tensor;
    vx_tensor tfTensorOut = output.tensor;
    if(input0.dimensionCount == 4 && input0.dimensions[3] > 1 && input0.dimensions[1] * input0.dimensions[2] > 1)
    {
        vx_uint32 cwhnFromNhwc[] = {input0.dimensions[3], input0.dimensions[2],input0.dimensions[1],input0.dimensions[0]};
        tfTensorIn = createVirtualTenosrByParam(graph, 4,cwhnFromNhwc, input0.type, input0.scale, input0.zeroPoint);
        VX_NODE_CHECK( vxTensorPermuteNode(graph, input0.tensor, tfTensorIn, perm_whcn2cwhn, 4) );
    }

    if(output.dimensionCount == 4 && output.dimensions[3] > 1 && output.dimensions[1] * output.dimensions[2] > 1){
        vx_uint32 whcnFromNhwc[] = {output.dimensions[2], output.dimensions[1], output.dimensions[3], output.dimensions[0]};
        tfTensorOut = createVirtualTenosrByParam(graph, 4, whcnFromNhwc, output.type, output.scale, output.zeroPoint);
    }

    vx_nn_reshape_params_t p = {input1.tensor};
    VX_NODE_CHECK( vxTensorReshapeNode(graph, tfTensorIn, &p, sizeof(p), tfTensorOut) );

    if(output.dimensionCount == 4 && output.dimensions[3] > 1 && output.dimensions[1] * output.dimensions[2] > 1){
        VX_NODE_CHECK( vxTensorPermuteNode(graph, tfTensorOut, output.tensor, perm_cwhn2whcn, 4) );
    }

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_MUL(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                   std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 3, 1,"ANEURALNETWORKS_MUL") );

    AnnOperand& operand0 = (*operands)[inputs[0]];
    AnnOperand& operand1 = (*operands)[inputs[1]];
    AnnOperand& output = (*operands)[outputs[0]];
    CHECK_PARAMETER(operand0.type == operand1.type, INPUT, 1);
    CHECK_PARAMETER(operand0.type == output.type, OUTPUT, 0);

    vx_float32 scale = 1; //????
    vx_scalar scale_s;
    scale_s = vxCreateScalar(model->getVXContext(), VX_TYPE_FLOAT32, &scale);
    if(!scale_s)
    {
        error_log("ANEURALNETWORKS_MUL CreateScalar fail\n");
        return ANEURALNETWORKS_OP_FAILED;
    }

    uint32_t fuseCode = (*operands)[inputs[inputs.size() - 1]].scalar.i32;
    vx_tensor convTensor = createTmpTensor(graph, output, fuseCode, tmpTensors);

    VX_NODE_CHECK( vxTensorMultiplyNode( graph,
                                 operand0.tensor,
                                 operand1.tensor,
                                 scale_s,
                                 VX_CONVERT_POLICY_WRAP,
                                 VX_ROUND_POLICY_TO_ZERO,
                                         convTensor ) );

    NN_ERR_CHECK( addActiveOp( graph, convTensor, output.tensor, fuseCode, nodeContainer) );

    return ANEURALNETWORKS_NO_ERROR;
}

int operationPooling(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> &inputs, std::vector<uint32_t> &outputs,
                     std::vector<vx_tensor> *tmpTensors, std::vector<vx_node> &nodeContainer, int32_t poolingType)
{
    if( (inputs.size() != 7 && inputs.size() != 10) || outputs.size() != 1)
        error_log("POOLING: input/output parameter number wrong");

    AnnOperand& inputOp = (*operands)[inputs[0]];
    AnnOperand& outputOp = (*operands)[outputs[0]];
    CHECK_PARAMETER( inputOp.dimensionCount == 4, INPUT, 0);
    CHECK_PARAMETER( outputOp.dimensionCount == 4, OUTPUT, 0);

    for( uint32_t idx = 1; idx < inputs.size(); idx++) {
        CHECK_PARAMETER( (*operands)[ inputs[idx]].type == ANEURALNETWORKS_INT32, INPUT, idx);
    }

    vx_uint32 padding_left = 0, padding_right = 0;
    vx_uint32 padding_top = 0, padding_bottom = 0;
    vx_uint32 stride_width = 0, stride_height = 0;
    vx_uint32 filter_width = 0, filter_height = 0;
    //vx_uint32 activation;

    if(inputs.size() == 7)
    {
        int32_t padschme = (*operands)[inputs[1]].scalar.i32;
        stride_width = (*operands)[inputs[2]].scalar.i32;
        stride_height = (*operands)[inputs[3]].scalar.i32;
        filter_width = (*operands)[inputs[4]].scalar.i32;
        filter_height = (*operands)[inputs[5]].scalar.i32;

        if(ANEURALNETWORKS_PADDING_SAME == (*operands)[inputs[1]].scalar.i32)
        {
            //TODO: need to fix
            computePadding(inputOp.dimensions[1], outputOp.dimensions[1], filter_height, stride_height, padding_top, padding_bottom, padschme);
            computePadding(inputOp.dimensions[2], outputOp.dimensions[2], filter_width, stride_width, padding_left, padding_right, padschme);
        }
        else if (ANEURALNETWORKS_PADDING_VALID == (*operands)[inputs[1]].scalar.i32)
        {
            padding_left = 0;
            padding_right = 0;
            padding_top = 0;
            padding_bottom = 0;
        } else
        {
            error_log("POOLing: input paddingCode error for implict padding mode");
        }
    }
    else if(inputs.size() == 10)
    {
        padding_left = (*operands)[inputs[1]].scalar.i32;
        padding_right = (*operands)[inputs[2]].scalar.i32;
        padding_top= (*operands)[inputs[3]].scalar.i32;
        padding_bottom = (*operands)[inputs[4]].scalar.i32;
        filter_width = (*operands)[inputs[7]].scalar.i32;
        filter_height = (*operands)[inputs[8]].scalar.i32;
        stride_width = (*operands)[inputs[5]].scalar.i32;
        stride_height = (*operands)[inputs[6]].scalar.i32;

        // check height and width of output
        CHECK_PARAMETER(outputOp.dimensions[2] == (inputOp.dimensions[2] + padding_right + padding_left - filter_width) / stride_width + 1, OUTPUT, 0 );
        CHECK_PARAMETER(outputOp.dimensions[1] == (inputOp.dimensions[1] + padding_top +  padding_bottom - filter_height) / stride_height + 1, OUTPUT, 0 );
    }

    vx_nn_pooling_params_t p = { poolingType, filter_width, filter_height, (vx_uint32)padding_left, (vx_uint32)padding_right,
                    (vx_uint32)padding_top, (vx_uint32)padding_bottom, /*stride_width, stride_height,*/ VX_NN_DS_SIZE_ROUNDING_FLOOR };

#ifdef NN_DEBUG_DETAIL_INFO
    printf("output:nhwc:(%d,%d,%d,%d); kernelsize:%d,%d; pad:%d,%d,%d,%d\n", outputOp.dimensions[0],
           outputOp.dimensions[1], outputOp.dimensions[2], outputOp.dimensions[3],p.pool_size_x, p.pool_size_y,
           p.pool_pad_x_left, p.pool_pad_x_right, p.pool_pad_y_top, p.pool_pad_y_bottom );
#endif

    int32_t fuseCode = (*operands)[inputs[inputs.size() - 1]].scalar.i32;
    vx_tensor convTensor = createTmpTensor(graph, (*operands)[outputs[0]], fuseCode, tmpTensors);
    VX_NODE_CHECK( vxPoolingLayer2(graph,
                                   (*operands)[inputs[0]].tensor,
                                   (const vx_nn_pooling_params_t*)&p,
                                   sizeof(p),
                                   convTensor) );

    NN_ERR_CHECK( addActiveOp( graph, convTensor, outputOp.tensor, fuseCode, nodeContainer) );
    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_AVERAGE_POOL_2D(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                               std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( operationPooling(graph, model, operands, inputs, outputs, tmpTensors, nodeContainer, VX_NN_POOLING_AVG_ANDROID) );
    return ANEURALNETWORKS_NO_ERROR;
}
int ACompilation::addOperation_MAX_POOL_2D(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                           std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( operationPooling(graph, model, operands, inputs, outputs, tmpTensors, nodeContainer, VX_NN_POOLING_MAX) );
    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_L2_POOL_2D(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                          std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                          std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( operationPooling(graph, model, operands, inputs, outputs, tmpTensors, nodeContainer, VX_NN_POOLING_L2) );
    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_SOFTMAX(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                       std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 2, 1,"ANEURALNETWORKS_SOFTMAX") );

    //input
    AnnOperand& inputOp = (*operands)[inputs[0]];

    //output
    AnnOperand& outputOp = (*operands)[outputs[0]];

    CHECK_PARAMETER(inputOp.dimensionCount == 4 || inputOp.dimensionCount == 2, INPUT, 0);
    CHECK_PARAMETER((*operands)[inputs[1]].type == ANEURALNETWORKS_FLOAT32, INPUT, 1);
    for(size_t i = 0; i < inputOp.dimensionCount; i++)
    {
        CHECK_PARAMETER(outputOp.dimensions[i] == inputOp.dimensions[i], OUTPUT, 0);
    }

    vx_nn_softmax_params_t p;
    p.beta = (*operands)[inputs[1]].scalar.fp32;
    VX_NODE_CHECK( vxSoftmaxLayer2(
            graph,
            inputOp.tensor,
            &p,
            sizeof(vx_nn_softmax_params_t),
            outputOp.tensor
    ));

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_FULLY_CONNECTED(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                               std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 4, 1,"ANEURALNETWORKS_FULLCONNECT") );

    AnnOperand& inputOp = (*operands)[inputs[0]];
    AnnOperand& weightOp = (*operands)[inputs[1]];
    AnnOperand& biasOp = (*operands)[inputs[2]];
    AnnOperand& outputOp = (*operands)[outputs[0]];

    CHECK_PARAMETER(weightOp.dimensionCount == 2, INPUT, 1);
    CHECK_PARAMETER(biasOp.dimensionCount == 1, INPUT, 2);
    CHECK_PARAMETER((*operands)[inputs[3]].type == ANEURALNETWORKS_INT32, INPUT, 3);
    CHECK_PARAMETER(outputOp.dimensionCount = 2, OUTPUT, 0);

    vx_enum precision = VX_TENSOR_PRECISION_HIGH;
    vxSetTensorAttribute((vx_tensor)biasOp.tensor, VX_TENSOR_PRECISION, &precision, sizeof(vx_enum));
    biasOp.tensorAttribute.precision = VX_TENSOR_PRECISION_HIGH;

    convertTensorDataFromFp322Fp16(vxGetContext((vx_reference)graph), weightOp);
    convertRankAndFormat(vxGetContext((vx_reference)graph), biasOp, true);

    int32_t fuseCode = (*operands)[inputs[3]].scalar.i32;
    vx_tensor convTensor = createTmpTensor(graph, outputOp, fuseCode, tmpTensors);

    VX_NODE_CHECK( vxFullyConnectedLayer(graph,
        inputOp.tensor,
        weightOp.tensor,
        biasOp.tensor,
        /*0, 0, */
        VX_CONVERT_POLICY_SATURATE,
        VX_ROUND_POLICY_TO_ZERO,
        /*VX_NN_DS_SIZE_ROUNDING_FLOOR, */
        convTensor) );

    NN_ERR_CHECK( addActiveOp(graph, convTensor, outputOp.tensor, fuseCode, nodeContainer) );


    return ANEURALNETWORKS_NO_ERROR;
}

int addOperationActive(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                       std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                       int32_t activeType, std::string nnType, std::vector<vx_node> &nodeContainer, int a = 0, int b = 0)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 1, 1, nnType) );

    CHECK_PARAMETER( (*operands)[inputs[0]].type > 2, INPUT, 0);
    CHECK_PARAMETER( (*operands)[outputs[0]].type > 2, OUTPUT, 0);

    VX_NODE_CHECK( vxActivationLayer(
            graph,
            (*operands)[inputs[0]].tensor,
            activeType,
            (vx_float32)a,
            (vx_float32)b,
            (*operands)[outputs[0]].tensor
    ));
    return  ANEURALNETWORKS_NO_ERROR;
}
int ACompilation::addOperation_TANH(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                    std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK(  addOperationActive(graph, model, operands, inputs, outputs, VX_NN_ACTIVATION_HYPERBOLIC_TAN, "ANEURALNETWORKS_TANH", nodeContainer, 1 , 1) );
    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_LOGISTIC(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                        std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK(  addOperationActive(graph, model, operands, inputs, outputs, VX_NN_ACTIVATION_LOGISTIC, "ANEURALNETWORKS_LOGISTIC", nodeContainer) );
    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_RELU(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                     std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK(  addOperationActive(graph, model, operands, inputs, outputs, VX_NN_ACTIVATION_RELU, "ANEURALNETWORKS_RELU", nodeContainer) );
    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_RELU1(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                     std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK(  addOperationActive(graph, model, operands, inputs, outputs, VX_NN_ACTIVATION_RELU1, "ANEURALNETWORKS_RELU1", nodeContainer) );
    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_RELU6(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                     std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( addOperationActive(graph, model, operands, inputs, outputs, VX_NN_ACTIVATION_RELU6, "ANEURALNETWORKS_RELU6", nodeContainer) );
    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_CONV_2D(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                       std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    vx_int32   pad_const_val = 0;
    vx_scalar  pad_const = NULL;

    size_t inputSize = inputs.size();
    size_t outputSize = outputs.size();
    if( (inputSize != 7 && inputSize != 10) || outputSize != 1)
        error_log("addOperation_CONV_2D: input/output parameter number wrong");

    AnnOperand& inputOp = (*operands)[inputs[0]];
    AnnOperand& weightOp = (*operands)[inputs[1]];
    AnnOperand& biasOp = (*operands)[inputs[2]];
    AnnOperand& outputOp = (*operands)[outputs[0]];

    CHECK_PARAMETER(inputOp.dimensionCount == 4, INPUT, 0);
    CHECK_PARAMETER(weightOp.dimensionCount == 4, INPUT, 1);
    CHECK_PARAMETER(biasOp.dimensionCount == 1, INPUT, 2);
    for(size_t i = 3; i < inputs.size(); i++) {
        CHECK_PARAMETER((*operands)[inputs[i]].type == ANEURALNETWORKS_INT32, INPUT, i);
    }

    pad_const = vxCreateScalar(model->getVXContext(), VX_TYPE_INT32, &pad_const_val);
    if (!pad_const)
    {
        error_log("vxCreateScalar failure! at line");
    }

    uint32_t  pad_x_left = 0, pad_x_right = 0;
    uint32_t  pad_y_top = 0, pad_y_bottom = 0;
    int32_t stride_x, stride_y;
    if (inputSize == 10)
    {
        pad_x_left = (*operands)[inputs[3]].scalar.i32;
        pad_x_right = (*operands)[inputs[4]].scalar.i32;
        pad_y_top = (*operands)[inputs[5]].scalar.i32;
        pad_y_bottom = (*operands)[inputs[6]].scalar.i32;
        stride_x = (*operands)[inputs[7]].scalar.i32;
        stride_y = (*operands)[inputs[8]].scalar.i32;
    }
    else
    {
        int32_t padschme = (*operands)[inputs[3]].scalar.i32;
        stride_x = (*operands)[inputs[4]].scalar.i32;
        stride_y = (*operands)[inputs[5]].scalar.i32;

        computePadding(inputOp.dimensions[1], outputOp.dimensions[1], weightOp.dimensions[1], stride_y, pad_y_top, pad_y_bottom, padschme);
        computePadding(inputOp.dimensions[2], outputOp.dimensions[2], weightOp.dimensions[2], stride_x, pad_x_left, pad_x_right, padschme);
    }
#ifdef NN_DEBUG_DETAIL_INFO
    printf("output:nhwc:(%d,%d,%d,%d); pad:%d,%d,%d,%d\n", outputOp.dimensions[0],
           outputOp.dimensions[1], outputOp.dimensions[2], outputOp.dimensions[3],
           pad_x_left, pad_x_right, pad_y_top, pad_y_bottom );
#endif

    uint32_t fuseCode = (*operands)[inputs[inputs.size() - 1]].scalar.i32;
    vx_tensor convTensor = outputOp.tensor;
    int32_t depth_multiplier = 0;

    vx_enum precision = VX_TENSOR_PRECISION_HIGH;
    vxSetTensorAttribute((vx_tensor)biasOp.tensor, VX_TENSOR_PRECISION, &precision, sizeof(vx_enum));
    biasOp.tensorAttribute.precision = VX_TENSOR_PRECISION_HIGH;

    CONVERT_RANK_FORMAT();

    vx_uint32 inputDims[4], outputDims[4];
    convertDims(inputDims, inputOp.dimensions, inputOp.dimensionCount);
    convertDims(outputDims, outputOp.dimensions, outputOp.dimensionCount);

    {
        convTensor= createTmpTensor(graph, outputOp, fuseCode, tmpTensors);

        vx_nn_convolution_params_ext2_t params = {
            {
                {
                    (vx_size)pad_x_left, (vx_size)pad_y_top, VX_CONVERT_POLICY_SATURATE,  VX_ROUND_POLICY_TO_ZERO, VX_NN_DS_SIZE_ROUNDING_FLOOR, 0, 0
                },
                (vx_size)pad_x_right, (vx_size)pad_y_bottom, VX_PAD_CONSTANT, 0
            }, (vx_uint32)stride_x, (vx_uint32)stride_y, (vx_int32)depth_multiplier
        };

        VX_NODE_CHECK( vxConvolutionLayer(graph, inputOp.tensor, weightOp.tensor, biasOp.tensor,
                                            (const vx_nn_convolution_params_t *)&params, sizeof(params),convTensor));
        NN_ERR_CHECK( addActiveOp(graph, convTensor, outputOp.tensor, fuseCode, nodeContainer) );
    }

    VX_ERR_CHECK( vxReleaseScalar(&pad_const) );

    return ANEURALNETWORKS_NO_ERROR;
}

 int  ACompilation::addOperation_ADD(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                     std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(),3, 1, "ANEURALNETWORKS_ADD") );

    AnnOperand& operand0 = (*operands)[inputs[0]];
    AnnOperand& operand1 = (*operands)[inputs[1]];
    AnnOperand& operand2 = (*operands)[inputs[2]];
    AnnOperand& output = (*operands)[outputs[0]];

    CHECK_PARAMETER(operand0.type >2 && operand0.dimensionCount <= 4, INPUT, 0);
    CHECK_PARAMETER(operand0.type == operand1.type && (operand1.dimensionCount <= 4), INPUT, 1);
    CHECK_PARAMETER(operand2.type == ANEURALNETWORKS_INT32, INPUT, 2);
    CHECK_PARAMETER(operand0.type == output.type, OUTPUT, 0);

    uint32_t fuseCode = (*operands)[inputs[inputs.size() - 1]].scalar.i32;
    vx_tensor convTensor = createTmpTensor(graph, output, fuseCode, tmpTensors);

    VX_NODE_CHECK( vxTensorAddNode(graph, operand0.tensor, operand1.tensor, VX_CONVERT_POLICY_WRAP, convTensor) );
    NN_ERR_CHECK( addActiveOp( graph, convTensor, output.tensor, fuseCode, nodeContainer) );

    return ANEURALNETWORKS_NO_ERROR;
}

int  ACompilation::addOperation_MEAN(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
    std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK(checkIndex(inputs, outputs, operands->size(), 3, 1, "ANEURALNETWORKS_MEAN"));

    AnnOperand& operand0 = (*operands)[inputs[0]];
    AnnOperand& operand1 = (*operands)[inputs[1]];
    AnnOperand& operand2 = (*operands)[inputs[2]];
    AnnOperand& output = (*operands)[outputs[0]];
    vx_nn_mean_params_t params = { operand1.tensor, operand2.scalar.i32 };

    CHECK_PARAMETER(operand0.dimensionCount <= 4, INPUT, 0);
    CHECK_PARAMETER((operand1.type == ANEURALNETWORKS_TENSOR_INT32) && (operand1.dimensionCount == 1), INPUT, 1);
    CHECK_PARAMETER(operand2.type == ANEURALNETWORKS_INT32, INPUT, 2);
    CHECK_PARAMETER(operand0.type == output.type, OUTPUT, 0);

    vx_int32 axisDims_org[4] = {0};
    vx_int32 axisDims[4] = { 0 };

    vxcMemcpy(model->getVXContext(), operand1.tensor, axisDims_org, VX_READ_ONLY);

    vx_int32 converts[][4] = {
        { 0 },
        { 1, 0 },
        { 2, 1, 0 },
        { 3, 1, 0, 2},
    };

    for (vx_uint32 i = 0; i < operand1.dimensions[0]; i ++)
    {
        vx_int32 index = (axisDims_org[i] < 0) ? (axisDims_org[i] + operand0.dimensionCount) : axisDims_org[i];
        axisDims[i] = converts[operand0.dimensionCount - 1][index];
    }

    vxcMemcpy(model->getVXContext(), operand1.tensor, axisDims, VX_WRITE_ONLY);

    VX_NODE_CHECK(vxTensorMeanNode(graph, operand0.tensor, &params, sizeof(vx_nn_mean_params_t), output.tensor));

    return ANEURALNETWORKS_NO_ERROR;
}

int  ACompilation::addOperation_SQUEEZE(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
    std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK(checkIndex(inputs, outputs, operands->size(), 2, 1, "ANEURALNETWORKS_SQUEEZE"));

    AnnOperand& operand0 = (*operands)[inputs[0]];
    AnnOperand& operand1 = (*operands)[inputs[1]];
    AnnOperand& output = (*operands)[outputs[0]];
    vx_nn_squeeze_params_t params = { operand1.tensor};

    CHECK_PARAMETER(operand0.dimensionCount <= 4, INPUT, 0);
    CHECK_PARAMETER((operand1.type == ANEURALNETWORKS_TENSOR_INT32) && (operand1.dimensionCount == 1), INPUT, 1);
    CHECK_PARAMETER(operand0.type == output.type, OUTPUT, 0);

    VX_NODE_CHECK(vxTensorSqueezeNode(graph, operand0.tensor, &params, sizeof(vx_nn_squeeze_params_t), output.tensor));

    return ANEURALNETWORKS_NO_ERROR;
}

int  ACompilation::addOperation_STRIDED_SLICE(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
    std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK(checkIndex(inputs, outputs, operands->size(), 7, 1, "ANEURALNETWORKS_STRIDED_SLICE"));

    AnnOperand& operand0 = (*operands)[inputs[0]];
    AnnOperand& operand1 = (*operands)[inputs[1]];
    AnnOperand& operand2 = (*operands)[inputs[2]];
    AnnOperand& operand3 = (*operands)[inputs[3]];
    AnnOperand& operand4 = (*operands)[inputs[4]];
    AnnOperand& operand5 = (*operands)[inputs[5]];
    AnnOperand& operand6 = (*operands)[inputs[6]];
    AnnOperand& output = (*operands)[outputs[0]];

    CHECK_PARAMETER(operand0.dimensionCount <= 4, INPUT, 0);
    CHECK_PARAMETER((operand1.type == ANEURALNETWORKS_TENSOR_INT32) && (operand1.dimensionCount == 1), INPUT, 1);
    CHECK_PARAMETER((operand2.type == ANEURALNETWORKS_TENSOR_INT32) && (operand2.dimensionCount == 1), INPUT, 2);
    CHECK_PARAMETER((operand3.type == ANEURALNETWORKS_TENSOR_INT32) && (operand3.dimensionCount == 1), INPUT, 3);
    CHECK_PARAMETER(operand4.type == ANEURALNETWORKS_INT32, INPUT, 4);
    CHECK_PARAMETER(operand5.type == ANEURALNETWORKS_INT32, INPUT, 5);
    CHECK_PARAMETER(operand6.type == ANEURALNETWORKS_INT32, INPUT, 6);
    CHECK_PARAMETER(operand0.type == output.type, OUTPUT, 0);

    convertTensor(operand1, model->getVXContext(), operand0.dimensionCount);
    convertTensor(operand2, model->getVXContext(), operand0.dimensionCount);
    convertTensor(operand3, model->getVXContext(), operand0.dimensionCount);

    convertScalar(&operand4.scalar.i32, operand0.dimensionCount);
    convertScalar(&operand5.scalar.i32, operand0.dimensionCount);
    convertScalar(&operand6.scalar.i32, operand0.dimensionCount);

    vx_nn_stride_slice_params_t params = { operand1.tensor, operand2.tensor, operand3.tensor, operand4.scalar.i32, operand5.scalar.i32, operand6.scalar.i32 };
    VX_NODE_CHECK(vxTensorStrideSliceNode(graph, operand0.tensor, &params, sizeof(vx_nn_stride_slice_params_t), output.tensor));

    return ANEURALNETWORKS_NO_ERROR;
}


int ACompilation::addOperation_FLOOR(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                        std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(),1, 1, "ANEURALNETWORKS_FLOOR") );

    AnnOperand& input = (*operands)[inputs[0]];
    AnnOperand& output =(*operands)[outputs[0]];

    CHECK_PARAMETER( input.type == output.type, OUTPUT, 0);
    CHECK_PARAMETER( input.dimensionCount== output.dimensionCount, OUTPUT, 0);
    for(size_t i = 0; i < input.dimensionCount;i ++)
        CHECK_PARAMETER(input.dimensions[i] == output.dimensions[i], OUTPUT, 0);

    vx_nn_rounding_params_t p;
    p.mode = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    VX_NODE_CHECK( vxTensorRoundingNode(graph, input.tensor, &p, sizeof(p), output.tensor) );

#ifdef NN_DEBUG
    fprintf(stderr, "and operation: ANEURALNETWORKS_FLOOR done\n");
#endif
    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_RESIZE_BILINEAR(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                               std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                               std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(),3, 1, "ANEURALNETWORKS_RESIZE_BILINEAR") );

    AnnOperand& input0 = (*operands)[inputs[0]];
    AnnOperand& input1 = (*operands)[inputs[1]];
    AnnOperand& input2 = (*operands)[inputs[2]];
    AnnOperand& output = (*operands)[outputs[0]];

    CHECK_PARAMETER(input0.dimensionCount == 4 && input0.type == ANEURALNETWORKS_TENSOR_FLOAT32, INPUT, 0);
    CHECK_PARAMETER(input1.type == ANEURALNETWORKS_INT32, INPUT, 1);
    CHECK_PARAMETER(input2.type == ANEURALNETWORKS_INT32, INPUT, 2);
    CHECK_PARAMETER( (vx_int32)output.dimensions[1] == input1.scalar.i32, OUTPUT,0);
    CHECK_PARAMETER( (vx_int32)output.dimensions[2] == input2.scalar.i32, OUTPUT,0);

    vx_nn_scale_params_t p;
    p.type = VX_INTERPOLATION_BILINEAR;
    VX_NODE_CHECK( vxTensorScaleNode(graph, input0.tensor, &p, sizeof(p), output.tensor));

    return ANEURALNETWORKS_NO_ERROR;
}


int ACompilation::addOperation_SPACE_TO_DEPTH(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                              std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                              std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 2, 1,"ANEURALNETWORKS_SPACE_TO_DEPTH") );

    AnnOperand& input0 = (*operands)[inputs[0]];
    AnnOperand& input1 = (*operands)[inputs[1]];
    AnnOperand& output0 = (*operands)[outputs[0]];

    int32_t blockSize = input1.scalar.i32;
    CHECK_PARAMETER(input0.type > 2, INPUT, 0);
    CHECK_PARAMETER(input1.type == ANEURALNETWORKS_INT32 && input1.scalar.i32 >= 1, INPUT, 1);
    CHECK_PARAMETER(0 == input0.dimensions[1] % blockSize || 0 == input0.dimensions[2] % blockSize, INPUT, 1);
    CHECK_PARAMETER(output0.type > 2, OUTPUT, 0);

    // TODO: computing output tensor size????
    CHECK_PARAMETER(output0.dimensions[3] == input0.dimensions[3] * blockSize * blockSize ||\
                    output0.dimensions[1] == input0.dimensions[1] / blockSize ||\
                    output0.dimensions[2] == input0.dimensions[2] / blockSize, OUTPUT, 0);

    convertScalar2Tensor(vxGetContext((vx_reference)graph), input1);

    vx_nn_reorg_params_t p;
    p.block_size = input1.tensor;
    p.type = VX_REORG_SPACE_TO_DEPTH;

    VX_NODE_CHECK( vxReorgLayer2( graph, input0.tensor, &p, sizeof(p),output0.tensor));

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_DEPTH_TO_SPACE(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                              std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                              std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 2, 1,"ANEURALNETWORKS_DEPTH_TO_SPACE") );

    AnnOperand& input0 = (*operands)[inputs[0]];
    AnnOperand& input1 = (*operands)[inputs[1]];
    AnnOperand& output0 = (*operands)[outputs[0]];

    CHECK_PARAMETER(input0.type > 2, INPUT, 0);
    CHECK_PARAMETER(input1.type == ANEURALNETWORKS_INT32 && input1.scalar.i32 >= 1, INPUT, 1);
    CHECK_PARAMETER(output0.type > 2, OUTPUT, 0);

    int32_t block_size = input1.scalar.i32;
    CHECK_PARAMETER(output0.dimensions[1] == input0.dimensions[1] * block_size \
                            && output0.dimensions[2] == input0.dimensions[2] * block_size \
                            && output0.dimensions[3] == input0.dimensions[3]/(block_size * block_size), OUTPUT, 0);

    convertScalar2Tensor(vxGetContext((vx_reference)graph), input1);

    vx_nn_reorg_params_t p;
    p.block_size = input1.tensor;
    p.type = VX_REORG_DEPTH_TO_SPACE;

    VX_NODE_CHECK( vxReorgLayer2( graph, input0.tensor, &p, sizeof(p),output0.tensor));

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_BATCH_TO_SPACE_ND(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
    std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
    std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK(checkIndex(inputs, outputs, operands->size(), 2, 1, "ANEURALNETWORKS_BATCH_TO_SPACE_ND"));

    AnnOperand& input0 = (*operands)[inputs[0]];
    AnnOperand& block_sizes = (*operands)[inputs[1]];
    AnnOperand& output0 = (*operands)[outputs[0]];
    vx_int32 block_size[4] = { 0 }, temp = 0;

    CHECK_PARAMETER(input0.type > 2, INPUT, 0);
    CHECK_PARAMETER(block_sizes.type == ANEURALNETWORKS_TENSOR_INT32 && block_sizes.dimensionCount == 1, INPUT, 1);
    CHECK_PARAMETER(output0.type > 2, OUTPUT, 0);

    vxcMemcpy(model->getVXContext(), block_sizes.tensor, &block_size, VX_READ_ONLY);
    temp = block_size[0];
    block_size[0] = block_size[1];
    block_size[1] = temp;
    vxcMemcpy(model->getVXContext(), block_sizes.tensor, &block_size, VX_WRITE_ONLY);

    vx_nn_reorg_params_t p = { block_sizes.tensor, VX_REORG_BATCH_TO_SPACE_ND };

    VX_NODE_CHECK(vxReorgLayer2(graph, input0.tensor, &p, sizeof(p), output0.tensor));

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_PAD(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                                std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                                std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK(checkIndex(inputs, outputs, operands->size(), 2, 1, "ANEURALNETWORKS_PAD"));

    AnnOperand& input0      = (*operands)[inputs[0]];
    AnnOperand& padding     = (*operands)[inputs[1]];
    AnnOperand& output0     = (*operands)[outputs[0]];
    int32_t pad_fronts[4] = { 0 }, pad_backs[4] = { 0 }, pads[8] = {0};

    CHECK_PARAMETER(input0.type > 2, INPUT, 0);
    CHECK_PARAMETER(padding.type == ANEURALNETWORKS_TENSOR_INT32 && padding.dimensionCount == 2, INPUT, 1);
    CHECK_PARAMETER(output0.type > 2, OUTPUT, 0);

    vxcMemcpy(model->getVXContext(), padding.tensor, pads, VX_READ_ONLY);

    /*WHCN: W  */
    pad_fronts[0] = pads[4];
    pad_backs[0]  = pads[5];

    /*WHCN: H  */
    pad_fronts[1] = pads[2];
    pad_backs[1]  = pads[3];

    /*WHCN: C  */
    pad_fronts[2] = pads[6];
    pad_backs[2]  = pads[7];

    /*WHCN: N  */
    pad_fronts[3] = pads[0];
    pad_backs[3]  = pads[1];

    vx_uint32 data = 0;
    vx_scalar padv = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &data);
    vx_nn_pad_params_t p = { pad_fronts, pad_backs, (vx_uint8)padding.dimensions[0], VX_PAD_CONSTANT, padv};

    VX_NODE_CHECK(vxTensorPadNode(graph, input0.tensor, output0.tensor, &p, sizeof(p)));

    vxReleaseScalar(&padv);

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_SPACE_TO_BATCH_ND(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
    std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
    std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK(checkIndex(inputs, outputs, operands->size(), 3, 1, "ANEURALNETWORKS_SPACE_TO_BATCH_ND"));

    AnnOperand& input0      = (*operands)[inputs[0]];
    AnnOperand& block_sizes = (*operands)[inputs[1]];
    AnnOperand& paddings     = (*operands)[inputs[2]];
    AnnOperand& output0     = (*operands)[outputs[0]];

    CHECK_PARAMETER(input0.type > 2, INPUT, 0);
    CHECK_PARAMETER(block_sizes.type == ANEURALNETWORKS_TENSOR_INT32, INPUT, 1);
    CHECK_PARAMETER(paddings.type == ANEURALNETWORKS_TENSOR_INT32, INPUT, 2);
    CHECK_PARAMETER(output0.type > 2, OUTPUT, 0);

    //Convert block_sizes & paddings memory layout with khronos in space2batch
    Convert_tensor_for_space2batch(block_sizes.dimensionCount, graph, block_sizes.tensor);
    Convert_tensor_for_space2batch(paddings.dimensionCount, graph, paddings.tensor);

    vx_nn_reorg_params_ext_t p = { { block_sizes.tensor, VX_REORG_SPACE_TO_BATCH_ND }, paddings.tensor };

    VX_NODE_CHECK(vxReorgLayer2(graph, input0.tensor, (vx_nn_reorg_params)&p, sizeof(p), output0.tensor));

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_EMBEDDING_LOOKUP(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                                std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                                std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 2, 1,"ANEURALNETWORKS_EMBEDDING_LOOKUP") );

    AnnOperand& input0 = (*operands)[inputs[0]];
    AnnOperand& input1 = (*operands)[inputs[1]];
    AnnOperand& output0 = (*operands)[outputs[0]];

    CHECK_PARAMETER(input0.type == ANEURALNETWORKS_TENSOR_INT32 && input0.dimensionCount == 1, INPUT, 0);
    CHECK_PARAMETER(input1.dimensionCount >= 2, INPUT, 1);

    CHECK_PARAMETER(output0.dimensionCount == input1.dimensionCount, OUTPUT, 0);
    for(size_t i = 1; i < input1.dimensionCount; i++) {
        CHECK_PARAMETER(output0.dimensions[i] == input1.dimensions[i], OUTPUT, 0);
    }
#ifdef NN_DEBUG
    PRINT_PARAMETER_INFO();
#endif

    vx_enum rank = VX_TENSOR_RANK_WHCN;
    vxSetTensorAttribute((vx_tensor)input1.tensor, VX_TENSOR_RANK, &rank, sizeof(vx_enum));
    vxSetTensorAttribute((vx_tensor)output0.tensor, VX_TENSOR_RANK, &rank, sizeof(vx_enum));

    CONVERT_RANK_FORMAT();

    vx_uint32 *tableDims = new vx_uint32 [4];
    vx_uint32 *outputDims = new vx_uint32 [4];

    convertDims( tableDims ,input1.dimensions, input1.dimensionCount);
    convertDims( outputDims ,output0.dimensions, output0.dimensionCount);

    vx_tensor tableTensor = vxReshapeTensor(input1.tensor, (vx_int32 *)tableDims, input1.dimensionCount);
    vx_tensor outputTensor = vxReshapeTensor(output0.tensor, (vx_int32 *)outputDims, output0.dimensionCount);

    VX_NODE_CHECK( vxTensorTableLookupNode2(graph, input0.tensor, tableTensor, outputTensor ) );

    delete [] tableDims;
    delete [] outputDims;

    vxReleaseTensor(&tableTensor);
    vxReleaseTensor(&outputTensor);

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_RNN(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                   std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                   std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 6, 2,"ANEURALNETWORKS_RNN") );

    size_t inputidx_2d[4] = {0, 1,2,4};
    for(auto idx:inputidx_2d)
    {
        CHECK_PARAMETER( (*operands)[inputs[idx]].dimensionCount == 2, INPUT, idx);
    }

    CHECK_PARAMETER( (*operands)[inputs[3]].dimensionCount == 1, INPUT, 3);
    CHECK_PARAMETER( (*operands)[outputs[0]].dimensionCount == 2, OUTPUT, 0);
    CHECK_PARAMETER( (*operands)[outputs[1]].dimensionCount == 2, OUTPUT, 1);

    CONVERT_RANK_FORMAT();

    convertScalar2Tensor(vxGetContext((vx_reference)graph), (*operands)[inputs[5]]);

    vx_nn_rnn_params_t p;
    p.weights = (*operands)[inputs[1]].tensor;
    p.recurrent_weights = (*operands)[inputs[2]].tensor;
    p.bias = (*operands)[inputs[3]].tensor;
    p.state_in = (*operands)[inputs[4]].tensor;
    p.activation = (*operands)[inputs[5]].tensor;

    VX_NODE_CHECK( vxRNNLayer(graph,
                              (*operands)[inputs[0]].tensor,
                              &p,
                              sizeof(p),
                              (*operands)[outputs[0]].tensor,
                              (*operands)[outputs[1]].tensor
    ));

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_LOCAL_RESPONSE_NORMALIZATION(vx_graph graph, AModel *model,
                                                            std::vector<AnnOperand> *operands,
                                                            std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                                            std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 5, 1,"ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION") );
    AnnOperand& input0 = (*operands)[inputs[0]];
    AnnOperand& input1 = (*operands)[inputs[1]];
    AnnOperand& input2 = (*operands)[inputs[2]];
    AnnOperand& input3 = (*operands)[inputs[3]];
    AnnOperand& input4 = (*operands)[inputs[4]];
    AnnOperand& output0 = (*operands)[outputs[0]];

    CHECK_PARAMETER(input0.dimensionCount == 4, INPUT, 0);
    CHECK_PARAMETER(input1.type == ANEURALNETWORKS_INT32, INPUT, 1);
    //CHECK_PARAMETER(input2.type == ANEURALNETWORKS_FLOAT32 && input2.scalar.fp32 != 0, INPUT, 2);
    CHECK_PARAMETER(input3.type == ANEURALNETWORKS_FLOAT32, INPUT, 3);
    CHECK_PARAMETER(input4.type == ANEURALNETWORKS_FLOAT32, INPUT, 4);
    CHECK_PARAMETER(input0.dimensionCount == output0.dimensionCount, OUTPUT, 0);

    CONVERT_RANK_FORMAT();

    vx_nn_normalization_params_t p;
    p.type = VX_NN_NORMALIZATION_ACROSS_MAPS;
    p.norm_size = input1.scalar.i32 * 2 + 1;
    p.bias = input2.scalar.fp32;
    p.alpha = input3.scalar.fp32;
    p.beta = input4.scalar.fp32;
#ifdef NN_DEBUG_DETAIL_INFO
    printf("ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION:\n");
    std::cout<<"norm_size: "<<p.norm_size<<endl;
    std::cout<<"bias: "<<p.bias<<endl;
    std::cout<<"alpha: "<<p.alpha<<endl;
    std::cout<<"beta: "<<p.beta<<endl;
#endif
    VX_NODE_CHECK( vxNormalizationLayer2(graph, input0.tensor,&p, sizeof(vx_nn_normalization_params_t), output0.tensor) );

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_HASHTABLE_LOOKUP(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                                std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                                std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 3, 2,"ANEURALNETWORKS_HASHTABLE_LOOKUP") );

    AnnOperand& input0 = (*operands)[inputs[0]];
    AnnOperand& input1 = (*operands)[inputs[1]];
    AnnOperand& input2 = (*operands)[inputs[2]];
    AnnOperand& output0 = (*operands)[outputs[0]];
    AnnOperand& output1 = (*operands)[outputs[1]];

    CHECK_PARAMETER(input0.type == ANEURALNETWORKS_TENSOR_INT32 && input0.dimensionCount == 1, INPUT, 0);
    CHECK_PARAMETER(input1.type == ANEURALNETWORKS_TENSOR_INT32 && input1.dimensionCount == 1, INPUT, 1);
    CHECK_PARAMETER(input2.dimensions[0] == input1.dimensions[0], INPUT, 2);
    CHECK_PARAMETER(input2.type > 2, INPUT, 2);

    CHECK_PARAMETER(output0.dimensions[0] == input0.dimensions[0],OUTPUT,0);
    CHECK_PARAMETER(output1.dimensions[0] == input0.dimensions[0] && output1.type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM ,OUTPUT,0);

    CONVERT_RANK_FORMAT();

    vx_nn_hashlut_params_t p;
    p.keys = input1.tensor;
    p.values = input2.tensor;
    VX_NODE_CHECK( vxHashTableLookupLayer(graph, input0.tensor, &p, sizeof(vx_nn_hashlut_params_t), output1.tensor, output0.tensor));

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_LSTM(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                    std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                    std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{

    //NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 23, 4,"ANEURALNETWORKS_LSTM") );
    vx_bool enable_layernorm = (inputs.size() > 23) ? vx_true_e : vx_false_e;

    //store 1-D or 2-D tensor idxes of inputs for check parameter
    size_t inputidx_2d[12];
    for (int k = 0; k < 9; ++k) {
        inputidx_2d[k] = k;
    }
    inputidx_2d[9] = 16;
    inputidx_2d[10] = 18;
    inputidx_2d[11] = 19;

    for(auto i : inputidx_2d)
    {
        CHECK_PARAMETER( (*operands)[inputs[i]].dimensionCount == 2, INPUT, i);
    }

    // check 1-D tensor
    for (size_t j = 9; j < 16; ++j) {
        CHECK_PARAMETER((*operands)[inputs[j]].dimensionCount == 1, INPUT, j);
    }
    CHECK_PARAMETER((*operands)[inputs[17]].dimensionCount == 1, INPUT, 17);


#define SET_HIGH_PRECISION(operands, idx) {\
        AnnOperand &operand = (*operands)[idx];\
        if( (operand).tensorAttribute.valued && (operand).tensorAttribute.precision != VX_TENSOR_PRECISION_HIGH)\
        {\
            (operand).tensorAttribute.precision = VX_TENSOR_PRECISION_HIGH;\
            VX_ERR_CHECK( vxSetTensorAttribute((operand).tensor, VX_TENSOR_PRECISION, &(operand).tensorAttribute.precision, sizeof(vx_enum)));\
        }\
    }

    SET_HIGH_PRECISION(operands, inputs[12]);
    SET_HIGH_PRECISION(operands, inputs[13]);
    SET_HIGH_PRECISION(operands, inputs[14]);
    SET_HIGH_PRECISION(operands, inputs[15]);
    SET_HIGH_PRECISION(operands, inputs[17]);

    CONVERT_RANK_FORMAT();

    vx_context context = vxGetContext((vx_reference)graph);
    convertScalar2Tensor(context, (*operands)[inputs[20]]);
    convertScalar2Tensor(context, (*operands)[inputs[21]]);
    convertScalar2Tensor(context, (*operands)[inputs[22]]);


    vx_nn_lstm_params_ext_t p = {0};
    /* 1 ~ 4 */ //TODO: check tensor is null?
    p.base.input2input_weight = (*operands)[inputs[1]].tensor;
    p.base.input2forget_weight = (*operands)[inputs[2]].tensor;
    p.base.input2cell_weight = (*operands)[inputs[3]].tensor;
    p.base.input2output_weight = (*operands)[inputs[4]].tensor;

    /* 5 ~ 8 */
    p.base.recurrent2input_weight = (*operands)[inputs[5]].tensor;
    p.base.recurrent2forget_weight  = (*operands)[inputs[6]].tensor;
    p.base.recurrent2cell_weight  = (*operands)[inputs[7]].tensor;
    p.base.recurrent2output_weight = (*operands)[inputs[8]].tensor;

    /* 9 ~ 11 */
    p.base.cell2input_weight  = (*operands)[inputs[9]].tensor;
    p.base.cell2forget_weight = (*operands)[inputs[10]].tensor;
    p.base.cell2output_weight = (*operands)[inputs[11]].tensor;

    /* 12 ~ 15 */
    p.base.input_gate_bias = (*operands)[inputs[12]].tensor;
    p.base.forget_gate_bias = (*operands)[inputs[13]].tensor;
    p.base.cell_bias = (*operands)[inputs[14]].tensor;
    p.base.output_gate_bias = (*operands)[inputs[15]].tensor;

    /* 16 ~ 17 */
    p.base.projection_weight = (*operands)[inputs[16]].tensor;
    p.base.projection_bias = (*operands)[inputs[17]].tensor;

    /* 20 ~ 22 */
    p.base.activation = (*operands)[inputs[20]].tensor;
    p.base.cell_clip = (*operands)[inputs[21]].tensor;
    p.base.proj_clip = p.base.projection_weight == NULL ? NULL:(*operands)[inputs[22]].tensor;

    if (enable_layernorm)
    {
        p.layernorm2input_weight = (*operands)[inputs[23]].tensor;
        p.layernorm2forget_weight = (*operands)[inputs[24]].tensor;
        p.layernorm2cell_weight = (*operands)[inputs[25]].tensor;
        p.layernorm2output_weight = (*operands)[inputs[26]].tensor;
    }

    if (p.base.input2input_weight == NULL || (p.base.cell2input_weight == NULL && p.base.cell2output_weight != NULL))
    {
        p.base.input2input_weight = NULL;
        p.base.recurrent2input_weight = NULL;
        p.base.input_gate_bias = NULL;
    }

    VX_NODE_CHECK( vxLstmUnitLayer(graph,
                                   (*operands)[inputs[0]].tensor,
                                   (*operands)[inputs[18]].tensor,
                                   (*operands)[inputs[19]].tensor,
                                   (vx_nn_lstm_params_t*)&p,
                                   (enable_layernorm ? sizeof(vx_nn_lstm_params_ext_t):sizeof(vx_nn_lstm_params_t)),
                                   (*operands)[outputs[0]].tensor,
                                   (*operands)[outputs[1]].tensor,
                                   (*operands)[outputs[2]].tensor,
                                   (*operands)[outputs[3]].tensor) );

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_SVDF(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                    std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                    std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 7, 2,"ANEURALNETWORKS_SVDF") );

    AnnOperand& input0 = (*operands)[inputs[0]];
    AnnOperand& input1 = (*operands)[inputs[1]];
    AnnOperand& input2 = (*operands)[inputs[2]];
    AnnOperand& input3 = (*operands)[inputs[3]];
    AnnOperand& input4 = (*operands)[inputs[4]];
    AnnOperand& input5 = (*operands)[inputs[5]];
    AnnOperand& input6 = (*operands)[inputs[6]];

    AnnOperand& output0 = (*operands)[outputs[0]];
    AnnOperand& output1 = (*operands)[outputs[1]];



    CHECK_PARAMETER( isTensor(input0.type) && input0.dimensionCount == 2, INPUT, 0);
    CHECK_PARAMETER( isTensor(input1.type) && input1.dimensions[1] == input0.dimensions[1], INPUT, 1);
    CHECK_PARAMETER( isTensor(input2.type) && input2.dimensions[0]== input1.dimensions[0], INPUT, 2);
    CHECK_PARAMETER( isTensor(input4.type) && input4.dimensionCount == 2, INPUT, 4);
    CHECK_PARAMETER( isTensor(output0.type) && output0.dimensions[0] == input0.dimensions[0] && output0.dimensions[1] == input4.dimensions[1], OUTPUT, 0);
    CHECK_PARAMETER( input5.type == ANEURALNETWORKS_INT32, INPUT, 5);
    CHECK_PARAMETER( input5.scalar.i32 != 0, INPUT, 5);
    CHECK_PARAMETER( isTensor(input3.type) && input3.dimensionCount == 1 && input3.dimensions[0] == input1.dimensions[0] / input5.scalar.i32, INPUT, 3);
    CHECK_PARAMETER( isTensor(output1.type) && output1.dimensions[0] == input0.dimensions[0] && output1.dimensions[1] == input1.dimensions[0] / input5.scalar.i32, OUTPUT, 1);

    CHECK_PARAMETER( input6.type == ANEURALNETWORKS_INT32, INPUT, 6);

    SET_HIGH_PRECISION(operands, inputs[3]);

    CONVERT_RANK_FORMAT();

    /*update the implement as required by NNCTS 1.1*/
    vx_context context = vxGetContext((vx_reference)graph);
    convertScalar2Tensor(context, input5);
    convertScalar2Tensor(context, input6);

    vx_nn_svdf_params_t p;
    p.weights_feature = input1.tensor;
    p.recurrent_time = input2.tensor;
    p.bias = input3.tensor;
    p.state_in = input4.tensor;
    p.rank = input5.tensor;
    p.activation = (*operands)[inputs[6]].tensor;

    VX_NODE_CHECK( vxSVDFLayer(graph, input0.tensor, &p, sizeof(p), output0.tensor, output1.tensor) );

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_LSH_PROJECTION(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                              std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                              std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 4, 1,"ANEURALNETWORKS_LSH_PROJECTION") );

    AnnOperand& hash = (*operands)[inputs[0]];
    AnnOperand& input = (*operands)[inputs[1]];
    AnnOperand& weight = (*operands)[inputs[2]];
    AnnOperand& type = (*operands)[inputs[3]];
    AnnOperand& output = (*operands)[outputs[0]];

    CHECK_PARAMETER( hash.dimensionCount==2, INPUT, 0);
    CHECK_PARAMETER( input.dimensionCount>=1, INPUT, 1);

    vx_enum precision = VX_TENSOR_PRECISION_HIGH;
    vxSetTensorAttribute((vx_tensor)hash.tensor, VX_TENSOR_PRECISION, &precision, sizeof(vx_enum));
    vxSetTensorAttribute((vx_tensor)weight.tensor, VX_TENSOR_PRECISION, &precision, sizeof(vx_enum));
    hash.tensorAttribute.precision = VX_TENSOR_PRECISION_HIGH;
    weight.tensorAttribute.precision = VX_TENSOR_PRECISION_HIGH;

    if( ANEURALNETWORKS_INT32 == type.type)
    {
        vx_uint32 size[] = {1};
        if(NULL == type.tensor)
        {
            vx_tensor_create_params_t param0 = { 1, size, VX_TYPE_INT32, 0, {{0}}};
            vx_tensor tensor = vxCreateTensor2(model->getVXContext(), &param0, sizeof(vx_tensor_create_params_t) );
            if (tensor == NULL)
            {
                printf("vxCreateTensor failure! at line %d\n", __LINE__);
                assert(0);
            }
            type.tensor = tensor;
        }

        vxcMemcpy(model->getVXContext(), type.tensor, &type.scalar.i32, VX_WRITE_ONLY);

        type.type = ANEURALNETWORKS_TENSOR_INT32;
        type.tensorAttribute.dataType = VX_TYPE_INT32;
        type.tensorAttribute.lifeTime = VX_TENSOR_LIFE_TIME_STATIC;
        type.tensorAttribute.valued = vx_true_e;
    }

    CONVERT_RANK_FORMAT();

    vx_nn_lshproj_params_t p;
    p.hash_func = hash.tensor;
    p.weights = weight.tensor;
    p.type = type.tensor;

    VX_NODE_CHECK( vxLSHProjectionLayer(graph, input.tensor, &p, sizeof(p), output.tensor));

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_DEQUANTIZE(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                          std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                          std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(), 1, 1,"ANEURALNETWORKS_DEQUANTIZE") );

    AnnOperand& input = (*operands)[inputs[0]];
    AnnOperand& output = (*operands)[outputs[0]];
    CHECK_PARAMETER( input.type ==  ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, INPUT, 0);
    CHECK_PARAMETER( output.type ==  ANEURALNETWORKS_TENSOR_FLOAT32, OUTPUT, 1);
    CHECK_PARAMETER( output.dimensionCount == input.dimensionCount, OUTPUT, 0);
    for(size_t i = 0; i < input.dimensionCount; i++)
        CHECK_PARAMETER(output.dimensions[i] == input.dimensions[i], OUTPUT, 0);

    VX_NODE_CHECK(vxTensorCopyNode(graph, input.tensor, output.tensor));

    return ANEURALNETWORKS_NO_ERROR;
}


int ACompilation::addOperation_DEPTHWISE_CONV_2D(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                                 std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                                 std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    size_t inputSize = inputs.size();
    size_t outputSize = outputs.size();
    if( ((inputSize != 8) && (inputSize != 11)) || outputSize != 1)
    {
        error_log("ANEURALNETWORKS_DEPTHWISE_CONV_2D  paramter error\n");
        return ANEURALNETWORKS_OP_FAILED;
    }

    AnnOperand& inputOp = (*operands)[inputs[0]];
    AnnOperand& weightOp = (*operands)[inputs[1]];
    AnnOperand& biasOp = (*operands)[inputs[2]];
    AnnOperand& outputOp = (*operands)[outputs[0]];

    CHECK_PARAMETER(inputOp.dimensionCount == 4, INPUT, 0);
    CHECK_PARAMETER(weightOp.dimensionCount == 4, INPUT, 1);
    CHECK_PARAMETER(biasOp.dimensionCount == 1, INPUT, 2);
    for(int i = 3; i < (int)inputSize; i++)
        CHECK_PARAMETER((*operands)[inputs[i]].type == ANEURALNETWORKS_INT32, INPUT, i);

    CHECK_PARAMETER(outputOp.dimensionCount == 4, OUTPUT, 0);

    uint32_t pad_x_left = 0,pad_x_right = 0;
    uint32_t pad_y_top = 0, pad_y_bottom = 0;
    uint32_t stride_x = 0, stride_y = 0;
    if(inputSize == 11){
        pad_x_left = (*operands)[inputs[3]].scalar.i32;
        pad_x_right = (*operands)[inputs[4]].scalar.i32;
        pad_y_top = (*operands)[inputs[5]].scalar.i32;
        pad_y_bottom = (*operands)[inputs[6]].scalar.i32;
        stride_x = (*operands)[inputs[7]].scalar.i32;
        stride_y = (*operands)[inputs[8]].scalar.i32;
    } else{
        int32_t padschme = (*operands)[inputs[3]].scalar.i32;
        stride_x = (*operands)[inputs[4]].scalar.i32;
        stride_y = (*operands)[inputs[5]].scalar.i32;

        computePadding(inputOp.dimensions[1], outputOp.dimensions[1], weightOp.dimensions[1], stride_y, pad_y_top, pad_y_bottom, padschme);
        computePadding(inputOp.dimensions[2], outputOp.dimensions[2], weightOp.dimensions[2], stride_x, pad_x_left, pad_x_right, padschme);
    }

    int32_t depth_multiplier = (*operands)[inputs[inputs.size() - 2]].scalar.i32;
    vx_nn_convolution_params_ext2_t params = {
        {
            {
                (vx_size)pad_x_left, (vx_size)pad_y_top, VX_CONVERT_POLICY_SATURATE,  VX_ROUND_POLICY_TO_ZERO, VX_NN_DS_SIZE_ROUNDING_FLOOR, 0, 0
            },
            (vx_size)pad_x_right, (vx_size)pad_y_bottom, VX_PAD_CONSTANT, 0
            },
            (vx_uint32)stride_x, (vx_uint32)stride_y, depth_multiplier
    };

    uint32_t fuseCode = (*operands)[inputs[inputs.size() - 1]].scalar.i32;
    vx_tensor convTensor = createTmpTensor(graph, outputOp, fuseCode, tmpTensors);

    vx_enum precision = VX_TENSOR_PRECISION_HIGH;
    vxSetTensorAttribute((vx_tensor)biasOp.tensor, VX_TENSOR_PRECISION, &precision, sizeof(vx_enum));
    biasOp.tensorAttribute.precision = VX_TENSOR_PRECISION_HIGH;

    CONVERT_RANK_FORMAT();

    VX_NODE_CHECK( vxConvolutionLayer(graph,
        inputOp.tensor,
        weightOp.tensor,
        biasOp.tensor,
        (const vx_nn_convolution_params_t *)&params,
        sizeof(vx_nn_convolution_params_ext2_t),
        (vx_tensor)convTensor) );

    NN_ERR_CHECK( addActiveOp( graph, convTensor, outputOp.tensor, fuseCode, nodeContainer) );

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_CONCATENATION(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                                 std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                                 std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    int32_t concatAxis = (*operands)[ inputs[inputs.size() - 1] ].scalar.i32;
    CHECK_PARAMETER(concatAxis < 4, INPUT, inputs.size() -1 );

    uint32_t *dims = (*operands)[inputs[0]].dimensions;
    for(uint32_t i = 1; i < inputs.size()-1; i++)
    {
        CHECK_PARAMETER((*operands)[inputs[0]].type == (*operands)[inputs[i]].type, INPUT, i);
        CHECK_PARAMETER( ( (ANEURALNETWORKS_TENSOR_FLOAT32 == (*operands)[inputs[i]].type) || (ANEURALNETWORKS_TENSOR_QUANT8_ASYMM == (*operands)[inputs[i]].type) ),
            INPUT, i);

        if((*operands)[inputs[0]].type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM)
        {
            CHECK_SPECIAL((*operands)[inputs[0]].scale== (*operands)[inputs[i]].scale, i, "scale");
            CHECK_SPECIAL((*operands)[inputs[0]].zeroPoint== (*operands)[inputs[i]].zeroPoint, i, "zeropoint");
        }

        for(uint32_t idx = 0; idx < (*operands)[inputs[0]].dimensionCount; idx ++)
        {
            if (idx == (uint32_t)concatAxis)
                continue;
            CHECK_PARAMETER( dims[idx] == (*operands)[inputs[i]].dimensions[idx], INPUT, i);
        }
    }

    int32_t sumConcatAxis = 0;
    for(uint32_t i = 0; i < inputs.size()-1; i++)
    {
       sumConcatAxis += (*operands)[inputs[i]].dimensions[concatAxis];
    }
    CHECK_PARAMETER((vx_uint32)sumConcatAxis == (*operands)[outputs[0]].dimensions[concatAxis], OUTPUT, 0);

    std::vector<vx_tensor> vTensor;
    for (uint32_t i = 0; i < inputs.size() - 1; i++)
    {
        vTensor.push_back((*operands)[inputs[i]].tensor);
    }

    vx_object_array objectArray = vxCreateTensorObjectArray(vxGetContext((vx_reference)graph), inputs.size()-1, &vTensor[0]);

    auto convertAsixFromNHWCtoWHCN = [&](vx_uint32 TFAxis)->vx_uint32
    {
        if ((*operands)[inputs[0]].dimensionCount== 4)
        {
            vx_int32 nhwc2whcn[4] = { 3, 1, 0, 2 };
            return nhwc2whcn[TFAxis];
        }
        else if ((*operands)[inputs[0]].dimensionCount == 2)
        {
            vx_int32 nhwc2whcn[2] = { 1, 0 };
            return nhwc2whcn[TFAxis];
        }
        else if((*operands)[inputs[0]].dimensionCount == 3)
        {
            vx_int32 nhwc2whcn[3] = { 2, 1, 0};/*convert dimension as nhw*/
            return nhwc2whcn[TFAxis];
        }
        return (*operands)[inputs[0]].dimensionCount - 1;
    };

    vx_nn_concat_params_t p = {convertAsixFromNHWCtoWHCN((vx_uint32)concatAxis)};
    VX_NODE_CHECK( vxConcatIndefiniteLayer(graph, objectArray, &p, sizeof(p), (*operands)[outputs[0]].tensor) );

    VX_ERR_CHECK( vxReleaseObjectArray(&objectArray) );

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_DIV(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                                 std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                                 std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(),3, 1, "ANEURALNETWORKS_DIV") );

    AnnOperand& operand0 = (*operands)[inputs[0]];
    AnnOperand& operand1 = (*operands)[inputs[1]];
    AnnOperand& operand2 = (*operands)[inputs[2]];
    AnnOperand& output = (*operands)[outputs[0]];

    CHECK_PARAMETER(operand0.type >2 && operand0.dimensionCount <= 4, INPUT, 0);
    CHECK_PARAMETER(operand0.type == operand1.type && (operand1.dimensionCount <= 4), INPUT, 1);
    CHECK_PARAMETER(operand2.type == ANEURALNETWORKS_INT32, INPUT, 2);
    CHECK_PARAMETER(operand0.type == output.type, OUTPUT, 0);

    uint32_t fuseCode = (*operands)[inputs[inputs.size() - 1]].scalar.i32;
    vx_tensor convTensor = createTmpTensor(graph, output, fuseCode, tmpTensors);

    vx_float32 scaler = 1.0f;
    vx_scalar vxScaler = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &scaler);

    VX_NODE_CHECK( vxTensorDivideNode(graph, operand0.tensor, operand1.tensor,  vxScaler, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, convTensor) );
    NN_ERR_CHECK( addActiveOp( graph, convTensor, output.tensor, fuseCode, nodeContainer) );

    vxReleaseScalar(&vxScaler);

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_SUB(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                                 std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                                 std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(),3, 1, "ANEURALNETWORKS_SUB") );

    AnnOperand& operand0 = (*operands)[inputs[0]];
    AnnOperand& operand1 = (*operands)[inputs[1]];
    AnnOperand& operand2 = (*operands)[inputs[2]];
    AnnOperand& output = (*operands)[outputs[0]];

    CHECK_PARAMETER(operand0.type >2 && operand0.dimensionCount <= 4, INPUT, 0);
    CHECK_PARAMETER(operand0.type == operand1.type && (operand1.dimensionCount <= 4), INPUT, 1);
    CHECK_PARAMETER(operand2.type == ANEURALNETWORKS_INT32, INPUT, 2);
    CHECK_PARAMETER(operand0.type == output.type, OUTPUT, 0);

    uint32_t fuseCode = (*operands)[inputs[inputs.size() - 1]].scalar.i32;
    vx_tensor convTensor = createTmpTensor(graph, output, fuseCode, tmpTensors);

    VX_NODE_CHECK( vxTensorSubtractNode(graph, operand0.tensor, operand1.tensor, VX_CONVERT_POLICY_WRAP, convTensor) );
    NN_ERR_CHECK( addActiveOp( graph, convTensor, output.tensor, fuseCode, nodeContainer) );

    return ANEURALNETWORKS_NO_ERROR;
}

int ACompilation::addOperation_TRANSPOSE(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands,
                                                 std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                                                 std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors)
{
    NN_ERR_CHECK( checkIndex(inputs, outputs, operands->size(),2, 1, "ANEURALNETWORKS_TRANSPOSE") );

    AnnOperand& input = (*operands)[inputs[0]];
    AnnOperand& perm = (*operands)[inputs[1]];
    AnnOperand& output = (*operands)[outputs[0]];

    CHECK_PARAMETER(input.type >2 && input.dimensionCount <= 4, INPUT, 0);
    CHECK_PARAMETER(perm.type == ANEURALNETWORKS_TENSOR_INT32 && (perm.dimensionCount == 1), INPUT, 1);
    CHECK_PARAMETER(input.type == output.type, OUTPUT, 0);

    vx_uint32 permuteIdx[4];
    for(vx_uint32 i = 0; i < input.dimensionCount; i++)
        permuteIdx[i] = input.dimensions[input.dimensionCount - 1 - i];

    if(perm.tensorAttribute.valued == vx_true_e)
    {
        vxcMemcpy(model->getVXContext(), perm.tensor, permuteIdx, VX_READ_ONLY);
    }
    /*adjust the order to ovx driver.
      because the rank of tensor would be convered from nhwc to whcn.
    */
    vx_uint32 ovxRank[4] = {0};
    if(input.dimensionCount == 4)
    {
        vx_uint32 order[] = { 2, 1, 3, 0 };
        vx_uint32 whcn_order[] = { 3, 1, 0, 2 };
        vx_uint32 whcn_out[] = { permuteIdx[order[0]], permuteIdx[order[1]], permuteIdx[order[2]], permuteIdx[order[3]] };
        for(int i = 0; i < 4; i ++)
            ovxRank[i] = whcn_order[whcn_out[i] ];
    }
    else if(input.dimensionCount == 3)
    {
        vx_uint32 order[] = {2, 0, 1};
        for(int i = 0; i < 3; i++)
            ovxRank[i] = permuteIdx[ order[i] ];
    }
    else if(input.dimensionCount == 2)
    {
        vx_uint32 order[] = {0,1};
        for(int i = 0; i < 2; i++)
            ovxRank[i] = permuteIdx[order[i]];
    }

    VX_NODE_CHECK( vxTensorPermuteNode(graph, input.tensor, output.tensor, ovxRank, input.dimensionCount) );

    return ANEURALNETWORKS_NO_ERROR;
}

