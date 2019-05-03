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



#ifndef __NNVX_TYPE_DEF_h
#define __NNVX_TYPE_DEF_h

#include <assert.h>
#include "NeuralNetworks.h"

#define NN_TENSOR_MAX_DIMENSION 4

#define HIGH_PRECISION_COMPUTE 1
#define  INPUT 0
#define  OUTPUT 1

#define NN_DEBUG_GRAPH 0 // 0:vxScheduleGraph 1:vxProcessGraph

#define VX_ERR_CHECK(err){\
    if(err != VX_SUCCESS){\
    fprintf(stderr, "error code : %d, line: %d, file: %s.\n", err, __LINE__, __FILE__);\
    }\
}

#define NN_ERR_CHECK(err){\
    if(err != ANEURALNETWORKS_NO_ERROR){\
        fprintf(stderr, "error code : %d, line: %d, function: %s, file: %s.\n", err, __LINE__, __FUNCTION__, __FILE__);\
        assert(0);\
        return err;\
    }\
}
#define CHECK_PARAMETER(condition, input, idx ){\
    if(!(condition)){\
        fprintf(stderr, "function %s :", __FUNCTION__) ;\
        fprintf(stderr, " parameter %d of %ss error\n", (int)(idx), (input) == INPUT ? "input" : "output") ;\
        assert(0);\
    return ANEURALNETWORKS_OP_FAILED;\
}\
}

#define CHECK_SPECIAL(condition, idx, str){\
    if(!(condition)){\
        fprintf(stderr, "Warning: different %s parameter %d of input tensors in function %s: \n", str, (int)(idx), __FUNCTION__) ;\
        assert(0);\
}\
}

#define CHECK_LESS(a, b){\
    if( (a) >= (b) ){\
        fprintf(stderr, "error: %d is not less than %d, line: %d, function: %s, file: %s.\n", (a), (b), __LINE__, __FUNCTION__, __FILE__);\
        assert(0);\
        return ANEURALNETWORKS_BAD_DATA;\
    }\
}

#define CHECK_EQUAL(a, b){\
    if( (a) != (b) ){\
        fprintf(stderr, "error: %d is not equal %d, line: %d, function: %s, file: %s.\n", (int)(a), (int)(b), __LINE__, __FUNCTION__, __FILE__);\
        assert(0);\
        return ANEURALNETWORKS_BAD_DATA;\
    }\
}

#define PRINT_PARAMETER_DIMS(input_or_output)\
{\
    for(size_t i = 0; i < input_or_output.size(); i++){\
std::cout<<"intput "<< i<<" : ";\
for(size_t j = 0; j < (*operands)[input_or_output[i]].dimensionCount; j++){\
std::cout<<(*operands)[input_or_output[i]].dimensions[j]<<" ";\
}\
std::cout<<"\n";\
}\
}
#define PRINT_PARAMETER_INFO()\
{\
    printf("function: %s\n", __FUNCTION__);\
    PRINT_PARAMETER_DIMS(inputs);\
    PRINT_PARAMETER_DIMS(outputs);\
}
inline bool isTensor(int32_t type)
{
    if( (type == ANEURALNETWORKS_TENSOR_FLOAT32) || (type == ANEURALNETWORKS_TENSOR_INT32) || (type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM))
    {
        return true;
    }

    return false;
}

inline bool isEmpty(uint32_t dimensionCount, uint32_t *dims)
{
    if(dimensionCount == 0)
        return true;
    for(uint32_t i = 0; i < dimensionCount; i++)
        if(dims[i] == 0)
            return true;
    return false;
}
inline int enumConvertorANN2VX(int type)
{
    switch (type){
        case ANEURALNETWORKS_TENSOR_INT32:
            return VX_TYPE_INT32;
        case ANEURALNETWORKS_TENSOR_FLOAT32:
            return VX_TYPE_FLOAT32;
        case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            return VX_TYPE_UINT8;
        default:
            break;
    }
    return VX_TYPE_INT32;
}

inline int MAX(int a, int b)
{
    return a > b? a:b;
}

inline void warning_log(std::string log){
    fprintf(stderr, "WARNING: %s\n", log.c_str());
}

#define error_log(log)\
{\
    fprintf(stderr, "ERROR: %s\n", std::string(log).c_str());\
    assert(0);\
}

#define INITIALIZE_STRUCT(values) {\
    memset((void*)&(values), 0, sizeof((values)));\
}

#define ADD_OPERATION(func) static int addOperation_##func(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs, std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors = NULL)

#endif