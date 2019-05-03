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


#ifndef _A_COMPILATION_H_
#define _A_COMPILATION_H_
#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>
#include <map>
#include "AModel.h"
#include "util.h"
#include "NeuralNetworksOEM.h"

/*
 * Create a graph and Add oprand and opration to it, applying the model.
 */

#define VX_NODE_CHECK(node){\
    vx_node newNode = node;\
    if(!newNode){\
        fprintf(stderr, "add operation fail: line: %d, function: %s, file: %s.\n", __LINE__, __FUNCTION__, __FILE__);\
        assert(0);\
    }\
    else\
    {\
       nodeContainer.push_back(newNode);\
    }\
}

#define CONVERT_RANK_FORMAT() {\
    for(size_t i =0; i< inputs.size(); i++){\
        vx_uint32 idx = inputs[i];\
        convertRankAndFormat(vxGetContext((vx_reference)graph), (*operands)[idx]);\
    }\
}

class ACompilation
{
public:
    ACompilation(AModel *model);
    ~ACompilation();
    AModel* getAModel() { return m_model;}
    vx_graph getGraph() { return  m_graph;}
    int setPreference(int32_t preference);
    int run();


private:
    typedef int (*opFunc)(vx_graph graph, AModel *model, std::vector<AnnOperand> *operands, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs, std::vector<vx_node> &nodeContainer, std::vector<vx_tensor> *tmpTensors);

#define PUSH_OP(TYPE) {\
    std::map<int, opFunc>::const_iterator got = m_opContainer.find( ANEURALNETWORKS_##TYPE);\
    if(got == m_opContainer.end())\
    {\
        std::pair<int, opFunc> pp(ANEURALNETWORKS_##TYPE, addOperation_##TYPE);\
        m_opContainer.insert(pp);\
    }\
    else{\
        warning_log("current function has been insert");\
    }\
}

    ADD_OPERATION(ADD);
    ADD_OPERATION(CONV_2D);
    ADD_OPERATION(RELU);
    ADD_OPERATION(RELU1);
    ADD_OPERATION(RELU6);
    ADD_OPERATION(FULLY_CONNECTED);
    ADD_OPERATION(SOFTMAX);
    ADD_OPERATION(MAX_POOL_2D);
    ADD_OPERATION(AVERAGE_POOL_2D);
    ADD_OPERATION(MUL);
    ADD_OPERATION(RESHAPE);
    ADD_OPERATION(L2_NORMALIZATION);
    ADD_OPERATION(LOGISTIC);
    ADD_OPERATION(TANH);
    ADD_OPERATION(FLOOR);
    ADD_OPERATION(RESIZE_BILINEAR);
    ADD_OPERATION(SPACE_TO_DEPTH);
    ADD_OPERATION(DEPTH_TO_SPACE);
    ADD_OPERATION(EMBEDDING_LOOKUP);

    ADD_OPERATION(LOCAL_RESPONSE_NORMALIZATION);
    ADD_OPERATION(HASHTABLE_LOOKUP);
    ADD_OPERATION(LSTM);

    ADD_OPERATION(RNN);
    ADD_OPERATION(SVDF);
    ADD_OPERATION(LSH_PROJECTION);
    ADD_OPERATION(L2_POOL_2D);
    ADD_OPERATION(DEQUANTIZE);
    ADD_OPERATION(DEPTHWISE_CONV_2D);
    ADD_OPERATION(CONCATENATION);

    ADD_OPERATION(DIV);
    ADD_OPERATION(SUB);
    ADD_OPERATION(TRANSPOSE);
    ADD_OPERATION(MEAN);
    ADD_OPERATION(SQUEEZE);
    ADD_OPERATION(STRIDED_SLICE);
    ADD_OPERATION(SPACE_TO_BATCH_ND);
    ADD_OPERATION(BATCH_TO_SPACE_ND);
    ADD_OPERATION(PAD);


    std::map<int, opFunc> m_opContainer;
    std::vector<vx_tensor> m_tempTensor;
    std::vector<vx_node> m_nodes;

    vx_context   m_context;
    vx_graph     m_graph;
    AModel *m_model;

    bool m_lockFlag;
    PreferenceCode m_compilationRefence;
};



#endif

