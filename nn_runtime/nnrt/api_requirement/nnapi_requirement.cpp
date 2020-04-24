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
#include "nnapi_requirement.hpp"

namespace api {
namespace requirement {
namespace nnapi {

static OpSpecCollection nnapiOpSpecCollection;

MatchedArgumentPtr match(const std::string& name, const std::vector<nnrt::OperandType>& args) {
    auto matchedArg = nnapiOpSpecCollection.match(name, args);
    if (nullptr == matchedArg) {
        return MatchedArgumentPtr();
    }
    return std::make_shared<MatchedArgument>(args.size(), matchedArg);
}

void NNapiRequirementRegister(const std::string& opName, const IArgList* arglist) {
    if (nnapiOpSpecCollection.m_Collection.end() == nnapiOpSpecCollection.m_Collection.find(opName)) {
        std::vector<const IArgList*> args;
        args.push_back(arglist);
        nnapiOpSpecCollection.m_Collection.insert(std::make_pair(opName, args));
    } else {
        nnapiOpSpecCollection.m_Collection[opName].push_back(arglist);
    }
}
}  // end of namespace nnapi
}
}

#define OP_SPEC_REGISTER nnapi::NNapiRequirement
#include "spec_macros.hpp"

#include "nnapi_spec/ANEURALNETWORKS_CONV_2D.hpp"
#include "nnapi_spec/ANEURALNETWORKS_RESIZE_BILINEAR.hpp"
#include "nnapi_spec/ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR.hpp"
#include "nnapi_spec/ANEURALNETWORKS_REDUCTION.hpp"
#include "nnapi_spec/ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION.hpp"
#include "nnapi_spec/ANEURALNETWORKS_L2_NORMALIZATION.hpp"
#include "nnapi_spec/ANEURALNETWORKS_CHANNEL_SHUFFLE.hpp"
#include "nnapi_spec/ANEURALNETWORKS_AXIS_ALIGNED_BBOX_TRANSFORM.hpp"
#include "nnapi_spec/ANEURALNETWORKS_GENERATE_PROPOSALS.hpp"
#include "nnapi_spec/ANEURALNETWORKS_RANDOM_MULTINOMIAL.hpp"
#include "nnapi_spec/ANEURALNETWORKS_ROI_POOLING.hpp"
#include "nnapi_spec/ANEURALNETWORKS_ROI_ALIGN.hpp"
#include "nnapi_spec/ANEURALNETWORKS_BOX_WITH_NMS_LIMIT.hpp"
#include "nnapi_spec/ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_RNN.hpp"
#include "nnapi_spec/ANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_RNN.hpp"
#include "nnapi_spec/ANEURALNETWORKS_LSTM.hpp"
#include "nnapi_spec/ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_LSTM.hpp"
#include "nnapi_spec/ANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_LSTM.hpp"
#include "nnapi_spec/ANEURALNETWORKS_TILE.hpp"
#include "nnapi_spec/ANEURALNETWORKS_TOPK_V2.hpp"
#include "nnapi_spec/ANEURALNETWORKS_LOG_SOFTMAX.hpp"
#include "nnapi_spec/ANEURALNETWORKS_HEATMAP_MAX_KEYPOINT.hpp"
#include "nnapi_spec/ANEURALNETWORKS_DETECTION_POSTPROCESSING.hpp"
#include "nnapi_spec/ANEURALNETWORKS_TRANSPOSE_CONV_2D.hpp"
#include "nnapi_spec/ANEURALNETWORKS_DEPTHWISE_CONV_2D.hpp"
#include "nnapi_spec/ANEURALNETWORKS_EXPAND_DIMS.hpp"
#include "nnapi_spec/ANEURALNETWORKS_INSTANCE_NORMALIZATION.hpp"
#include "nnapi_spec/ANEURALNETWORKS_SPACE_TO_DEPTH.hpp"
#include "nnapi_spec/ANEURALNETWORKS_SPLIT.hpp"
#include "nnapi_spec/ANEURALNETWORKS_SOFTMAX.hpp"
#include "nnapi_spec/ANEURALNETWORKS_PAD_V2.hpp"
#include "nnapi_spec/ANEURALNETWORKS_GROUPED_CONV_2D.hpp"