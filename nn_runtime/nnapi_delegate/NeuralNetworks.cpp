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
#include <memory>

#include "vsi_nn_pub.h"
#include "vsi_nn_log.h"
#include "nnrt/event.hpp"
#include "nnrt/compilation.hpp"
#include "nnrt/execution.hpp"
#include "nnrt/model.hpp"
#include "nnrt/file_map_memory.hpp"
#include "nnrt/error.hpp"
#include "nnrt/types.hpp"
#include "nnrt/op/operation.hpp"

#include "NeuralNetworks.h"

using namespace nnrt;
using namespace nnrt::op;

struct EventShell {
    EventShell()
        : event_(std::make_shared<Event>())
        {}
    int wait() { return event_->wait(); };
    EventPtr event() { return event_; }
private:
    EventPtr event_;
};

#define AERROR_CODE(CODE) (ANEURALNETWORKS_##CODE)
int translateErrorCode(int errCode)
{
    #define ERROR_CODE_TRANS(CODE)      \
        case NNA_ERROR_CODE(CODE):         \
            return AERROR_CODE(CODE);

    switch(errCode) {
        ERROR_CODE_TRANS(NO_ERROR);
        ERROR_CODE_TRANS(OUT_OF_MEMORY);
        ERROR_CODE_TRANS(INCOMPLETE);
        ERROR_CODE_TRANS(UNEXPECTED_NULL);
        ERROR_CODE_TRANS(BAD_DATA);
        ERROR_CODE_TRANS(OP_FAILED);
        ERROR_CODE_TRANS(BAD_STATE);
        ERROR_CODE_TRANS(UNMAPPABLE);

        default:
            return errCode;
    }
    #undef ERROR_CODE_TRANS
}

static OperationType mapOperationCode(OperationCode code)
{
    switch (code)
    {
#define REGISTER_OP(NAME)   do {        \
    case ANEURALNETWORKS_##NAME:        \
        return OperationType::NAME;     \
    } while(0)
        REGISTER_OP(ADD);
        REGISTER_OP(CONV_2D);
        REGISTER_OP(DEPTHWISE_CONV_2D);
        REGISTER_OP(RELU);
        REGISTER_OP(RESHAPE);
        REGISTER_OP(FULLY_CONNECTED);
        REGISTER_OP(TRANSPOSE);
        REGISTER_OP(SOFTMAX);
        REGISTER_OP(CONCATENATION);
        REGISTER_OP(AVERAGE_POOL_2D);
        REGISTER_OP(SQUEEZE);
        REGISTER_OP(MAX_POOL_2D);
        REGISTER_OP(PAD);
        REGISTER_OP(MUL);
        REGISTER_OP(MEAN);
        REGISTER_OP(RELU1);
        REGISTER_OP(RELU6);
        REGISTER_OP(TANH);
        REGISTER_OP(LOGISTIC);
        REGISTER_OP(FLOOR);
        REGISTER_OP(DIV);
        REGISTER_OP(SUB);
        REGISTER_OP(QUANTIZE);
        REGISTER_OP(DEQUANTIZE);
        REGISTER_OP(SPACE_TO_DEPTH);
        REGISTER_OP(DEPTH_TO_SPACE);
        REGISTER_OP(SPACE_TO_BATCH_ND);
        REGISTER_OP(BATCH_TO_SPACE_ND);
        REGISTER_OP(L2_NORMALIZATION);
        REGISTER_OP(RESIZE_BILINEAR);
        REGISTER_OP(LOCAL_RESPONSE_NORMALIZATION);
        REGISTER_OP(EMBEDDING_LOOKUP);
        REGISTER_OP(RNN);
        REGISTER_OP(HASHTABLE_LOOKUP);
        REGISTER_OP(LSTM);
        REGISTER_OP(SVDF);
        REGISTER_OP(LSH_PROJECTION);
        REGISTER_OP(L2_POOL_2D);
        REGISTER_OP(STRIDED_SLICE);
        REGISTER_OP(RESIZE_NEAREST_NEIGHBOR);
        REGISTER_OP(ABS);
        REGISTER_OP(ARGMAX);
        REGISTER_OP(ARGMIN);
        REGISTER_OP(EQUAL);
        REGISTER_OP(EXP);
        //REGISTER_OP(EXPAND_DIMS);
        REGISTER_OP(GATHER);
        REGISTER_OP(CHANNEL_SHUFFLE);
        REGISTER_OP(GREATER);
        REGISTER_OP(GREATER_EQUAL);
        REGISTER_OP(GROUPED_CONV_2D);
        REGISTER_OP(INSTANCE_NORMALIZATION);
        REGISTER_OP(LESS);
        REGISTER_OP(LESS_EQUAL);
        REGISTER_OP(LOGICAL_AND);
        REGISTER_OP(LOGICAL_OR);
        REGISTER_OP(MAXIMUM);
        REGISTER_OP(MINIMUM);
        REGISTER_OP(NEG);
        REGISTER_OP(NOT_EQUAL);
        REGISTER_OP(POW);
        REGISTER_OP(PRELU);
        REGISTER_OP(SIN);
        //REGISTER_OP(ROI_ALIGN);
        //REGISTER_OP(ROI_POOL);
        REGISTER_OP(RSQRT);
        REGISTER_OP(SELECT);
        //REGISTER_OP(SLICE);
        REGISTER_OP(SPLIT);
        REGISTER_OP(TRANSPOSE_CONV_2D);
        REGISTER_OP(REDUCE_ALL);
        REGISTER_OP(REDUCE_ANY);
        REGISTER_OP(REDUCE_SUM);
        REGISTER_OP(REDUCE_PROD);
        REGISTER_OP(REDUCE_MAX);
        REGISTER_OP(REDUCE_MIN);
        REGISTER_OP(AXIS_ALIGNED_BBOX_TRANSFORM);
        REGISTER_OP(GENERATE_PROPOSALS);
        REGISTER_OP(RANDOM_MULTINOMIAL);
        REGISTER_OP(LOG_SOFTMAX);
        REGISTER_OP(HEATMAP_MAX_KEYPOINT);
        REGISTER_OP(BOX_WITH_NMS_LIMIT);
        REGISTER_OP(TILE);
#undef REGISTER_OP
        case ANEURALNETWORKS_TOPK_V2:
            return OperationType::TOPK;
        default:
            break;
    }
    VSILOGW("Unknown operation code %d", code);
    return OperationType::NONE;
}

static OperandType mapOperandCode(OperandCode code)
{
    switch (code)
    {
        case ANEURALNETWORKS_BOOL:
            return OperandType::BOOL;
        case ANEURALNETWORKS_FLOAT16:
            return OperandType::FLOAT16;
        case ANEURALNETWORKS_FLOAT32:
            return OperandType::FLOAT32;
        case ANEURALNETWORKS_INT32:
            return OperandType::INT32;
        case ANEURALNETWORKS_UINT32:
            return OperandType::UINT32;
        case ANEURALNETWORKS_TENSOR_FLOAT32:
            return OperandType::TENSOR_FLOAT32;
        case ANEURALNETWORKS_TENSOR_INT32:
            return OperandType::TENSOR_INT32;
        case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            return OperandType::TENSOR_QUANT8_ASYMM;
        case ANEURALNETWORKS_TENSOR_QUANT8_SYMM:
            return OperandType::TENSOR_QUANT8_SYMM;
        case ANEURALNETWORKS_TENSOR_QUANT16_ASYMM:
            return OperandType::TENSOR_QUANT16_ASYMM;
        case ANEURALNETWORKS_TENSOR_QUANT16_SYMM:
            return OperandType::TENSOR_QUANT16_SYMM;
        case ANEURALNETWORKS_TENSOR_FLOAT16:
            return OperandType::TENSOR_FLOAT16;
        case ANEURALNETWORKS_TENSOR_BOOL8:
            return OperandType::TENSOR_BOOL8;
        case ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL:
            return OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL;
        default:
            break;
    }
    return OperandType::NONE;
}

static void _convert_operand_type(OperandPtr operand_type,
        const ANeuralNetworksOperandType* type)
{
    operand_type->type = mapOperandCode((OperandCode)type->type);
    operand_type->quant.scalar.scale = type->scale;
    operand_type->quant.scalar.zeroPoint = type->zeroPoint;
    if (type->dimensionCount > 0)
    {
        operand_type->dimensions.insert(operand_type->dimensions.begin(),
                type->dimensions, type->dimensions + type->dimensionCount);
    }

    // tensor shape with zero should set as Null, ovx won't create concrete tensor
    // for this operand
    //for (auto d: operand_type->dimensions) {
    //    if (0 == d) {
    //        operand_type->setNull();
    //        break;
    //    }
    //}
}

int ANeuralNetworksMemory_createFromFd(size_t size, int protect, int fd, size_t offset,
                                       ANeuralNetworksMemory** memory)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (nullptr == memory)
    {
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    *memory = nullptr;
    Memory* shared_memory   = new Memory();
    if (nullptr == shared_memory)
    {
        VSILOGW("New shared memory fail.");
        return AERROR_CODE(OUT_OF_MEMORY);
    }
    int err = translateErrorCode(shared_memory->readFromFd(size, protect, fd, offset));
    if (err != AERROR_CODE(NO_ERROR))
    {
        delete shared_memory;
    }
    else
    {
    *memory = reinterpret_cast<ANeuralNetworksMemory*>(shared_memory);
    }
    return err;
}

void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (nullptr == memory)
    {
        return;
    }
    Memory* shared_memory = reinterpret_cast<Memory*>(memory);
    delete shared_memory;
}

int ANeuralNetworksModel_create(ANeuralNetworksModel** model)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!model)
    {
        VSILOGW("Invalid mode pointer.");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    *model = nullptr;
    Model* m = new Model();
    if (m == nullptr)
    {
        return AERROR_CODE(OUT_OF_MEMORY);
    }
    *model = reinterpret_cast<ANeuralNetworksModel*>(m);

    return AERROR_CODE(NO_ERROR);
}

void ANeuralNetworksModel_free(ANeuralNetworksModel* model)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!model)
    {
        return;
    }
    Model* m = reinterpret_cast<Model*>(model);
    delete m;
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel* model)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!model)
    {
        VSILOGW("Pass null pointer to model.");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    Model* m = reinterpret_cast<Model*>(model);
    if(m->isFinished())
    {
        VSILOGW("Cannot modify a finished model.");
        return AERROR_CODE(BAD_STATE);
    }

    m->finish();
    return AERROR_CODE(NO_ERROR);
}

int ANeuralNetworksModel_addOperand(ANeuralNetworksModel* model,
                                    const ANeuralNetworksOperandType* type)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!model || !type)
    {
        VSILOGW("Passs null pointer to model or type");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    Model* m = reinterpret_cast<Model*>(model);
    if(m->isFinished())
    {
        VSILOGW("Cannot modify a finished model.");
        return AERROR_CODE(BAD_DATA);
    }

    uint32_t out_index;
    int err = AERROR_CODE(NO_ERROR);
    OperandPtr operand = m->addOperand(nullptr, &out_index);
    if (!operand)
    {
        err = AERROR_CODE(OUT_OF_MEMORY);
    }
    _convert_operand_type(operand, type);
    //operand->type = (OperandCode)type->type;
    //operand->dimensions.insert(operand->dimensions.begin(),
    //        type->dimensions, type->dimensions + type->dimensionCount);
    //operand->scale = type->scale;
    //operand->zeroPoint = type->zeroPoint;
    return err;
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model, int32_t index,
                                         const void* buffer, size_t length)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!model || !buffer)
    {
        VSILOGW("Passs null pointer to model or buffer");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    Model* m = reinterpret_cast<Model*>(model);
    if(m->isFinished())
    {
        VSILOGW("Cannot modify a finished model.");
        return AERROR_CODE(BAD_DATA);
    }
    return translateErrorCode(m->setOperandValue((uint32_t)index, buffer, length));
}

int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel* model, int32_t index,
                                                   const ANeuralNetworksMemory* memory,
                                                   size_t offset, size_t length)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!model || !memory)
    {
        VSILOGW("Passs null pointer to model or memory");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    Model* m = reinterpret_cast<Model*>(model);
    if(m->isFinished())
    {
        VSILOGW("Cannot modify a finished model.");
        return AERROR_CODE(BAD_DATA);
    }
    const Memory* shared_memory = reinterpret_cast<const Memory*>(memory);

    return translateErrorCode(m->setOperandValueFromMemory(index, shared_memory, offset, length));
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                      ANeuralNetworksOperationType type, uint32_t inputCount,
                                      const uint32_t* inputs, uint32_t outputCount,
                                      const uint32_t* outputs)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!model || !inputs || !outputs)
    {
        VSILOGW("Passs null pointer to model or inputs, outputs");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    Model* m = reinterpret_cast<Model*>(model);
    if(m->isFinished())
    {
        VSILOGW("Cannot modify a finished model.");
        return AERROR_CODE(BAD_DATA);
    }
    OperationPtr op = m->addOperation(mapOperationCode((OperationCode)type),
            inputs, inputCount, outputs, outputCount);
    if (!op)
    {
        return AERROR_CODE(OUT_OF_MEMORY);
    }
    op->setDataLayout(DataLayout::NHWC);
    return AERROR_CODE(NO_ERROR);
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel* model,
                                                  uint32_t inputCount, const uint32_t* inputs,
                                                  uint32_t outputCount, const uint32_t* outputs)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!model || !inputs || !outputs)
    {
        VSILOGW("Passs null pointer to model or inputs, outputs");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    Model* m = reinterpret_cast<Model*>(model);

    if(m->isFinished())
    {
        VSILOGW("Cannot modify a finished model.");
        return AERROR_CODE(BAD_DATA);
    }
    m->identifyInputsAndOutputs(inputs, inputCount, outputs, outputCount);
    if (!m->validate()) {
        return AERROR_CODE(BAD_DATA);
    }
    return AERROR_CODE(NO_ERROR);
}


int ANeuralNetworksCompilation_create(ANeuralNetworksModel* model,
                                      ANeuralNetworksCompilation** compilation)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!model || !compilation)
    {
        VSILOGW("Passs null pointer to model or compilation");
        return AERROR_CODE(UNEXPECTED_NULL);
    }

    Model* m = reinterpret_cast<Model*>(model);
    Compilation* c =  new Compilation(m);
    if (!c)
    {
        VSILOGW("Failt to new compilation.");
        return AERROR_CODE(OUT_OF_MEMORY);
    }

    // nnapi_delegate don't have to setup interpreter, the default interpreter
    // is designed for nnapi
    *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(c);
    return AERROR_CODE(NO_ERROR);
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation* compilation)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    Compilation* c = reinterpret_cast<Compilation*>(compilation);
    delete c;
}

int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation* compilation,
                                             int32_t preference)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
#if 0
    return  (reinterpret_cast<Compilation *>(compilation))->setPreference(preference);
#endif
    return 0;
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation* compilation)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    return  translateErrorCode((reinterpret_cast<Compilation *>(compilation))->run());
}

int ANeuralNetworksExecution_create(ANeuralNetworksCompilation* compilation,
                                    ANeuralNetworksExecution** execution)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);

    if (!compilation || !execution)
    {
        VSILOGW("Pass nullptr to compilation or execution.");
        return AERROR_CODE(UNEXPECTED_NULL);
    }

    Compilation* c = reinterpret_cast<Compilation*>(compilation);
    Execution* exec = new Execution(c);
    if(!exec)
    {
        VSILOGW("Out of memory.");
        return AERROR_CODE(OUT_OF_MEMORY);
    }

    *execution = reinterpret_cast<ANeuralNetworksExecution*>(exec);

    return AERROR_CODE(NO_ERROR);
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!execution)
    {
        return;
    }

    Execution * exec =reinterpret_cast<Execution*>(execution);
    delete exec;
}

int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution* execution, int32_t index,
                                      const ANeuralNetworksOperandType* type,
                                      const void* buffer, size_t length)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!execution)
    {
        VSILOGW("Pass nullptr to execution");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    Execution* exec = reinterpret_cast<Execution*>(execution);

    OperandPtr operand_ptr = std::make_shared<Operand>();
    if (type)
    {
        _convert_operand_type(operand_ptr, type);
    }
    return translateErrorCode(exec->setInput(index, operand_ptr, buffer, length));
}

int ANeuralNetworksExecution_setInputFromMemory(ANeuralNetworksExecution* execution,
                                                int32_t index,
                                                const ANeuralNetworksOperandType* type,
                                                const ANeuralNetworksMemory* memory,
                                                size_t offset, size_t length)
 {
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!execution)
    {
        VSILOGW("Pass nullptr to exection.");
        return AERROR_CODE(UNEXPECTED_NULL);
    }

    const Memory* shared_memory = reinterpret_cast<const Memory*>(memory);
    Execution* exec = reinterpret_cast<Execution*>(execution);

    OperandPtr operand_ptr = std::make_shared<Operand>();
    if (type)
    {
        _convert_operand_type(operand_ptr, type);
    }
    return translateErrorCode(exec->setInputFromMemory(index, operand_ptr, shared_memory, offset, length));
}


int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution* execution, int32_t index,
                                       const ANeuralNetworksOperandType* type, void* buffer,
                                       size_t length)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!execution)
    {
        VSILOGW("Pass nullptr to execution");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    Execution* exec = reinterpret_cast<Execution*>(execution);
    OperandPtr operand_ptr = std::make_shared<Operand>();
    if (type)
    {
        _convert_operand_type(operand_ptr, type);
    }
    return translateErrorCode(exec->setOutput(index, operand_ptr, buffer, length));
}

int ANeuralNetworksExecution_setOutputFromMemory(ANeuralNetworksExecution* execution,
                                                 int32_t index,
                                                 const ANeuralNetworksOperandType* type,
                                                 const ANeuralNetworksMemory* memory,
                                                 size_t offset, size_t length)
 {
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!execution)
    {
        VSILOGW("Pass nullptr to execution");
        return AERROR_CODE(UNEXPECTED_NULL);
    }

    Execution* exec = reinterpret_cast<Execution*>(execution);
    const Memory* shared_memory = reinterpret_cast<const Memory*>(memory);

    OperandPtr operand_ptr = std::make_shared<Operand>();
    if (type)
    {
        _convert_operand_type(operand_ptr, type);
    }
    return translateErrorCode(exec->setOutputFromMemory(index, operand_ptr, shared_memory, offset, length));
}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution* execution,
                                          ANeuralNetworksEvent** event)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);

    if (!event) {
        VSILOGW("Pass nullptr to event");
        return AERROR_CODE(UNEXPECTED_NULL);
    }

    *event = nullptr;
    if (!execution) {
        VSILOGW("Pass nullptr to execution");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    EventShell* clbk = new EventShell();
    if (!clbk)
    {
        return AERROR_CODE(OUT_OF_MEMORY);
    }

    Execution* exec = reinterpret_cast<Execution*>(execution);

    int error = translateErrorCode(exec->startCompute(clbk->event()));
    if (error == AERROR_CODE(NO_ERROR))
    {
        *event = reinterpret_cast<ANeuralNetworksEvent*>(clbk);
    }
    else
    {
        delete clbk;
    }

    return error;
}


int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!event)
    {
        VSILOGW("Pass nullptr to event.");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    return  (reinterpret_cast<EventShell*> (event))->wait();
}


void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if(!event)
    {
        return ;
    }
    EventShell* e = reinterpret_cast<EventShell *>(event);
    e->wait();
    delete e;
}

// it is always relaxed in our driver.
int ANeuralNetworksModel_relaxComputationFloat32toFloat16(ANeuralNetworksModel* model, bool allow)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!model)
    {
        return AERROR_CODE(BAD_DATA);
    }

    return AERROR_CODE(NO_ERROR);
}

int ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(
        ANeuralNetworksModel* model, int32_t index,
        const ANeuralNetworksSymmPerChannelQuantParams* channelQuant)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!model || !channelQuant) {
        return AERROR_CODE(UNEXPECTED_NULL);
    }

    // TODO: Add implementation
    /*
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    return m->setOperandSymmPerChannelQuantParams(index, *channelQuant);
    */
    return AERROR_CODE(INCOMPLETE);
}

int ANeuralNetworksExecution_compute(ANeuralNetworksExecution* execution)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!execution) {
        VSILOGW("Pass nullptr to execution");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    Execution* exec = reinterpret_cast<Execution*>(execution);

    return exec->compute();
}

int ANeuralNetworksBurst_create(ANeuralNetworksCompilation* compilation,
                                ANeuralNetworksBurst** burst)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!compilation || !burst) {
        return AERROR_CODE(UNEXPECTED_NULL);
    }

    // TODO: Add implementation
    /*
    CompilationBuilder* c = reinterpret_cast<CompilationBuilder*>(compilation);
    BurstBuilder* b = nullptr;
    int result = c->createBurst(&b);
    *burst = reinterpret_cast<ANeuralNetworksBurst*>(b);
    return result;
    */
    return AERROR_CODE(INCOMPLETE);
}

void ANeuralNetworksBurst_free(ANeuralNetworksBurst* burst)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    // No validation.  Free of nullptr is valid.
    // TODO: Add implementation
    /*
    BurstBuilder* b = reinterpret_cast<BurstBuilder*>(burst);
    delete b;
    */
}

int ANeuralNetworksExecution_burstCompute(ANeuralNetworksExecution* execution,
                                          ANeuralNetworksBurst* burst)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!execution || !burst) {
        return AERROR_CODE(UNEXPECTED_NULL);
    }

    // TODO: Add implementation
    /*
    ExecutionBuilder* r = reinterpret_cast<ExecutionBuilder*>(execution);
    BurstBuilder* b = reinterpret_cast<BurstBuilder*>(burst);

    if (r->getCompilation() != b->getCompilation()) {
        LOG(ERROR) << "ANeuralNetworksBurst and ANeuralNetworksExecution "
                      "used in ANeuralNetworksExecution_burstCompute must "
                      "originate from the same ANeuralNetworksCompilation";
        return ANEURALNETWORKS_BAD_DATA;
    }

    const bool locked = b->tryLock();
    if (!locked) {
        LOG(ERROR) << "ANeuralNetworksBurst is already being used in another "
                      "call to ANeuralNetworksExecution_burstCompute";
        return ANEURALNETWORKS_BAD_STATE;
    }

    const int n = r->burstCompute(b);
    b->unlock();

    return n;
    */
    return AERROR_CODE(INCOMPLETE);
}

int ANeuralNetworksExecution_getOutputOperandRank(ANeuralNetworksExecution* execution,
                                                  int32_t index, uint32_t* rank) {
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!execution) {
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    Execution* exec = reinterpret_cast<Execution*>(execution);
    return exec->getOutputOperandRank(index, rank);
}

int ANeuralNetworksExecution_getOutputOperandDimensions(ANeuralNetworksExecution* execution,
                                                        int32_t index, uint32_t* dimensions)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if (!execution) {
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    Execution* exec = reinterpret_cast<Execution*>(execution);
    return exec->getOutputOperandDimensions(index, dimensions);
}

#if defined(ANDROID_STUB)
int ANeuralNetworksMemory_createFromAHardwareBuffer(const AHardwareBuffer* ahwb,
                                                    ANeuralNetworksMemory** memory)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);


    // TODO: Add implementation
    /*
    *memory = nullptr;
    std::unique_ptr<MemoryAHWB> m = std::make_unique<MemoryAHWB>();
    if (m == nullptr) {
        return ANEURALNETWORKS_OUT_OF_MEMORY;
    }
    int n = m->set(ahwb);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    *memory = reinterpret_cast<ANeuralNetworksMemory*>(m.release());
    return ANEURALNETWORKS_NO_ERROR;
    */
    return AERROR_CODE(INCOMPLETE);
}
#endif
