#include "vsi_nn_pub.h"
#include "NeuralNetworks.h"
#include "event.h"
#include "compilation.h"
#include "execution.h"
#include "model.h"
#include "file_map_memory.h"
#include "error.h"
#include "types.h"

using namespace ovxlib;

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
#undef REGISTER_OP
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
        default:
            break;
    }
    return OperandType::NONE;
}

static void _convert_operand_type(Operand* operand_type,
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
    int err = shared_memory->readFromFd(size, protect, fd, offset);
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

    int out_index;
    int err = AERROR_CODE(NO_ERROR);
    Operand * operand = m->addOperand(nullptr, &out_index);
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
    return m->setOperandValue((uint32_t)index, buffer, length);
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

    return m->setOperandValueFromMemory(index, shared_memory, offset, length);
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
    Operation * op = m->addOperation(mapOperationCode((OperationCode)type),
            inputs, inputCount, outputs, outputCount);
    if (!op)
    {
        return AERROR_CODE(OUT_OF_MEMORY);
    }
    op->setOperandLayout(OperandLayout::NHWC);
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
    return  (reinterpret_cast<Compilation *>(compilation))->run();
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
    if( exec->isRunning())
    {
        exec->quit();
    }
    else
    {
        delete exec;
    }
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
    Operand operand_type;
    Operand * operand_ptr = nullptr;
    if (type)
    {
        _convert_operand_type(&operand_type, type);
        operand_ptr = &operand_type;
    }
    return exec->setInput(index, operand_ptr, buffer, length);
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
    Operand operand_type;
    Operand * operand_ptr = nullptr;
    if (type)
    {
        _convert_operand_type(&operand_type, type);
        operand_ptr = &operand_type;
    }
    return exec->setInputFromMemory(index, operand_ptr, shared_memory, offset, length);
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
    Operand operand_type;
    Operand * operand_ptr = nullptr;
    if (type)
    {
        _convert_operand_type(&operand_type, type);
        operand_ptr = &operand_type;
    }
    return exec->setOutput(index, operand_ptr, buffer, length);
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
    Operand operand_type;
    Operand * operand_ptr = nullptr;
    if (type)
    {
        _convert_operand_type(&operand_type, type);
        operand_ptr = &operand_type;
    }
    return exec->setOutputFromMemory(index, operand_ptr, shared_memory, offset, length);
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
    Event* clbk = new Event();
    if (!clbk)
    {
        return AERROR_CODE(OUT_OF_MEMORY);
    }

    Execution* exec = reinterpret_cast<Execution*>(execution);

    int error = exec->startCompute(clbk);
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
    return  (reinterpret_cast<Event*> (event))->wait();
}


void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event)
{
    VSILOGD("%s: %d", __FUNCTION__, __LINE__);
    if(!event)
    {
        return ;
    }
    Event* e = reinterpret_cast<Event *>(event);
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

//    Model* m = reinterpret_cast<Model*>(model);
//    m->relax(allow);

    return AERROR_CODE(NO_ERROR);
}

