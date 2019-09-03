#include <vector>
#include <assert.h>
#include "model.h"
#include "error.h"
#include "operand.h"
#include "operation.h"
#include "graph_transformations/transformations.h"

namespace ovxlib
{
static int32_t convert_t2c_4d_axis(int32_t axis)
{
    int32_t perm[] = {0,2,3,1};
    return perm[axis];
}

template<typename T>
static void convert_t2c_4d_array(T* dst, T* src, size_t length)
{
    int32_t perm[] = {0,3,1,2};
    for (size_t i = 0; i < length; ++i) {
        dst[i] = src[perm[i]];
    }
}

template<typename T>
static void convert_t2c_array(std::vector<T> &array)
{
    std::vector<T> tmp(array.size());
    if (array.size() == 4) {
        convert_t2c_4d_array(tmp.data(), array.data(), array.size());
        array.assign(tmp.begin(), tmp.end());
    }
}

template<typename F>
static void mark_4d_tensors(
    std::vector<Operand*> operands, int type, Operation * operation, F &func)
{
    for (uint32_t i = 0; i < operands.size(); ++i) {
        if (nullptr != operands[i] && operands[i]->isTensor()) {
            if (operands[i]->ndim() == 4) {
                func(operands[i], type, operation);
            }
        }
    }
}

template<typename F>
static F permute_vector(F &array, std::vector<uint32_t> &perm, int type)
{
    F new_array;
    new_array.resize(perm.size());
    for (uint32_t i = 0; i < perm.size(); ++i) {
        if (MARK_DATA_DIRECTION_PRODUCER == type) {
            new_array[perm[i]] = array[i];
        }
        else {
            new_array[i] = array[perm[i]];
        }
    }
    return new_array;
}

static void reduce_mark_data(T2CMarkData *data)
{
    auto merge_perm = [](std::vector<std::vector<uint32_t>> &perms)
        -> std::vector<uint32_t> {
        std::vector<uint32_t> out = perms[0];
        std::vector<uint32_t> tmp;
        out.resize(perms[0].size());
        for (size_t i = 1; i < perms.size(); ++i) {
            tmp.resize(out.size());
            std::vector<uint32_t> &perm = perms[i];
            for (size_t j = 0; j < out.size(); ++j) {
                tmp[j] = out[perm[j]];
            }
            out = tmp;
            tmp.clear();
        }
        return out;
    };
    /* Optimzie mark data */
    auto is_unused_perm = [](std::vector<uint32_t> &perm) -> bool {
        bool unused = true;
        for (size_t i = 0; i < perm.size(); ++i) {
            if (i != perm[i]) {
                unused = false;
                break;
            }
        }
        return unused;
    };

    /* Reduce consumer first */
    std::vector<Operation*>  unused_perm;
    std::map<Operation*, std::vector<uint32_t>> &consumers = data->consumers;
    for (auto it = consumers.begin(); it != consumers.end(); ++it) {
        if (data->producers.size() > 0) {
            std::vector<uint32_t> &producer_perm = data->producers.begin()->second;
            std::vector<std::vector<uint32_t>> perms;
            perms.push_back(producer_perm);
            perms.push_back(it->second);
            it->second = merge_perm(perms);
        }
        if (is_unused_perm(it->second)) {
            unused_perm.push_back(it->first);
        }
    }
    if (consumers.size() > 0 && data->producers.size() > 0) {
        data->producers.clear();
    }
    for (size_t i = 0; i < unused_perm.size(); ++i) {
        consumers.erase(unused_perm[i]);
    }

    /* If only producer */
    unused_perm.clear();
    for (auto it = data->producers.begin(); it != data->producers.end(); ++it) {
        if (is_unused_perm(it->second)) {
            unused_perm.push_back(it->first);
        }
    }
    for (size_t i = 0; i < unused_perm.size(); ++i) {
        data->producers.erase(unused_perm[i]);
    }
}

static void insert_permute(
    Model *model, Operand *operand, Operation *operation, std::vector<uint32_t> &perm, int type)
{
    // New operand
    int new_operand_index = -1;
    int org_operand_index = model->getOperandIndex(operand);
    if (org_operand_index < 0) {
        assert(false);
    }
    Operand *new_output = model->cloneOperand(operand, &new_operand_index);

    // New operation
    uint32_t inputs[1] = {0};
    uint32_t outputs[1] = {0};
    if (MARK_DATA_DIRECTION_PRODUCER == type) {
        // Operand->dimensions has been transposed as producer type before.
        operand->dimensions = permute_vector(operand->dimensions, perm,
            MARK_DATA_DIRECTION_CONSUMER);
        inputs[0] = {(uint32_t)new_operand_index};
        outputs[0] = {(uint32_t)org_operand_index};
        operation->replaceOutputs(org_operand_index, new_operand_index);
    }
    else {
        new_output->dimensions = permute_vector(new_output->dimensions, perm, type);
        inputs[0] = { (uint32_t)org_operand_index };
        outputs[0] = { (uint32_t)new_operand_index };
        operation->replaceInputs(org_operand_index, new_operand_index);
    }

    int permute_index = -1;
    PermuteOperation* permute = new PermuteOperation();
    permute->perm.assign(perm.begin(), perm.end());
    permute->setInputs(inputs, 1);
    permute->setOutputs(outputs, 1);
    model->addOperation(permute, &permute_index);
}

static void apply_permute_to_const_operand(Model *model, Operand *operand, T2CMarkData *data)
{
    /* NOTE: If different operations share the same const operand may cause problems. */
    std::vector<uint32_t> &perm = data->consumers.begin()->second;
    if (operand->isTensor()) {
        if (operand->ndim() == perm.size()) {
            operand->setPerm(perm);
            operand->dimensions = permute_vector(operand->dimensions, perm,
                MARK_DATA_DIRECTION_CONSUMER);
        }
        else {
            VSILOGW("Unhandle issue, can not convert const operand.");
            assert(false);
        }
    }
    else {
        VSILOGW("Unsupport permute const operand type %d.", (int)operand->type);
        assert(false);
    }
}

static void insert_permute_with_mark_data(Model *model, Operand *operand, T2CMarkData *data)
{
    if (data->producers.size() > 0) {
        Operation *operation = data->producers.begin()->first;
        std::vector<uint32_t> &perm = data->producers.begin()->second;
        insert_permute(model, operand, operation, perm, MARK_DATA_DIRECTION_PRODUCER);
    }
    for (auto it = data->consumers.begin(); it != data->consumers.end(); ++it) {
        Operation *operation = it->first;
        std::vector<uint32_t> &perm = it->second;
        if (operand->isNull()) {
            continue;
        }
        insert_permute(model, operand, operation, perm, MARK_DATA_DIRECTION_CONSUMER);
    }
}

static void apply_mark_data(Model * model, std::map<Operand*, T2CMarkData*> mark_data)
{
    for (auto it = mark_data.begin(); it != mark_data.end(); ++it) {
        Operand* operand = it->first;
        T2CMarkData* data = it->second;
        reduce_mark_data(data);
        if (operand->isConst() || !operand->isTensor()) {
            apply_permute_to_const_operand(model, operand, data);
        }
        else {
            insert_permute_with_mark_data(model, operand, data);
        }
    }
}

static void mark_tensors(Model * model, std::map<Operand*, T2CMarkData*> &mark_data)
{
    //std::map<Operand*, T2CMarkData*> mark_data;
    auto get_mark_data = [&mark_data](Operand *operand) -> T2CMarkData * {
        if (mark_data.find(operand) == mark_data.end()) {
            mark_data[operand] = new T2CMarkData();
        }
        return mark_data[operand];
    };
    auto add_mark_data = [&mark_data, get_mark_data](Operand * operand,
            int type, Operation * operation, std::vector<uint32_t> perm) {
        T2CMarkData * mark_data = get_mark_data(operand);
        if (type == MARK_DATA_DIRECTION_PRODUCER) {
            // Operand should have only one producer.
            mark_data->producers[operation] = perm;
            operand->dimensions = permute_vector(operand->dimensions, perm,
                MARK_DATA_DIRECTION_PRODUCER);
        }
        else if (type == MARK_DATA_DIRECTION_CONSUMER) {
            mark_data->consumers[operation] = perm;
        }
    };
    auto mark_no_convert = [add_mark_data](Operand * operand, int type,
            Operation * operation) {
        std::vector<uint32_t> perm;
        for (uint32_t i = 0; i < operand->ndim(); ++i) {
            perm.push_back(i);
        }
        add_mark_data(operand, type, operation, perm);
    };
    auto force_mark_4d_t2c = [add_mark_data](
            Operand * operand, int type, Operation * operation) {
        std::vector<uint32_t> perm = {0,3,1,2};
        add_mark_data(operand, type, operation, perm);
    };
    auto force_mark_4d_c2t = [add_mark_data](
            Operand * operand, int type, Operation * operation) {
        std::vector<uint32_t> perm = {0,2,3,1};
        add_mark_data(operand, type, operation, perm);
    };
    auto mark_4d_t2c_if_necessary = [force_mark_4d_t2c](
            Operand *operand, int type, Operation * operation) {
        if (operand->ndim() == 4) {
            force_mark_4d_t2c(operand, type, operation);
        }
    };
    auto mark_4d_c2t_if_necessary = [force_mark_4d_c2t](
        Operand *operand, int type, Operation * operation) {
        if (operand->ndim() == 4) {
            force_mark_4d_c2t(operand, type, operation);
        }
    };
    auto mark_3d_c2t_if_necessary = [add_mark_data](
        Operand *operand, int type, Operation *operation) {
        if (operand->ndim() == 3) {
            std::vector<uint32_t> perm = {0,2,1};
            add_mark_data(operand, type, operation, perm);
        }
    };

    auto operations = model->operations();
    for (auto it = operations.begin(); it != operations.end(); ++it) {
        Operation * operation = it->second;
        std::vector<Operand*> inputs = model->getOperands(operation->inputs());
        std::vector<Operand*> outputs = model->getOperands(operation->outputs());
        switch (operation->type()) {
            case OperationType::RESHAPE:
            case OperationType::TRANSPOSE:
                mark_no_convert(inputs[0], MARK_DATA_DIRECTION_CONSUMER, operation);
                mark_no_convert(outputs[0], MARK_DATA_DIRECTION_PRODUCER, operation);
                break;
            case OperationType::SQUEEZE:
                if (inputs[0]->ndim() == 4) {
                    mark_no_convert(inputs[0], MARK_DATA_DIRECTION_CONSUMER, operation);
                }
                if (outputs[0]->ndim() == 4) {
                    mark_no_convert(outputs[0], MARK_DATA_DIRECTION_PRODUCER, operation);
                }
                break;
            case OperationType::CONCATENATION:
            {
                if (inputs[0]->ndim() == 4) {
                    ConcatOperation *concat = reinterpret_cast<ConcatOperation*>(operation);
                    concat->axis = convert_t2c_4d_axis(concat->axis);
                    mark_4d_tensors(inputs, MARK_DATA_DIRECTION_CONSUMER, operation,
                        mark_4d_t2c_if_necessary);
                    mark_4d_c2t_if_necessary(outputs[0], MARK_DATA_DIRECTION_PRODUCER, operation);
                }
                break;
            }
            case OperationType::MEAN:
            {
                MeanOperation *mean = reinterpret_cast<MeanOperation*>(operation);
                bool reduce_channel = true;
                if (inputs[0]->ndim() == 4) {
                    std::vector<int32_t> nchw_axes(mean->axes.size());
                    for (size_t i = 0; i < mean->axes.size(); ++i) {
                        nchw_axes[i] = convert_t2c_4d_axis(mean->axes[i]);
                        if (mean->axes[i] < 3) {
                            reduce_channel = false;
                        }
                    }
                    mean->axes.assign(nchw_axes.begin(), nchw_axes.end());
                    mark_4d_t2c_if_necessary(inputs[0], MARK_DATA_DIRECTION_CONSUMER, operation);
                }
                if (outputs[0]->ndim() == 4) {
                    mark_4d_c2t_if_necessary(outputs[0], MARK_DATA_DIRECTION_PRODUCER, operation);
                }
                else if (outputs[0]->ndim() == 3 && !reduce_channel) {
                    mark_3d_c2t_if_necessary(outputs[0], MARK_DATA_DIRECTION_PRODUCER, operation);
                }
                break;
            }
            case OperationType::PAD:
            {
                PadOperation *pad = reinterpret_cast<PadOperation*>(operation);
                std::vector<Operand*> inputs = model->getOperands(pad->inputs());
                std::vector<Operand*> outputs = model->getOperands(pad->outputs());
                mark_4d_t2c_if_necessary(inputs[0], MARK_DATA_DIRECTION_CONSUMER, operation);
                mark_4d_c2t_if_necessary(outputs[0], MARK_DATA_DIRECTION_PRODUCER, operation);
                convert_t2c_array(pad->padFront);
                convert_t2c_array(pad->padBack);
                break;
            }
            case OperationType::STRIDED_SLICE:
            {
                StridedSliceOperation *stride_slice =
                    reinterpret_cast<StridedSliceOperation*>(operation);
                if (inputs[0]->ndim() == 4) {
                    mark_4d_t2c_if_necessary(inputs[0], MARK_DATA_DIRECTION_CONSUMER, operation);
                    convert_t2c_array(stride_slice->starts);
                    convert_t2c_array(stride_slice->ends);
                    convert_t2c_array(stride_slice->strides);
                    //TODO: convert mask
                }
                if (outputs[0]->ndim() == 4) {
                    mark_4d_c2t_if_necessary(outputs[0], MARK_DATA_DIRECTION_PRODUCER, operation);
                }
                //else{
                // TODO: handle reduce dimension
                //}
                break;
            }
            case OperationType::FULLY_CONNECTED:
            {
                if (inputs[0]->ndim() == 4) {
                    mark_no_convert(inputs[0], MARK_DATA_DIRECTION_CONSUMER, operation);
                }
                break;
            }
            default:
                mark_4d_tensors(inputs, MARK_DATA_DIRECTION_CONSUMER, operation,
                    mark_4d_t2c_if_necessary);
                mark_4d_tensors(outputs, MARK_DATA_DIRECTION_PRODUCER, operation,
                    mark_4d_c2t_if_necessary);
                break;
        }
    }
}

int T2C::run(Model * model, bool * modified)
{
    if (nullptr == model) {
        return AERROR_CODE(NO_ERROR);
    }

    *modified = true;

    std::map<Operand*, T2CMarkData*> mark_data;
    mark_tensors(model, mark_data);
    apply_mark_data(model, mark_data);

    return AERROR_CODE(NO_ERROR);
}

}
