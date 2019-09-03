#include "vsi_nn_pub.h"
#include "prepared_model.h"
#include "error.h"
#include "ovxlib_delegate.h"
#include "graph_transformations/transformations.h"

namespace ovxlib
{
PreparedModel::PreparedModel(Model* model)
    : model_(model)
{}

PreparedModel::~PreparedModel()
{
    vsi_nn_ReleaseGraph(&graph_);
}

int PreparedModel::prepare(vsi_nn_context_t context)
{
    if (graph_) {
        return AERROR_CODE(NO_ERROR);
    }
    VSILOGD("Prepare model ...");
    // Compile
    int err = AERROR_CODE(NO_ERROR);
    if (!model_->isCompiled()) {
        TransformationSet transformations;
        // NOTE: The transformation order is important.
        model_->echo();
        transformations.add(new NnApiInterpreter());
        transformations.add(new AlignBroadcastOp());
        transformations.add(new T2C());
        transformations.add(new OptimizePermute());
        transformations.add(new ValidateQuantizedGraph());

        // relaxed mode
        if (model_->isRelaxed())
            transformations.add(new Fp32ToFp16());

        err = transformations.once(model_);
        model_->freezeCompile();
        if (err != AERROR_CODE(NO_ERROR)) {
            return err;
        }
        model_->echo();
    }
    // Build ovxlib graph
    OvxlibDelegate delegate;
    err = delegate.process(model_, context);
    tensor_mapping_ = delegate.getTensorMapping();
    graph_ = delegate.throwGraph();
    if (err != AERROR_CODE(NO_ERROR)) {
        VSILOGW("Prepare graph fail.");
        vsi_nn_ReleaseGraph(&graph_);
    }
    return err;
}

int PreparedModel::execute()
{
    int error = AERROR_CODE(OP_FAILED);
    if (graph_) {
        //vsi_nn_PrintGraph(graph_);
        vsi_status run_state = vsi_nn_RunGraph(graph_); if (VSI_SUCCESS == run_state) {
            error = AERROR_CODE(NO_ERROR);
            //vsi_nn_DumpGraphNodeOutputs(graph_, "nodes", nullptr, 0, false, VSI_NN_DIM_FMT_NCHW);
        }
        VSILOGD("Process complete state: %d.", run_state);
    }
    return error;
}

int PreparedModel::setInput(uint32_t index, const void* data, size_t length)
{
    //bool input_mapped_as_vx_tensor = tensor_mapping_.find(index) != tensor_mapping_.end();

    //if (!input_mapped_as_vx_tensor) {
    //    VSILOGD("Should not set input for no-value optional parameter");
    //    return AERROR_CODE(OP_FAILED);
    //}

    //index = tensor_mapping_[index];

    if (index >= graph_->input.num) {
        return AERROR_CODE(OP_FAILED);
    }
    vsi_nn_tensor_t* tensor = vsi_nn_GetTensor(graph_, graph_->input.tensors[index]);
    if (!tensor) {
        // TODO: Optional tensor.
        return AERROR_CODE(NO_ERROR);
    } else if (!data || !length) {
        return AERROR_CODE(OP_FAILED);
    } else {
        uint32_t tensor_size = vsi_nn_GetTensorSize(tensor->attr.size, tensor->attr.dim_num,
                tensor->attr.dtype.vx_type);
        if (length != tensor_size) {
            VSILOGW("Tensor size mismatch %u vs %u.", tensor_size, length);
            return AERROR_CODE(OP_FAILED);
        }
        vsi_nn_CopyDataToTensor(graph_, tensor, (uint8_t*)data);
    }
    return AERROR_CODE(NO_ERROR);
}

int PreparedModel::getOutput(uint32_t index, void* data, size_t length)
{
    if (index >= graph_->input.num || !data || !length) {
        return AERROR_CODE(OP_FAILED);
    }
    vsi_nn_tensor_t* tensor = vsi_nn_GetTensor(graph_, graph_->output.tensors[index]);
    if (!tensor) {
        return AERROR_CODE(OP_FAILED);
    } else {
        uint32_t tensor_size = vsi_nn_GetTensorSize(tensor->attr.size, tensor->attr.dim_num,
                tensor->attr.dtype.vx_type);
        if (length != tensor_size) {
            VSILOGW("Tensor size mismatch %u vs %u.", tensor_size, length);
            return AERROR_CODE(OP_FAILED);
        }
        if (VSI_SUCCESS != vsi_nn_CopyTensorToBuffer(graph_, tensor, (uint8_t*)data)) {
            return AERROR_CODE(OP_FAILED);
        }
    }
    return AERROR_CODE(NO_ERROR);
}

int PreparedModel::updateOutputOperand(uint32_t index, const Operand* operand_type)
{
    uint32_t operand_index = model_->outputIndex(index);
    return model_->updateOperand(operand_index, operand_type);
}

int PreparedModel::updateInputOperand(uint32_t index, const Operand* operand_type)
{
    uint32_t operand_index = model_->inputIndex(index);
    return model_->updateOperand(operand_index, operand_type);
}

}

