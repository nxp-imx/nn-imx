#ifndef __OVXLIB_PREPARED_MODEL_H__
#define __OVXLIB_PREPARED_MODEL_H__

#include <vector>
#include "vsi_nn_pub.h"
#include "model.h"

namespace ovxlib
{
class PreparedModel
{
    public:
        PreparedModel(Model* model);
        ~PreparedModel();

        vsi_nn_graph_t* get() {return graph_;}

        int prepare(vsi_nn_context_t context);

        int execute();

        int setInput(uint32_t index, const void* data, size_t length);

        int getOutput(uint32_t index, void* data, size_t length);

        int updateOutputOperand(uint32_t index, const Operand* operand_type);

        int updateInputOperand(uint32_t index, const Operand* operand_type);

    private:
        vsi_nn_graph_t* graph_{nullptr};
        Model* model_{nullptr};
        std::map<uint32_t, vsi_nn_tensor_id_t> tensor_mapping_;

};
}

#endif
