#ifndef __OVXLIB_DELEGATE_H__
#define __OVXLIB_DELEGATE_H__

#include "vsi_nn_pub.h"
#include "model.h"

namespace ovxlib
{
class OvxlibDelegate
{
    public:
        OvxlibDelegate();
        virtual ~OvxlibDelegate();

        int process(Model* model, vsi_nn_context_t ctx = nullptr);

        vsi_nn_graph_t* throwGraph();

        std::map<uint32_t, vsi_nn_tensor_id_t> getTensorMapping()const;

        vsi_nn_pad_e getPaddingType(PadType type);

        vsi_nn_round_type_e getRoundType(Rounding type);

        vsi_nn_op_t getActivation(FusedType fused_code);

        vsi_nn_type_e mapTensorType(OperandType code);

        int addNode(vsi_nn_op_t op,
                std::vector<uint32_t> & inputs,
                std::vector<uint32_t> & outputs, FusedType fused_code,
                std::vector<vsi_nn_node_t*>* output_nodes, uint32_t uid);

        inline int addNode(vsi_nn_op_t op, Operation* operation,
                std::vector<vsi_nn_node_t*>* output_nodes, uint32_t uid) {
            return addNode(op, operation->inputs(),
                    operation->outputs(), operation->fusedType(), output_nodes, uid);
        }

        inline bool hasFusedCode(FusedType code) {
            return (code != FusedType::NONE);
        }

        void packTensorAttr(vsi_nn_tensor_attr_t* attr,
            vsi_nn_type_e dtype, std::vector<uint32_t> & nchw_shape,
            bool is_quantized, float scale, int32_t zero_point,
            TensorLifeTime type);

        void packTensorAttr(vsi_nn_tensor_attr_t* attr,
            Operand* operand, TensorLifeTime type);

        void mapTensorId(uint32_t operand_id, vsi_nn_tensor_id_t tensor_id);

        int addTensor(vsi_nn_graph_t* graph, Operand* operand,
                TensorLifeTime type, size_t idx, const void* data = nullptr);

        int addTensor(vsi_nn_graph_t* graph, vsi_nn_type_e dtype,
            std::vector<uint32_t> & shape, bool is_quantized,
            float scale, int32_t zero_point, TensorLifeTime type, size_t idx,
            const void* data = nullptr);

        int addTensor(vsi_nn_graph_t* graph,
            vsi_nn_tensor_attr_t* attr, size_t idx,
            const void* data = nullptr);

        inline uint32_t newNodeUid() {
            node_unique_id_ --;
            return node_unique_id_;
        }

        vsi_nn_pad_mode_e mapPadMode(PadMode mode);

        std::vector<uint32_t> reorderOperands(
                std::vector<uint32_t>& operands, std::vector<int> order);

        std::vector<vsi_nn_tensor_id_t> getMappedTensors(
                std::vector<uint32_t> & operand_indexes);

        vsi_nn_tensor_id_t getMappedTensor(uint32_t operand_index);

        vsi_nn_lsh_projection_type_e mapLshProjectionType(LshProjectionType type);

        void fillVxParam(vsi_nn_vx_param_t* c_vx_param, VxParam& vx_param);

        template<typename T>
        std::vector<T> reverseArray(std::vector<T> &data)
        {
            std::vector<T> buf(data.size());
            buf.assign(data.rbegin(), data.rend());
            return buf;
        };

        template<typename T>
        T *addParamPool(std::vector<T> &data, bool reverse = false)
        {
            std::vector<int8_t> handler(data.size() * sizeof(T));
            if (reverse == false) {
                memcpy(handler.data(), data.data(), data.size() * sizeof(T));
            }
            else {
                std::vector<T> reverse_buf = reverseArray(data);
                memcpy(handler.data(), reverse_buf.data(), data.size() * sizeof(T));
            }
            size_pool_.push_back(handler);
            return reinterpret_cast<T*>(size_pool_.back().data());
        };

        template<typename T>
        std::vector<T> convertPermute(std::vector<T> &perm)
        {
            return convertAxes(perm, perm.size());
        };

        template<typename T>
        std::vector<T> convertAxes(std::vector<T> &axes, size_t dim_num)
        {
            std::vector<T> new_axes(axes.size());
            size_t max_size = axes.size() - 1;
            for (size_t i = 0; i < axes.size(); ++i) {
                new_axes[i] = convertAxis(axes[max_size - i], dim_num);
            }
            return new_axes;
        };

        int32_t convertAxis(int32_t axis, int32_t dim_num)
        {
            if (axis < 0) {
                axis += dim_num;
            }
            return (dim_num - axis - 1);
        };

        int32_t reverseMask(int32_t mask, size_t dim_num)
        {
            auto get_bit_in_mask = [](int mask, int index) -> int {
                return (((int)0x1) << index) & mask;
            };
            int32_t new_mask = 0;
            for (int i = (int)dim_num - 1; i >= 0; --i) {
                new_mask |= (get_bit_in_mask(mask, i) >> i) << ((dim_num - 1) - i);
            }
            return new_mask;
        };

#define REGISTER_OP(NAME)   \
        int addNode_##NAME(Model* model, Operation* operation, uint32_t)
        REGISTER_OP(ADD);
        REGISTER_OP(CONV_2D);
        REGISTER_OP(DEPTHWISE_CONV_2D);
        REGISTER_OP(RELU);
        REGISTER_OP(RESHAPE);
        REGISTER_OP(FULLY_CONNECTED);
        REGISTER_OP(TRANSPOSE);
        REGISTER_OP(CONCATENATION);
        REGISTER_OP(AVERAGE_POOL_2D);
        REGISTER_OP(SQUEEZE);
        REGISTER_OP(SOFTMAX);
        REGISTER_OP(MAX_POOL_2D);
        REGISTER_OP(PAD);
        REGISTER_OP(MUL);
        REGISTER_OP(MEAN);
        REGISTER_OP(RELU1);
        REGISTER_OP(RELU6);
        REGISTER_OP(SIGMOID);
        REGISTER_OP(TANH);
        REGISTER_OP(DIV);
        REGISTER_OP(SUB);
        REGISTER_OP(DEQUANTIZE);
        REGISTER_OP(SPACE_TO_DEPTH);
        REGISTER_OP(DEPTH_TO_SPACE);
        REGISTER_OP(SPACE_TO_BATCH_ND);
        REGISTER_OP(BATCH_TO_SPACE_ND);
        REGISTER_OP(L2_NORM);
        REGISTER_OP(RESIZE_BILINEAR);
        REGISTER_OP(LOCAL_RESPONSE_NORM);
        REGISTER_OP(STRIDED_SLICE);
        REGISTER_OP(EMBEDDING_LOOKUP);
        REGISTER_OP(RNN);
        REGISTER_OP(HASHTABLE_LOOKUP);
        REGISTER_OP(LSTM);
        REGISTER_OP(SVDF);
        REGISTER_OP(LSH_PROJECTION);
        REGISTER_OP(L2_POOL_2D);
        REGISTER_OP(FLOOR);

        REGISTER_OP(DATA_CONVERT);
#undef  REGISTER_OP

    private:
        typedef int (OvxlibDelegate::*AddNodeFunc)(Model*, Operation*, uint32_t);
        std::map<OperationType, AddNodeFunc> op_container_;
        uint32_t node_unique_id_;
        std::map<uint32_t, vsi_nn_tensor_id_t> tensor_map_;
        std::map<uint32_t, vsi_nn_node_id_t> node_map_;
        vsi_nn_graph_t* graph_;
        std::vector<std::vector<int8_t>> size_pool_;
};

inline std::map<uint32_t, vsi_nn_tensor_id_t> OvxlibDelegate::getTensorMapping()const{
    return tensor_map_;
}

}

#endif
