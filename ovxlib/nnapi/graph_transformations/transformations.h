#ifndef __OVXLIB_TRANSFORMATIONS_H__
#define __OVXLIB_TRANSFORMATIONS_H__

#include <vector>
#include <set>
#include "model.h"

namespace ovxlib
{
class Transformation
{
    public:
        virtual ~Transformation(){};

        virtual int run(Model * model, bool * modified) = 0;

        virtual const char* name() = 0;
};
#define NEW_GRAPH_TRANSFORMATION(Name)                      \
    class Name : public Transformation                      \
    {                                                       \
        public:                                             \
            int run(Model* model,                           \
                    bool* modified) override;               \
            const char* name() override {return #Name;}     \
    }

class TransformationSet
{
    public:
        virtual ~TransformationSet();
        virtual void add(Transformation* transformation);
        virtual int run(Model* model);
        virtual int once(Model* model);

    private:
        std::vector<Transformation*> transformations_;
};

NEW_GRAPH_TRANSFORMATION(T2C);
NEW_GRAPH_TRANSFORMATION(OptimizePermute);
NEW_GRAPH_TRANSFORMATION(Fp32ToFp16);
NEW_GRAPH_TRANSFORMATION(ValidateQuantizedGraph);
NEW_GRAPH_TRANSFORMATION(AlignBroadcastOp);

#undef NEW_GRAPH_TRANSFORMATION

class NnApiInterpreter : public Transformation
{
    public:
        NnApiInterpreter();
        virtual ~NnApiInterpreter();

        const char* name() override {return "NnApiInterpreter";}

        int run(Model* model, bool* modified) override;

        FusedType mapFusedType(int fused_code);

        PadType mapPadType(int code);

        FusedType mapLstmActivationType(int value);

        LshProjectionType mapLshProjectionType(int value);

        template <typename T>
        std::vector<T> reverseArray(T* data, size_t length) {
            std::vector<T> array(length);
            for (size_t i = 0; i < length; ++ i) {
                array[i] = data[length - i - 1];
            }
            return array;
        }

        template <typename T>
        std::vector<T> reverseArray(std::vector<T>& data) {
            std::vector<T> array(data.size());
            for (size_t i = 0; i < data.size(); ++ i) {
                array[i] = data[data.size() - i - 1];
            }
            return array;
        }

        int32_t reverseMask(int32_t mask, size_t dim_num);

        inline std::vector<int32_t> convertPermute(std::vector<int32_t> & perm) {
            return convertAxes(perm, perm.size());
        }

        inline std::vector<int32_t> convertPermute(int32_t* perm_buffer, size_t length) {
            return convertAxes(perm_buffer, length, length);
        }

        std::vector<int32_t> convertAxes(int32_t* axes_buffer, size_t length, size_t dim_num);

        std::vector<int32_t> convertAxes(std::vector<int32_t> & axes, size_t dim_num);

        inline int32_t convertAxis(int32_t axis, int32_t dim_num) {
            return (dim_num - computeAxis(axis, dim_num) - 1);
        }

        void fillIntArray(Model* model, Operation* operation,
                std::vector<int32_t>& array, int32_t op_index, bool reverse, bool is_axis);

        inline int32_t computeAxis(int32_t axis, int32_t dim_num) {
            if (axis >= 0) {
                return axis;
            } else {
                return dim_num + axis;
            }
        }

        inline void truncateOperationIOs(Model* model, Operation* operation,
                int32_t input_num, int32_t output_num);

        inline void resetFusedType(Model* model, Operation* operation,
                int32_t input_index) {
            Operand* operand = model->operand(operation->input(input_index));
            operation->setFusedType(mapFusedType(operand->scalar.int32));
        }

        void replaceOperation(Model* model, uint32_t op_index,
                Operation* new_operation);

#define REGISTER_OP(NAME)   \
        Operation* map_##NAME(Model* model, Operation* operation, uint32_t)
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
#undef  REGISTER_OP
    private:
        typedef Operation* (NnApiInterpreter::*AddNodeFunc)(Model*, Operation*, uint32_t);
        std::map<OperationType, AddNodeFunc> op_container_;
        std::set<uint32_t> operands_to_remove_;
};

#define MARK_DATA_DIRECTION_CONSUMER 0
#define MARK_DATA_DIRECTION_PRODUCER 1

typedef std::map<Operation*, std::vector<uint32_t>> MarkDataEdge;
typedef struct
{
    MarkDataEdge producers;
    MarkDataEdge consumers;
} T2CMarkData;

}
#endif
