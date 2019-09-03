#ifndef __OVXLIB_OPERAND_H__
#define __OVXLIB_OPERAND_H__

#include <vector>
#include "vsi_nn_pub.h"
#include "types.h"

#include "memory_pool.h"
#include <cassert>

namespace ovxlib
{
#define OVXLIB_INVALID_OPERAND_INDEX    ((uint32_t)0xFFFFFFFF)

// struct DataLocation
// {
//     uint32_t poolIndex = 0;

//     uint32_t offset = 0;

//     size_t length = 0;
// };

struct QuantizationParams
{
    struct {
        float scale;
        int32_t zeroPoint;
    } scalar;

    struct {
        std::vector<float> scale;
        std::vector<int32_t> zeroPoint;
    } vec;
};

struct BaseOperand
{
    OperandType type;
    std::vector<uint32_t> dimensions;
    QuantizationParams quant;
    union
    {
        int32_t     int32;
        uint32_t    uint32;
        float       float32;
        double      float64;
    } scalar;

    mem_pool::shared_ref mem_ref;
};

class Operand: public BaseOperand
{
    public:

        void setPerm(std::vector<uint32_t>& perm) {
            perm_ = perm;
        }

        std::vector<uint32_t>& perm() {return perm_;};

        size_t bytes() const;

        size_t size() const {
            if (dimensions.size() == 0) {
                return 0;
            }
            uint32_t num = 1;
            for (auto i : dimensions) {
                num *= i;
            }
            return num;
        }

        size_t ndim() const {return dimensions.size();}

        // life time
        // data location
        bool isTensor() const;

        bool isQuantized() const;

        bool isConst() const {
            return (mem_ref && mem_ref->len_ > 0 && !is_graph_input_output_);
        }

        void setGraphInputOutput() {
            is_graph_input_output_ = true;
        }

        bool isNull() const {
            return optional_null_;
        }

        void setNull() {
            optional_null_ = true;
        }

        void clearNull() {
            optional_null_ = false;
        }

        bool isValid() {
            if (!isConst()) {
                return true;
            }
            for (auto i : dimensions) {
                if (i <= 0) {
                    return false;
                }
            }
            return true;
        }

        Operand* clone();

        void cloneQuantParams(Operand* operand);

        void echo(uint32_t index = 0) const;

    private:
        uint32_t number_of_consumers_;
        std::vector<uint32_t> perm_;
        bool optional_null_ = false;
        bool is_graph_input_output_ = false;
};

}
#endif
