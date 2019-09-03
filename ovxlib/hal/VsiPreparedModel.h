#ifndef ANDROID_ML_NN_COMMON_OPENVX_EXECUTOR_H
#define ANDROID_ML_NN_COMMON_OPENVX_EXECUTOR_H

#include "HalInterfaces.h"

#include "Utils.h"
#include "../nnapi/model.h"
#include "../nnapi/types.h"
#include "../nnapi/compilation.h"
#include "../nnapi/execution.h"
#include "../nnapi/file_map_memory.h"
#include <vector>

using android::sp;

namespace android {
namespace nn {
namespace vsi_driver {
    class VsiPreparedModel : public IPreparedModel {
   public:
    VsiPreparedModel(const Model& model):model_(model) {
        native_model_ = std::make_shared<ovxlib::Model>();
        Create(model);
        }

    ~VsiPreparedModel() override {
        }

    // TODO: Make this asynchronous
    Return<ErrorStatus> execute(
        const Request& request,
        const sp<IExecutionCallback>& callback) override;

   private:
        /*create ovxlib model and compliation*/
        Return<ErrorStatus> Create(const Model& model);

        void fill_operand_value(ovxlib::Operand* ovx_operand, const Operand& hal_operand) ;
        void construct_ovx_operand(ovxlib::Operand* ovx_oprand,const Operand& hal_operand);

        const Model model_;
        std::shared_ptr<ovxlib::Model> native_model_;
        std::shared_ptr<ovxlib::Compilation> native_compile_;
        std::shared_ptr<ovxlib::Execution> native_exec_;

        std::vector<std::shared_ptr<ovxlib::Memory>> ovx_memory_;
        std::vector<sp<IMemory>> shared_memory_;
};
}
}
}
#endif
