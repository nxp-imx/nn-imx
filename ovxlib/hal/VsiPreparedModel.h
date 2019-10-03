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

    /*record the info that is gotten from hidl_memory*/
    struct VsiRTInfo{
        sp<IMemory>         shared_mem;             /* if hidl_memory is "ashmem", */
                                                    /* the shared_mem is relative to ptr */

        std::string         mem_type;               /* record type of hidl_memory*/
        uint8_t *           ptr;                    /* record the data pointer gotten from "ashmem" hidl_memory*/
        std::shared_ptr<ovxlib::Memory>  vsi_mem;   /* ovx memory object converted from "mmap_fd" hidl_memory*/
    };

    class VsiPreparedModel : public IPreparedModel {
   public:
    VsiPreparedModel(const Model& model):model_(model) {
        native_model_ = std::make_shared<ovxlib::Model>();
        Create(model);
        }

    ~VsiPreparedModel() override {
        release_rtinfo(const_buffer_);
        }

    // TODO: Make this asynchronous
    Return<ErrorStatus> execute(
        const Request& request,
        const sp<IExecutionCallback>& callback) override;

   private:
        /*create ovxlib model and compliation*/
        Return<ErrorStatus> Create(const Model& model);

        void fill_operand_value(ovxlib::Operand* ovx_operand, const Operand& hal_operand);
        void construct_ovx_operand(ovxlib::Operand* ovx_oprand,const Operand& hal_operand);
        int map_rtinfo_from_hidl_memory(const hidl_vec<hidl_memory>& pools,
            std::vector<VsiRTInfo>& rtInfos);
        void release_rtinfo(std::vector<VsiRTInfo>& rtInfos);

        const Model model_;
        std::shared_ptr<ovxlib::Model> native_model_;
        std::shared_ptr<ovxlib::Compilation> native_compile_;
        std::shared_ptr<ovxlib::Execution> native_exec_;

        /*store pointer of all of hidl_memory to buffer*/
        std::vector<VsiRTInfo> const_buffer_;
        std::vector<VsiRTInfo> io_buffer_;
};
}
}
}
#endif
