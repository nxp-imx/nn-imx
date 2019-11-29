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

#if ANDROID_SDK_VERSION > 28
#include "VsiDevice1_2.h"
#elif ANDROID_SDK_VERSION > 27
#include "VsiDevice.h"
#endif

#include "HalInterfaces.h"
#include "Utils.h"

#if ANDROID_SDK_VERSION > 27
#include "ValidateHal.h"
#endif

#include <android/hardware/neuralnetworks/1.2/IDevice.h>
#include <sys/system_properties.h>
#include <android-base/logging.h>
#include <hidl/LegacySupport.h>
#include <thread>


using namespace android::hardware::neuralnetworks::V1_2;

namespace android {
namespace nn {
namespace vsi_driver {


class VsiDriver : public VsiDevice {
   public:
    VsiDriver() : VsiDevice("vsi-npu") {initalizeEnv();}
    Return<void> getCapabilities(getCapabilities_cb _hidl_cb) ;
    Return<void> getSupportedOperations(const V1_0::Model& model, V1_0::IDevice::getSupportedOperations_cb cb) ;

#if ANDROID_SDK_VERSION > 27
    Return<void> getCapabilities_1_1(V1_2::IDevice::getCapabilities_1_1_cb _hidl_cb) ;
    Return<void> getSupportedOperations_1_1(const V1_1::Model& model,
                                                                V1_2::IDevice::getSupportedOperations_1_1_cb cb) ;
#endif

#if ANDROID_SDK_VERSION > 28
    Return<void> getCapabilities_1_2(V1_2::IDevice::getCapabilities_1_2_cb _hidl_cb) ;
    Return<void> getSupportedOperations_1_2(const V1_2::Model& model,
                                            V1_2::IDevice::getSupportedOperations_1_2_cb cb) ;

    Return<void>
        getVersionString(V1_2::IDevice::getVersionString_cb _hidl_cb) {
        _hidl_cb(ErrorStatus::NONE, "android hal vsi npu 1.2 alpha");
        return Void();
    };

    Return<void>
        getType(V1_2::IDevice::getType_cb _hidl_cb) {
        _hidl_cb(ErrorStatus::NONE, V1_2::DeviceType::ACCELERATOR);
        return Void();
    };

    Return<void>
        getSupportedExtensions(V1_2::IDevice::getSupportedExtensions_cb _hidl_cb) {
        _hidl_cb(ErrorStatus::NONE, {/* No extensions. */});
        return Void();
    };

    Return<void>
        getNumberOfCacheFilesNeeded(V1_2::IDevice::getNumberOfCacheFilesNeeded_cb _hidl_cb) {
        // Set both numbers to be 0 for cache not supported.
        _hidl_cb(ErrorStatus::NONE, /*numModelCache=*/0, /*numDataCache=*/0);
        return Void();
    };
#endif

    template<typename T_operation,typename T_Model>
    static bool isSupportedOperation(const T_operation &operation, const T_Model& model);

   private:
   int32_t disable_float_feature_; // switch that float-type running on hal
   private:
    void initalizeEnv();

    template <typename T_model, typename T_getSupportOperationsCallback>
    Return<void> getSupportedOperationsBase(const T_model& model,
                                            T_getSupportOperationsCallback cb);
};
}
}
}
