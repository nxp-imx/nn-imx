/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_ML_NN_SAMPLE_DRIVER_SAMPLE_DRIVER_H
#define ANDROID_ML_NN_SAMPLE_DRIVER_SAMPLE_DRIVER_H

//#include "OvxExecutor.h"
#include "../nnapi/model.h"
#include "HalInterfaces.h"
#include "NeuralNetworks.h"
#include "Utils.h"

#if ANDROID_SDK_VERSION > 27
#include "ValidateHal.h"
#endif
#include <pthread.h>

#include <string>

using android::sp;

namespace android {
namespace nn {
namespace vsi_driver {

#if ANDROID_SDK_VERSION == 27
namespace V1_0 = ::android::hardware::neuralnetworks::V1_0;
#endif

class VsiDevice : public IDevice {
   public:
    VsiDevice(const char* name) : name_(name) {}
    ~VsiDevice() override {}

    Return<ErrorStatus> prepareModel(const V1_0::Model& model,
                                     const sp<IPreparedModelCallback>& callback)  {
#if ANDROID_SDK_VERSION > 27
        return prepareModel_1_1(convertToV1_1(model), ExecutionPreference::FAST_SINGLE_ANSWER, callback);
#else
        return prepareModel_1_1(model, callback);
#endif
    }

    Return<ErrorStatus> prepareModel_1_1(const Model& model,
#if ANDROID_SDK_VERSION > 27
                                         ExecutionPreference preference,
#endif
                                         const sp<IPreparedModelCallback>& callback);

    Return<DeviceStatus> getStatus() override;

    bool Initialize() {
        run();
        return true;
    }

    // Device driver entry_point
    virtual int run();

   protected:
    std::string name_;
};
}
}
#endif
}
