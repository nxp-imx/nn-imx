LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)

AQROOT=$(NNRT_LOCAL_PATH)
OVXLIB_DIR = $(AQROOT)/ovxlib
ifeq ($(OVXLIB_DIR),)
$(error Please set OVXLIB_DIR env first)
endif

NNRT_ROOT = $(AQROOT)/nn_runtime
ifeq ($(NNRT_ROOT),)
$(error Please set NNRT_ROOT env first)
endif
include $(AQROOT)/Android.mk.def

ifeq ($(PLATFORM_VENDOR),1)
LOCAL_VENDOR_MODULE  := true
endif

LOCAL_C_INCLUDES := \
        frameworks/ml/nn/common/include/ \
        frameworks/ml/nn/runtime/include/  \
        vendor/nxp/fsl-proprietary/include/CL \
        vendor/nxp/fsl-proprietary/include/VX \
        vendor/nxp/fsl-proprietary/include \
	$(NNRT_ROOT)/nnrt \
        $(NNRT_ROOT)/nnrt/boost/libs/preprocessor/include \
        $(NNRT_ROOT) \
        $(OVXLIB_DIR)/include \
        $(OVXLIB_DIR)/include/ops \
        $(OVXLIB_DIR)/include/utils \
        $(OVXLIB_DIR)/include/infernce \
        $(OVXLIB_DIR)/include/platform \
        $(OVXLIB_DIR)/include/client \
        $(OVXLIB_DIR)/include/libnnext \

LOCAL_SHARED_LIBRARIES := \
    libbase \
    libdl   \
    libhardware \
    libhidlbase \
    libhidlmemory   \
    libhidltransport    \
    liblog  \
    libutils    \
    libcutils    \
    android.hardware.neuralnetworks@1.0 \
    android.hidl.allocator@1.0  \
    android.hidl.memory@1.0 \
    libneuralnetworks   \
    libovxlib\
    libnnrt

LOCAL_SRC_FILES:= \
    VsiRTInfo.cpp \
    VsiDriver.cpp \
    1.0/VsiDriver1_0.cpp \
    1.0/VsiDevice1_0.cpp \
    VsiPreparedModel.cpp

ifeq ($(shell expr $(PLATFORM_SDK_VERSION) ">=" 28),1)
LOCAL_SRC_FILES += 1.1/VsiDevice1_1.cpp \
                   1.1/VsiDriver1_1.cpp
endif

ifeq ($(shell expr $(PLATFORM_SDK_VERSION) ">=" 29),1)
LOCAL_SRC_FILES += \
    1.2/VsiDevice1_2.cpp\
    1.2/VsiPreparedModel1_2.cpp\
    1.2/VsiDriver1_2.cpp \
    1.2/VsiBurstExecutor.cpp    \
    hal_limitation/nnapi_limitation.cpp \
    hal_limitation/support.cpp

endif

ifeq ($(shell expr $(PLATFORM_SDK_VERSION) ">=" 28),1)

LOCAL_SHARED_LIBRARIES += android.hardware.neuralnetworks@1.1
LOCAL_STATIC_LIBRARIES += libneuralnetworks_common
LOCAL_C_INCLUDES += frameworks/native/libs/nativewindow/include \
                    frameworks//native/libs/arect/include

ifeq ($(shell expr $(PLATFORM_SDK_VERSION) ">=" 29),1)
LOCAL_C_INCLUDES += frameworks/native/libs/ui/include \
                    frameworks/native/libs/nativebase/include \
                    system/libfmq/include   \
                    $(LOCAL_PATH)/hal_limitation \
                    $(LOCAL_PATH)/op_validate

LOCAL_SHARED_LIBRARIES += libfmq \
                          libui \
                          android.hardware.neuralnetworks@1.2

ifeq ($(shell expr $(PLATFORM_VERSION) ">=" R),1)
LOCAL_SHARED_LIBRARIES += \
                          android.hardware.neuralnetworks@1.3

LOCAL_CFLAGS += -DANDROID_NN_API=30
endif

LOCAL_MODULE      := android.hardware.neuralnetworks@1.2-service-vsi-npu-server
else
LOCAL_SHARED_LIBRARIES += libneuralnetworks
LOCAL_MODULE      := android.hardware.neuralnetworks@1.1-service-vsi-npu-server
endif


else

LOCAL_MODULE      := android.hardware.neuralnetworks@1.0-service-vsi-npu-server
endif

LOCAL_MODULE_RELATIVE_PATH := hw
LOCAL_INIT_RC := VsiDriver.rc

LOCAL_CFLAGS += -DANDROID_SDK_VERSION=$(PLATFORM_SDK_VERSION)  -Wno-error=unused-parameter\
                -Wno-unused-private-field \
                -Wno-unused-parameter \
                -Wno-delete-non-virtual-dtor -Wno-non-virtual-dtor\

LOCAL_MODULE_TAGS := optional

include $(BUILD_EXECUTABLE)
