LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)

OVXLIB_DIR = $(AQROOT)/driver/nn/ovxlib
ifeq ($(OVXLIB_DIR),)
$(error Please set OVXLIB_DIR env first)
endif

NNRT_ROOT = $(AQROOT)/driver/nn/nn_runtime
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
        $(AQROOT)/sdk/inc   \
        $(AQROOT)/sdk/inc/CL \
        $(AQROOT)/sdk/inc/VX \
        $(NNRT_ROOT)/nnrt \
        $(NNRT_ROOT) \
        $(OVXLIB_DIR)/include \
        $(OVXLIB_DIR)/include/ops \
        $(OVXLIB_DIR)/include/utils \
        $(OVXLIB_DIR)/include/infernce \
        $(OVXLIB_DIR)/include/platform \
        $(OVXLIB_DIR)/include/client \
        $(OVXLIB_DIR)/include/libnnext


LOCAL_SRC_FILES:= \
    VsiDriver.cpp

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


ifeq ($(shell expr $(PLATFORM_SDK_VERSION) ">=" 28),1)

LOCAL_SHARED_LIBRARIES += android.hardware.neuralnetworks@1.1
LOCAL_STATIC_LIBRARIES += libneuralnetworks_common

LOCAL_CFLAGS += -Wno-error=unused-variable -Wno-error=unused-function -Wno-error=return-type \
                -Wno-unused-parameter

LOCAL_C_INCLUDES += frameworks/native/libs/nativewindow/include \
                    frameworks//native/libs/arect/include

ifeq ($(shell expr $(PLATFORM_SDK_VERSION) ">=" 29),1)
LOCAL_C_INCLUDES += frameworks/ml/nn/runtime/include \
                    frameworks/native/libs/ui/include \
                    frameworks/native/libs/nativebase/include \
                    system/libfmq/include

LOCAL_SHARED_LIBRARIES += libfmq \
                          libui \
                          android.hardware.neuralnetworks@1.2

LOCAL_SRC_FILES += VsiDevice1_2.cpp\
    VsiPreparedModel1_2.cpp
LOCAL_MODULE      := android.hardware.neuralnetworks@1.2-service-vsi-npu-server
else
LOCAL_SRC_FILES += VsiDevice.cpp\
    VsiPreparedModel.cpp

LOCAL_SHARED_LIBRARIES += libneuralnetworks
LOCAL_MODULE      := android.hardware.neuralnetworks@1.1-service-vsi-npu-server
endif


else

LOCAL_MODULE      := android.hardware.neuralnetworks@1.0-service-vsi-npu-server
endif

LOCAL_MODULE_RELATIVE_PATH := hw
LOCAL_INIT_RC := VsiDriver.rc

LOCAL_CFLAGS += -DANDROID_SDK_VERSION=$(PLATFORM_SDK_VERSION)  -Wno-error=unused-parameter\
                -Wno-delete-non-virtual-dtor -Wno-non-virtual-dtor\

LOCAL_MODULE_TAGS := optional

include $(BUILD_EXECUTABLE)
