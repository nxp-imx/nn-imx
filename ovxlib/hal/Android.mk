LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)

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
        $(LOCAL_PATH)/../include \
        $(LOCAL_PATH)/../include/ops \
        $(LOCAL_PATH)/../include/utils \
        $(LOCAL_PATH)/../include/infernce \
        $(LOCAL_PATH)/../include/platform \
        $(LOCAL_PATH)/../include/client \
        $(LOCAL_PATH)/../include/libnnext


LOCAL_SRC_FILES:= \
    VsiDriver.cpp   \
    VsiPreparedModel.cpp\
    VsiDevice.cpp

LOCAL_SHARED_LIBRARIES := \
    libbase \
    libdl   \
    libhardware \
    libhidlbase \
    libhidlmemory   \
    libhidltransport    \
    liblog  \
    libutils    \
    android.hardware.neuralnetworks@1.0 \
    android.hidl.allocator@1.0  \
    android.hidl.memory@1.0 \
    libneuralnetworks   \
    libovxlib

LOCAL_STATIC_LIBRARIES += libnnadapter

ifeq ($(shell expr $(PLATFORM_SDK_VERSION) ">=" 28),1)

LOCAL_SHARED_LIBRARIES += android.hardware.neuralnetworks@1.1
LOCAL_STATIC_LIBRARIES += libneuralnetworks_common

LOCAL_CFLAGS += -Wno-error=unused-variable -Wno-error=unused-function -Wno-error=return-type

LOCAL_CFLAGS += -DMULTI_CONTEXT

LOCAL_MODULE      := android.hardware.neuralnetworks@1.1-service-vsi-npu-server

else

LOCAL_MODULE      := android.hardware.neuralnetworks@1.0-service-vsi-npu-server
endif

LOCAL_CFLAGS += -DANDROID_SDK_VERSION=$(PLATFORM_SDK_VERSION)  -Wno-error=unused-parameter
LOCAL_MODULE_TAGS := optional

include $(BUILD_EXECUTABLE)
