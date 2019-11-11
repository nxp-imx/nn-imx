#
# Android build
#
LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)

LOCAL_CPP_EXTENSION := .cpp

ifeq ($(AQROOT),)
$(error Please set AQROOT env first)
endif

OVXLIB_DIR = $(AQROOT)/driver/nn/ovxlib
ifeq ($(OVXLIB_DIR),)
$(error Please set OVXLIB_DIR env first)
endif

include $(AQROOT)/Android.mk.def

ifeq ($(PLATFORM_VENDOR),1)
LOCAL_VENDOR_MODULE  := true
endif

# source file
LOCAL_SRC_FILES :=\
        memory_pool.cpp \
        model.cpp \
        compilation.cpp \
        event.cpp \
        utils.cpp \
        logging.cpp \
        execution.cpp \
        execution_task.cpp \
        prepared_model.cpp \
        ovxlib_delegate.cpp \
        file_map_memory.cpp \
        model_transform/layout_inference.cpp \
        model_transform/align_broadcast_op.cpp \
        model_transform/transformations.cpp \
        model_transform/optimize_permute.cpp \
        model_transform/fp32tofp16.cpp \
        model_transform/validate_quantized_graph.cpp \
        model_transform/nnapi_interpreter.cpp \
        op/activation.cpp \
        op/convolution.cpp \
        op/normalization.cpp \
        op/elementwise.cpp \
        op/operand.cpp \
        op/operation.cpp \
        op/pooling.cpp \
        api_requirement/nnapi_requirement.cpp \
        api_requirement/spec.cpp\
        dump_model/dump_json_model.cpp \
        dump_model/jsoncpp.cpp

LOCAL_C_INCLUDES += \
    $(AQROOT)/sdk/inc/CL \
    $(AQROOT)/sdk/inc/VX \
    $(OVXLIB_DIR)/include \
    $(OVXLIB_DIR)/include/ops \
    $(OVXLIB_DIR)/include/utils \
    $(OVXLIB_DIR)/include/infernce \
    $(OVXLIB_DIR)/include/platform \
    $(OVXLIB_DIR)/include/client \
    $(OVXLIB_DIR)/include/libnnext\
    $(AQROOT)/sdk/inc\
    $(LOCAL_PATH)/boost/libs/preprocessor/include\
    $(LOCAL_PATH)/api_requirement/

LOCAL_SHARED_LIBRARIES += libovxlib

LOCAL_CFLAGS :=  \
    -Werror \
    -fexceptions\
    -D'OVXLIB_API=__attribute__((visibility("default")))' \
    -Wno-unused-parameter\
    -Wno-implicit-fallthrough

LOCAL_LDLIBS := -llog
LOCAL_MODULE:= libnnrt
LOCAL_MODULE_TAGS := optional
LOCAL_PRELINK_MODULE := false
include $(BUILD_SHARED_LIBRARY)
