#
# Android build
#
LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)

LOCAL_CPP_EXTENSION := .cpp

AQROOT=$(NNRT_LOCAL_PATH)
ifeq ($(AQROOT),)
$(error Please set AQROOT env first)
endif

OVXLIB_DIR = $(AQROOT)/ovxlib
ifeq ($(OVXLIB_DIR),)
$(error Please set OVXLIB_DIR env first)
endif

ifeq ($NNRT_ROOT),)
$(error Please set NNRT_ROOT to parent dir of nnrt)
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
        op/convolution.cpp \
        op/normalization.cpp \
        op/elementwise.cpp \
        op/operand.cpp \
        op/operation.cpp \
        op/pooling.cpp \
        api_requirement/nnapi_requirement.cpp \
        api_requirement/spec.cpp

ifeq ($(DUMP_JSON_MODEL), 1)
LOCAL_SRC_FILES += \
        dump_model/dump_json_model.cpp \
        dump_model/jsoncpp.cpp
endif

LOCAL_C_INCLUDES += \
    vendor/nxp/fsl-proprietary/include/CL \
    vendor/nxp/fsl-proprietary/include/VX \
    vendor/nxp/fsl-proprietary/include \
    $(OVXLIB_DIR)/include \
    $(OVXLIB_DIR)/include/ops \
    $(OVXLIB_DIR)/include/utils \
    $(OVXLIB_DIR)/include/infernce \
    $(OVXLIB_DIR)/include/platform \
    $(OVXLIB_DIR)/include/client \
    $(OVXLIB_DIR)/include/libnnext\
    $(AQROOT)/sdk/inc\
    $(LOCAL_PATH)/boost/libs/preprocessor/include\
    $(LOCAL_PATH)/api_requirement/\
    $(LOCAL_PATH)/../../include

LOCAL_SHARED_LIBRARIES += libovxlib

LOCAL_CFLAGS :=  \
    -Werror \
    -fexceptions\
    -D'OVXLIB_API=__attribute__((visibility("default")))' \
    -Wno-unused-parameter\
    -Wno-implicit-fallthrough\
    -frtti

IS_GIT:=$(shell cd $(NNRT_ROOT)/nnrt/;  git status  1>&2 > /dev/null; echo $$?)
ifeq ($(IS_GIT),0)
       HEAD_VERSION:=$(shell cd $(NNRT_ROOT)/nnrt/; git log -n 1 --format=%h)
       DIRTY:=$(shell cd $(NNRT_ROOT)/nnrt/; git diff --quiet HEAD || echo '-dirty')
       LOCAL_CFLAGS += -DGIT_STRING='$(HEAD_VERSION)$(DIRTY)'

$(info $(HAED_VERSION))
$(info $(LOCAL_CFLAGS))
endif

ifeq ($(DUMP_JSON_MODEL), 1)
LOCAL_CFLAGS += \
    -D_DUMP_JSON_MODEL_
endif

LOCAL_LDLIBS := -llog
LOCAL_MODULE:= libnnrt
LOCAL_MODULE_TAGS := optional
LOCAL_PRELINK_MODULE := false
include $(BUILD_SHARED_LIBRARY)
