#
# Build Vivante chipinfo for android.
#
LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)

include $(AQROOT)/Android.mk.def

ifeq ($(PLATFORM_VENDOR),1)
LOCAL_VENDOR_MODULE  := true
endif

LOCAL_CPP_EXTENSION := .cc

# head file
LOCAL_SRC_FILES :=\
        model.h\
        operation.h\
        operand.h\
        file_map_memory.h\
        types.h\
        event.h\
        error.h\
        utils.h\
        execution.h\
        compilation.h\
        prepared_model.h\
        ovxlib_delegate.h\
        graph_transformations/transformations.h\
        memory_pool.h

# source file
LOCAL_SRC_FILES :=\
        model.cc \
        operation.cc\
        operand.cc\
        file_map_memory.cc\
        compilation.cc\
        event.cc\
        utils.cc\
        execution.cc\
        prepared_model.cc\
        ovxlib_delegate.cc\
        graph_transformations/t2c.cc\
		graph_transformations/align_broadcast_op.cc\
        graph_transformations/transformations.cc\
        graph_transformations/optimize_permute.cc\
        graph_transformations/fp32tofp16.cc\
        graph_transformations/validate_quantized_graph.cc\
        graph_transformations/nnapi_interpreter.cc \
        memory_pool.cc \

LOCAL_C_INCLUDES += \
    $(AQROOT)/sdk/inc/CL \
    $(AQROOT)/sdk/inc/VX \
    $(LOCAL_PATH)/../include \
    $(LOCAL_PATH)/../include/ops \
    $(LOCAL_PATH)/../include/utils \
    $(LOCAL_PATH)/../include/infernce \
    $(LOCAL_PATH)/../include/platform \
    $(LOCAL_PATH)/../include/client \
    $(LOCAL_PATH)/../include/libnnext\
    $(AQROOT)/sdk/inc
 
LOCAL_SHARED_LIBRARIES += \
	libovxlib


LOCAL_CFLAGS :=  \
	-Werror \
	-D'OVXLIB_API=__attribute__((visibility("default")))' \
        -Wno-sign-compare \
        -Wno-implicit-function-declaration \
        -Wno-sometimes-uninitialized \
        -Wno-unused-parameter \
        -Wno-enum-conversion \
        -Wno-missing-field-initializers \
        -Wno-tautological-compare \

LOCAL_MODULE:= libnnadapter
LOCAL_MODULE_TAGS := optional
LOCAL_PRELINK_MODULE := false
include $(BUILD_STATIC_LIBRARY)
