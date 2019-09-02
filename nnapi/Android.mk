##############################################################################
#
#    Copyright (c) 2005 - 2019 by Vivante Corp.  All rights reserved.
#
#    The material in this file is confidential and contains trade secrets
#    of Vivante Corporation. This is proprietary information owned by
#    Vivante Corporation. No part of this work may be disclosed,
#    reproduced, copied, transmitted, or used in any way for any purpose,
#    without the express written permission of Vivante Corporation.
#
##############################################################################


LOCAL_PATH := $(call my-dir)
include $(LOCAL_PATH)/../../../Android.mk.def


#
#  libNeuralNetworks
#

include $(CLEAR_VARS)

# Core
LOCAL_SRC_FILES := \
	AMemory.cpp 		\
	AModel.cpp 			\
	ACompilation.cpp	\
	AExecution.cpp		\
	AEvent.cpp			\
	util.cpp



ifeq ($(NN_DUMP), 1)
LOCAL_SRC_FILES += \
	jsoncpp.cpp\
	json.h
endif

LOCAL_CFLAGS += \
	-D__LINUX__\
	-std=c++0x\
	-Wno-unused-parameter\
	-Wno-unused-private-field\
	-Wno-ignored-qualifiers
	#-Wno-missing-field-initializers\
	-Wno-missing-braces\

LOCAL_C_INCLUDES := \
    $(AQROOT)/sdk/inc \


LOCAL_LDFLAGS := \
    -lm\
    -Wl,-z,defs

LOCAL_SHARED_LIBRARIES := \
    libOpenVX

LOCAL_MODULE         := libNeuralNetworks
LOCAL_MODULE_TAGS    := optional
LOCAL_PRELINK_MODULE := false
ifeq ($(PLATFORM_VENDOR),1)
LOCAL_VENDOR_MODULE  := true
endif
include $(BUILD_SHARED_LIBRARY)

include $(AQROOT)/copy_installed_module.mk
