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

ifeq ($(findstring imx, $(TARGET_BOARD_PLATFORM)), imx)
NNRT_LOCAL_PATH := $(call my-dir)
NNRT_ROOT = $(AQROOT)/nn_runtime
include $(NNRT_LOCAL_PATH)/Android.mk.def
include $(NNRT_LOCAL_PATH)/nn_runtime/nnrt/Android.mk
include $(NNRT_LOCAL_PATH)/nn_runtime/android_hal/Android.mk
include $(NNRT_LOCAL_PATH)/ovxlib/src/Android.mk
endif
