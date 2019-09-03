# it is to build android hal

LOCAL_PATH := $(call my-dir)

ifeq ($(AQROOT),)
$(error Please set AQROOT env first)
endif


NPU_HAL_MAKEFILES := $(LOCAL_PATH)/src/Android.mk \
                     $(LOCAL_PATH)/nnapi/Android.mk \
                     $(LOCAL_PATH)/hal/Android.mk

include $(NPU_HAL_MAKEFILES)
