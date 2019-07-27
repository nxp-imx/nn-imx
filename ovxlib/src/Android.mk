##############################################################################
#
#    Copyright 2012 - 2019 Vivante Corporation, Santa Clara, California.
#    All Rights Reserved.
#
#    Permission is hereby granted, free of charge, to any person obtaining
#    a copy of this software and associated documentation files (the
#    'Software'), to deal in the Software without restriction, including
#    without limitation the rights to use, copy, modify, merge, publish,
#    distribute, sub license, and/or sell copies of the Software, and to
#    permit persons to whom the Software is furnished to do so, subject
#    to the following conditions:
#
#    The above copyright notice and this permission notice (including the
#    next paragraph) shall be included in all copies or substantial
#    portions of the Software.
#
#    THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
#    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
#    IN NO EVENT SHALL VIVANTE AND/OR ITS SUPPLIERS BE LIABLE FOR ANY
#    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
##############################################################################


#
# Build Vivante chipinfo for android.
#
LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)

ifeq ($(AQROOT),)
$(error Please set AQROOT env first)
endif



LOCAL_SRC_FILES :=     \
            vsi_nn_context.c \
            vsi_nn_client_op.c \
            vsi_nn_graph.c  \
            vsi_nn_node_attr_template.c  \
            vsi_nn_node.c  \
            vsi_nn_ops.c  \
            vsi_nn_tensor.c \
            vsi_nn_version.c \
            vsi_nn_rnn.c \
            vsi_nn_internal_node.c \
            vsi_nn_log.c


LOCAL_SRC_FILES +=     \
            client/vsi_nn_vxkernel.c

LOCAL_SRC_FILES +=      \
             utils/vsi_nn_code_generator.c   \
             utils/vsi_nn_binary_tree.c   \
             utils/vsi_nn_map.c   \
             utils/vsi_nn_link_list.c   \
             utils/vsi_nn_math.c   \
             utils/vsi_nn_dtype_util.c   \
             utils/vsi_nn_limits.c   \
             utils/vsi_nn_vdata.c   \
             utils/vsi_nn_tensor_op.c   \
             utils/vsi_nn_util.c


LOCAL_SRC_FILES +=      \
             quantization/vsi_nn_dynamic_fixed_point.c   \
             quantization/vsi_nn_asymmetric_affine.c   \


LOCAL_SRC_FILES +=      \
             pycc/vsi_pycc_interface.c


LOCAL_SRC_FILES +=      \
            post/vsi_nn_post_fasterrcnn.c   \
            post/vsi_nn_post_cmupose.c


           
LOCAL_SRC_FILES += libnnext/ops/kernel/vsi_nn_kernel_argmax.c \
        libnnext/ops/kernel/vsi_nn_kernel_crop.c \
        libnnext/ops/kernel/vsi_nn_kernel_eltwisemax.c \
        libnnext/ops/kernel/vsi_nn_kernel_fullconnect2.c \
        libnnext/ops/kernel/vsi_nn_kernel_l2normalizescale.c \
        libnnext/ops/kernel/vsi_nn_kernel_poolwithargmax.c \
        libnnext/ops/kernel/vsi_nn_kernel_prelu.c \
        libnnext/ops/kernel/vsi_nn_kernel_elu.c \
        libnnext/ops/kernel/vsi_nn_kernel_upsample.c \
        libnnext/ops/kernel/vsi_nn_kernel_dropout.c \
        libnnext/ops/kernel/vsi_nn_kernel_resize.c \
        libnnext/ops/kernel/vsi_nn_kernel_reverse.c \
        libnnext/ops/kernel/vsi_nn_kernel_scale.c \
        libnnext/ops/kernel/vsi_nn_kernel_space2batch.c \
        libnnext/ops/kernel/vsi_nn_kernel_reduce.c \
        libnnext/ops/kernel/vsi_nn_kernel_batch2space.c \
        libnnext/ops/kernel/vsi_nn_kernel_space2depth.c \
        libnnext/ops/kernel/vsi_nn_kernel_imageprocess.c \
        libnnext/ops/kernel/vsi_nn_kernel_matrixmul.c \
        libnnext/ops/kernel/vsi_nn_kernel_shufflechannel.c \
        libnnext/ops/kernel/vsi_nn_kernel_layernormalize.c \
        libnnext/ops/kernel/vsi_nn_kernel_instancenormalize.c \
        libnnext/ops/kernel/vsi_nn_kernel_relational_ops.c \
        libnnext/ops/kernel/vsi_nn_kernel_tensorstackconcat.c \
        libnnext/ops/kernel/vsi_nn_kernel_signalframe.c \
        libnnext/ops/kernel/vsi_nn_kernel_sync_host.c \
        libnnext/ops/kernel/vsi_nn_kernel_minimum.c \
        libnnext/ops/kernel/vsi_nn_kernel_pow.c \
        libnnext/ops/kernel/vsi_nn_kernel_floordiv.c \
        libnnext/ops/kernel/vsi_nn_kernel_spatial_transformer.c \
        libnnext/ops/kernel/vsi_nn_kernel_logical_ops.c \
        libnnext/ops/kernel/vsi_nn_kernel_select.c \
        libnnext/ops/kernel/vsi_nn_kernel_lstmunit_activation.c \
        libnnext/ops/kernel/vsi_nn_kernel_tensor_add_mean_stddev_norm.c \
        libnnext/ops/kernel/vsi_nn_kernel_stack.c \
        libnnext/ops/kernel/vsi_nn_kernel_neg.c \
        libnnext/ops/kernel/vsi_nn_kernel_exp.c \
        libnnext/ops/kernel/vsi_nn_kernel_clip.c \
        libnnext/ops/kernel/vsi_nn_kernel_pre_process_gray.c \
        libnnext/ops/kernel/vsi_nn_kernel_unstack.c \
        libnnext/ops/kernel/vsi_nn_kernel_pre_process_rgb.c \
        libnnext/ops/kernel/vsi_nn_kernel_addn.c \
        libnnext/vsi_nn_libnnext_vx.c \

LOCAL_SRC_FILES +=      ops/vsi_nn_op_add.c   \
             ops/vsi_nn_op_batch_norm.c   \
             ops/vsi_nn_op_multiply.c   \
             ops/vsi_nn_op_common.c   \
             ops/vsi_nn_op_concat.c   \
             ops/vsi_nn_op_split.c   \
             ops/vsi_nn_op_conv2d.c   \
             ops/vsi_nn_op_conv1d.c   \
             ops/vsi_nn_op_conv_relu.c   \
             ops/vsi_nn_op_conv_relu_pool.c   \
             ops/vsi_nn_op_deconvolution.c   \
             ops/vsi_nn_op_fullconnect.c   \
             ops/vsi_nn_op_fullconnect_relu.c   \
             ops/vsi_nn_op_leaky_relu.c   \
             ops/vsi_nn_op_lrn.c   \
             ops/vsi_nn_op_noop.c   \
             ops/vsi_nn_op_pool.c   \
             ops/vsi_nn_op_reshape.c   \
             ops/vsi_nn_op_permute.c   \
             ops/vsi_nn_op_prelu.c   \
             ops/vsi_nn_op_elu.c   \
             ops/vsi_nn_op_proposal.c   \
             ops/vsi_nn_op_roi_pool.c   \
             ops/vsi_nn_op_softmax.c   \
             ops/vsi_nn_op_relu.c   \
             ops/vsi_nn_op_relu1.c   \
             ops/vsi_nn_op_relun.c   \
             ops/vsi_nn_op_reorg.c   \
             ops/vsi_nn_op_lstm.c   \
             ops/vsi_nn_op_variable.c   \
             ops/vsi_nn_op_l2_normalize.c   \
             ops/vsi_nn_op_upsample.c   \
             ops/vsi_nn_op_fullconnect2.c   \
             ops/vsi_nn_op_poolwithargmax.c   \
             ops/vsi_nn_op_argmax.c   \
             ops/vsi_nn_op_crop.c   \
             ops/vsi_nn_op_l2normalizescale.c   \
             ops/vsi_nn_op_eltwisemax.c \
             ops/vsi_nn_op_subtract.c    \
             ops/vsi_nn_op_relu6.c   \
             ops/vsi_nn_op_sigmoid.c \
             ops/vsi_nn_op_tanh.c \
             ops/vsi_nn_op_sqrt.c \
             ops/vsi_nn_op_rsqrt.c \
             ops/vsi_nn_op_softrelu.c \
             ops/vsi_nn_op_divide.c \
             ops/vsi_nn_op_dropout.c \
             ops/vsi_nn_op_resize.c   \
             ops/vsi_nn_op_reverse.c   \
             ops/vsi_nn_op_scale.c   \
             ops/vsi_nn_op_reduce.c   \
             ops/vsi_nn_op_slice.c   \
             ops/vsi_nn_op_shufflechannel.c \
             ops/vsi_nn_op_depth2space.c \
             ops/vsi_nn_op_space2depth.c \
             ops/vsi_nn_op_batch2space.c \
             ops/vsi_nn_op_space2batch.c \
             ops/vsi_nn_op_pad.c \
             ops/vsi_nn_op_imageprocess.c \
             ops/vsi_nn_op_matrixmul.c \
             ops/vsi_nn_op_lstmunit.c \
             ops/vsi_nn_op_dataconvert.c \
             ops/vsi_nn_op_layernormalize.c \
             ops/vsi_nn_op_instancenormalize.c \
             ops/vsi_nn_op_strided_slice.c \
             ops/vsi_nn_op_signalframe.c \
             ops/vsi_nn_op_a_times_b_plus_c.c \
             ops/vsi_nn_op_svdf.c \
             ops/vsi_nn_op_abs.c \
             ops/vsi_nn_op_nbg.c \
             ops/vsi_nn_op_tensorstackconcat.c \
             ops/vsi_nn_op_concatshift.c \
             ops/vsi_nn_op_relational_ops.c \
             ops/vsi_nn_op_minimum.c \
             ops/vsi_nn_op_pow.c \
             ops/vsi_nn_op_floordiv.c \
             ops/vsi_nn_op_sync_host.c \
             ops/vsi_nn_op_spatial_transformer.c \
             ops/vsi_nn_op_logical_ops.c \
             ops/vsi_nn_op_select.c \
             ops/vsi_nn_op_lstmunit_activation.c \
             ops/vsi_nn_op_lstmunit_ovxlib.c \
             ops/vsi_nn_op_tensor_add_mean_stddev_norm.c \
             ops/vsi_nn_op_lstm_ovxlib.c \
             ops/vsi_nn_op_hashtable_lookup.c \
             ops/vsi_nn_op_embedding_lookup.c \
             ops/vsi_nn_op_lsh_projection.c \
             ops/vsi_nn_op_rnn.c \
             ops/vsi_nn_op_stack.c \
             ops/vsi_nn_op_floor.c \
             ops/vsi_nn_op_neg.c \
             ops/vsi_nn_op_exp.c \
             ops/vsi_nn_op_clip.c \
             ops/vsi_nn_op_pre_process_tensor.c \
             ops/vsi_nn_op_post_process.c \
             ops/vsi_nn_op_pre_process_gray.c \
             ops/vsi_nn_op_unstack.c \
             ops/vsi_nn_op_pre_process_rgb.c \
             ops/vsi_nn_op_pre_process.c \
             ops/vsi_nn_op_addn.c \
             ops/vsi_nn_op_softmax_internal.c \
             ops/vsi_nn_op_lrn2.c \
             ops/vsi_nn_op_square.c


LOCAL_SRC_FILES +=      platform/nnapi0.4/vsi_nn_pf_softmax.c   \
                 platform/nnapi0.4/vsi_nn_pf_depth2space.c    \


#LOCAL_SRC_FILES +=      platform/nnapi0.3/vsi_nn_pf_softmax.c   \

LOCAL_SRC_FILES += custom/ops/vsi_nn_op_custom_softmax.c

LOCAL_SRC_FILES += custom/ops/kernel/vsi_nn_kernel_custom_softmax.c

LOCAL_SHARED_LIBRARIES := \
	liblog \
	libjpeg \
	libGAL \
	libOpenVX \
	libVSC \
	libdl

LOCAL_C_INCLUDES += \
    external/libjpeg-turbo \
    $(AQROOT)/sdk/inc/CL \
    $(AQROOT)/sdk/inc/VX \
    $(AQROOT)/sdk/inc/ \
    $(AQROOT)/sdk/inc/HAL \
    $(LOCAL_PATH)/../include \
    $(LOCAL_PATH)/../include/ops \
    $(LOCAL_PATH)/../include/utils \
    $(LOCAL_PATH)/../include/infernce \
    $(LOCAL_PATH)/../include/platform \
    $(LOCAL_PATH)/../include/client \
    $(LOCAL_PATH)/../include/libnnext \

LOCAL_CFLAGS :=  \
	-DLINUX \
	-D'OVXLIB_API=__attribute__((visibility("default")))' \
        -Wno-sign-compare \
        -Wno-implicit-function-declaration \
        -Wno-sometimes-uninitialized \
        -Wno-unused-parameter \
        -Wno-enum-conversion \
        -Wno-missing-field-initializers \
        -Wno-tautological-compare \
	-Wno-missing-braces

LOCAL_MODULE:= libovxlib
LOCAL_MODULE_TAGS := optional
LOCAL_PRELINK_MODULE := false
include $(BUILD_SHARED_LIBRARY)
