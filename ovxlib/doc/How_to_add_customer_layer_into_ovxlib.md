# How to add Customer Layer into ovxlib ?

## Good News

Now! You can use tools/add_op.py to generate Customer Layer's stub codes in ovxlib.

`python3 add_op.py --type <type> [--prefix <prefix>] [--without--kernel] op_name1 ...`

`--type` option:

 - embedded: add embedded layer into ovxlib.These ops are in /src/ops/, which is maintained by Vivante to support the ops that are not in OpenVX Spec.
 - custom: add custom layer into ovxlib. These ops are in /src/custom/ops. Customer can put their own ops in there.

`--without--kernel` option:

If you want to add a new op, which call OpenVX API directly and no kernel is needed, you can set this flag.

`--prefix` option:

Set the prefix only for the custom op.

## Introduction

Generally, Acuity Tools can generate Customer Layer's stub codes in App, when it meet unknown layer. But sometimes you will want to add customer layer to ovxlib, so it can be used as a built-in layer.

One Customer Layer includes these files:(For example, we add foo op as a customer layer)
 - src/ops/vsi_nn_op_foo.c & include/ops/vsi_nn_op_foo.h
 - src/libnnext/ops/kernel/vsi_nn_kernel_foo.c & include/libnnext/vx_lib_nnext.h
 - src/libnnext/ops/vx/vsi_nn_kernel_foo.vx (shader kernel needed)

If you want to add CPU kernel to ovxlib, you can reference the code of fullconnect2 op.

If you want to add Shader kernel to ovxlib, you can reference the code of prelu op.

## Step 1

Copy the reference op files, and rename reference op to foo op both file name and file contexts.

Add these new foo op files to project files:
 - ovxlib.vcxproj & ovxlib.vcxproj.filters (for VS build)
 - BUILD (for bazel Build)
 - src/makefile.linux (for make build)
 - src/Android.mk (for Android build)

## Step 2

Add `DEF_OP(FOO)` to the end of include/interface/ops.def.

Add new item for foo op in array:
 - src/utils/vsi_nn_code_generator.c: s_op_gen[]
 - src/vsi_nn_node_attr_template.c: s_template[]

## Key Points

### src/ops/vsi_nn_op_foo.c: op_setup

Compute the ouput tensor's shape.

### src/libnnext/ops/kernel/vsi_nn_kernel_foo.c: VxFooKernel

The CPU implement of foo op.

### include\vsi_nn_node_type.h: vsi_nn_nn_param_t

If foo op have some parameters, DO NOT forget add vsi_nn_foo_param into vsi_nn_nn_param_t.

## Add Shader Kernel

### src/libnnext/ops/kernel/vsi_nn_kernel_foo.c: kernel_info

We use kernel_info struct to input the information of kernel to the OpenVX driver.
 - resource_name: MUST BE SAME to the shader file name. For example, `src/libnnext/ops/vx/vsi_nn_kernel_foo.vx` its resource_name is `vsi_nn_kernel_foo`.
 - type: `VX_KERNEL_TYPE_CPU`(CPU Kernel), `VX_KERNEL_TYPE_VX`(Shader Kernel), `VX_KERNEL_TYPE_BIN`(Binary Kernel)
 - init_index: the index of init functions. The init of CPU kernel or Shader kernel can be different.
 - kernel_index: the index of kernel array. In src/libnnext/ops/kernel/vsi_nn_kernel_foo.c, we can define many kernels(both CPU and Shader) for foo op. These kernels are collected in vx_kernel_FOO_list.

### tools/build_vx_files.py

After update any shader script in src/libnnext/ops/vx, you MUST use tools/build_vx_files.py to generate src/libnnext/vsi_nn_libnnext_vx.c, which includes ALL shader scripts' text. So we don't need these shader scripts at runtime anymore.

If you only build .vx files for embeded op, You can:

`python3 build_vx_files.py --type embedded`

If you build .vx files for both embeded op and custom op, You can:

`python3 build_vx_files.py --type custom`
