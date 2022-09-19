# How to add Customer Layer into ovxlib ?

## Good News

Now! You can use tools/add_op2.py to generate Customer Layer's stub codes in ovxlib.

`python3 add_op2.py --type <type> op_name1 ...`

`--type` option:

 - embedded: add embedded layer into ovxlib.These ops are in /src/ops/, which is maintained by Vivante to support the ops that are not in OpenVX Spec.
 - custom: add custom layer into ovxlib. These ops are in /src/custom/ops. Customer can put their own ops in there.

`python3 add_op2.py --type <type> op_name1 ...`

## Introduction

Generally, Acuity Tools can generate Customer Layer's stub codes in App, when it meet unknown layer. But sometimes you will want to add customer layer to ovxlib, so it can be used as a built-in layer.

One Customer Layer includes these files:(For example, we add foo op as a customer layer)
 - src/custom/ops/vsi_nn_op_foo.c & include/custom/ops/vsi_nn_op_foo.h
 - src/custom/ops/kernel/evis/custom_foo_cpu.c
 - src/custom/ops/kernel/evis/custom_foo_evis.c
 - src/libnnext/ops/vx/foo.vx (shader kernel needed)

If you want to add CPU kernel to ovxlib, you can reference the code of clip op.

If you want to add Shader kernel to ovxlib, you can reference the code of clip op.

## Step 1

Copy the reference op files, and rename reference op to foo op both file name and file contexts.

Add these new foo op files to project files:
 - ovxlib.vcxproj & ovxlib.vcxproj.filters (for VS build)
 - BUILD (for bazel Build)
 - src/makefile.linux (for make build)
 - src/Android.mk (for Android build)

## Step 2

Add `DEF_OP(FOO)` to the end of include/custom/custom_ops.def.

Add new item for foo op in array:
 - src/utils/vsi_nn_code_generator.c: s_op_gen[]
 - src/vsi_nn_node_attr_template.c: s_template[]

## Key Points

### src/custom/ops/vsi_nn_op_foo.c: op_setup

Compute the ouput tensor's shape.

### src/custom/ops/kernel/cpu/custom_foo_cpu.c: _foo_exec

The CPU implement of foo op.

### include/custom/custom_node_type.h:DEF_NODE_TYPE(foo)

If foo op have some parameters, DO NOT forget add DEF_NODE_TYPE(foo).

## Add Shader Kernel

### src/custom/ops/kernel/evis/custom_foo_evis.c: _foo_initializer

We use _foo_initializer function to set the information of kernel to the OpenVX driver.

### tools/build_vx_files.py

After update any shader script in src/libnnext/ops/vx, you MUST use tools/build_vx_files.py to generate src/libnnext/vsi_nn_libnnext_vx.c, which includes ALL shader scripts' text. So we don't need these shader scripts at runtime anymore.

If you only build .vx files for embeded op, You can:

`python3 build_vx_files.py --type embedded`

If you build .vx files for both embeded op and custom op, You can:

`python3 build_vx_files.py --type custom`
