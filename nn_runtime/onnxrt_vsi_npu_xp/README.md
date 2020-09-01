## Integration Guild

ONNXRuntime base version: V1.1.2
*Precondition*
> you should build libnnrt.so, libovxlib.so and other driver libraries before intergate vsi_npu execution provider

### make sure you can compile onnxruntime v1.1.2 with your toolchain

### Enable vsi_npu execution provider
```sh
cp -r include/ $(ONNX_RT_PATH)/include/onnxruntime/core/providers/vsi_npu/
cp -r src/ $(ONNX_RT_PATH)/onnxruntime/core/providers/vsi_npu/
cp -r toolchain/ $(ONNX_RT_PATH)/
git apply ./0001-VSI_NPU-the-patch-for-ONNXRuntime-framework.patch
git apply ./0002-fix-bug-for-ONNXRuntime-v1.1.2.patch
export NNRT_ROOT=/path/to/your/nn_runtime/root/dir
export VIVANTE_SDK_DIR=/driver/root/build/sdk # driver build folder
```

### Build
``` sh
cd onnxruntime/root/folder
./build.sh --use_vsi_npu --cmake_toolchain --build_shared_lib # other options
```

*NOTE* vsi_npu execution provider static link to runtime in current version

## Supported model/op

Verified model : squeezenet, mobilenet_v2

Operation Support status go to vsi_npu_ort_interpreter.cc
```cpp
std::map<std::string, SetupFunc> vsi_npu_supported_ops = {
  REGISTER_OP(Relu),
  REGISTER_OP(Abs),
  REGISTER_OP(Conv),
  REGISTER_OP(Concat),
  REGISTER_OP(MaxPool),
  REGISTER_OP(GlobalMaxPool),
  REGISTER_OP(AveragePool),
  REGISTER_OP(GlobalAveragePool),
  REGISTER_OP(Softmax),
  REGISTER_OP(Add),
  REGISTER_OP(Sub),
  REGISTER_OP(Mul),
  REGISTER_OP(Div),
  REGISTER_OP(Reshape),
  REGISTER_OP(Gemm),
  };
```
