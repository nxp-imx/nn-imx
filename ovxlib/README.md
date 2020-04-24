# OVXLIB - A lightweight wrapper for OpenVX NN Extension

OVXLIB is a lightweight wrapper for OpenVX Neural-Network Extension provided by VeriSilicon to facilitate the deployment of Neural-Networks on OpenVX enabled hardware accelerators. It also contains special layer acceleration when used on Vivante Vision Image Processors.

Main Features
 - Allow dynamic insertion and removal of VX nodes and creation of VX graph
 - Simplified binding API calls to create NN Layers and Tensors
 - A set of utility functions for debugging
 - Built-in custom layer extensions

## Why OVXLIB?

Developing an OpenVX NN application can be difficult especially if one is not familiar with the API. Through OVXLIB's binding API abstraction, you can create a VX application with fewer lines of code. Also, inference engines often requires dynamic creation of an inference graph through compilation, this can be easily implemented through OVXLIB's wrapper interface, so that OpenVX can be adapted as a backend for TFlite, Android NN or systems alike. 

## Get started

### Get familiar with OpenVX spec
To use OVXLIB for development, you first need to get familiar with [OpenVX API](https://www.khronos.org/openvx/) and [OpenVX NN Extension API](https://www.khronos.org/registry/vx). Please head over to [Khronos](https://www.khronos.org/) to read the spec. 

### Build and Run
OVXLIB uses [bazel](https://bazel.build) build system by default, and you can adapt to your own build environment. [Install bazel](https://docs.bazel.build/versions/master/install.html) first to get started.

OVXLIB needs to be compiled and linked against the Vivante VIP SDK. The SDK provides VX related header files and pre-compiled libraries. The linux-x86_64 SDK is a simulator environment.  

Specify the SDK location and VIP configuration by modifying setup_env.sh

> SDK_PATH="/home/vip_user/VeriSilicon/VivanteIDE2.2.1/vcmdtools"   
> CONFIG="VIP8000_NANOD"   


Run setup_env.sh
```shell
source setup_env.sh
```

To build ovxlib
```shell
bazel build ovxlib
```

To run built-in test
```shell
bazel test //test/op:*
```
To run sample LeNet
```shell
bazel run //test/lenet:lenet
```

### Create real world NN applications
A real world NN application can be quite complex with hundreds of layers, and it is typically not reasonable to hand write such application. In reality, the inference graphs are either generated via code generator or dynamically created through an NN inference engine (for example, TFLite).

* Generating OVXLIB code

  VeriSilicon [ACUITY Toolkit](https://verisilicon.github.io/acuity-models/) can import a neural-network from Caffe or Tensorflow and generate inference code for OVXLIB. Check test/lenet as an example. 

* Inference Engine backend

  OVXLIB can be implemented as backend for either Tensorflow, TFLite, Android NN, or others alike. This will be made available in future updates. 

