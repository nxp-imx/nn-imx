#/****************************************************************************
#*
#*    Copyright (c) 2019 Vivante Corporation
#*
#*    Permission is hereby granted, free of charge, to any person obtaining a
#*    copy of this software and associated documentation files (the "Software"),
#*    to deal in the Software without restriction, including without limitation
#*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
#*    and/or sell copies of the Software, and to permit persons to whom the
#*    Software is furnished to do so, subject to the following conditions:
#*
#*    The above copyright notice and this permission notice shall be included in
#*    all copies or substantial portions of the Software.
#*
#*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#*    DEALINGS IN THE SOFTWARE.
#*
#*****************************************************************************/

package(
    default_visibility = ["//visibility:public"],
    features = ["-parse_headers"],
)

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

filegroup(
    name = "ovxArmnnBackendWorkload_files",
    srcs = glob([
        "workloads/*.hpp",
        "workloads/*.cpp",
    ])
)

filegroup(
    name = "ovxArmnnBackendImpl_files",
    srcs = glob([
        "*.hpp",
        "*.cpp",
    ])
)
filegroup(
    name = "ovxArmnnBackendImpl_headers",
    srcs = glob([
        "*.hpp",
    ])
)

cc_library(
    name = "armnn_ovx_backend",
    alwayslink = True,
    copts = ['-std=c++1z'],
    includes = [
        ".",
        "workloads/",
    ],
    hdrs =  ["ovxArmnnBackendImpl_headers"],
    srcs =  ["ovxArmnnBackendImpl_files"] +
            ["ovxArmnnBackendWorkload_files" ],
    deps = ["@OVXLIB//:ovxlib",
            "//nnrt:nnrt",
            "@armnn//:armnn"
    ]
)

cc_binary(
    name = "sample",
    linkstatic = False,
    copts = ['-std=c++1z'],
    srcs = ["@armnn//:sample"],
    deps = ["armnn_ovx_backend"] +
           ["@boost//:boost_filesystem"] +
           ["@boost//:boost_system"],
    linkopts = ["-lpthread"],
)

filegroup(
    name = "backend_ut_files",
    srcs = glob([
        "test/*.hpp",
        "test/NpuLayerTests.cpp",
        "test/NpuCreateWorkloadTests.cpp",
        "test/NpuEndToEndTests.cpp",
        "test/NpuLayerSupportTests.cpp"
    ])
)

cc_binary(
    name = "unit_test",
    linkstatic = True,
    copts = ["-std=c++1z"],
    includes = ["."],
    srcs = ["backend_ut_files"],
    deps =  ["armnn_ovx_backend"] +
            ["@armnn//:backend_common_ut"] +
            ["@boost//:boost_system"] +
            ["@boost//:boost_unit_test_framework"] +
            ["@boost//:boost_filesystem"] ,
    linkopts = ["-lpthread"],
)

cc_test(
    name = "ci-unit_test",
    srcs = ["backend_ut_files"],
    copts = ["-std=c++1z"],
    deps =  ["armnn_ovx_backend"] +
            ["@armnn//:backend_common_ut"] +
            ["@boost//:boost_system"] +
            ["@boost//:boost_unit_test_framework"] +
            ["@boost//:boost_filesystem"] ,
    linkopts = ["-lpthread"],
    testonly = False,
    visibility = ['//visibility:public'],
)

pkg_tar(
    name = "armnn-ut_pkg",
    extension = "tar.gz",
    srcs = [
        ":ci-unit_test",
        "@VIV_SDK//:VIV_EXT_HDR",
    ],
    deps = [
        "//:base_pkg",
    ],
    package_dir = "armnn-ut_pkg",
)