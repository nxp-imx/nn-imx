package(
    default_visibility = ["//visibility:public"],
    features = ["-parse_headers"],
)

cc_library(
    name = "nnapi-delegate",
    copts = ["-std=c++11", "-Werror", "-fvisibility=hidden"]
        + ['-DOVXLIB_API=__attribute__((visibility(\\"default\\")))', '-DANDROID_STUB'],
    includes = [
        ".",
    ],
    hdrs = [
        "NeuralNetworks.h",
    ],
    srcs = [
        "NeuralNetworks.cpp",
    ],
    deps = [
        "//nnrt:nnrt",
        "@//third-party/nn-cts/android_stubs:android_stubs",
    ],
)

cc_binary(
    name = "libneuralnetworks.so",
    copts = ["-std=c++11", "-Werror", "-fvisibility=hidden"]
        + ['-DOVXLIB_API=__attribute__((visibility(\\"default\\")))'],
    deps = ["nnapi-delegate"],
    linkshared=1,
)
