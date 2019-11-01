cc_test (
    name = "cts_p",
    copts = ["-Werror", "-fvisibility=hidden", '-std=c++1z', '-DNNTEST_ONLY_PUBLIC_API'],
    srcs = [
        "TestMain.cpp",
        ],
    deps = [
        "@android_ml_9_0//:cts_lib",
    ]
)

cc_test (
    name = "cts_o",
    copts = ["-Werror", "-fvisibility=hidden", '-std=c++1z', '-DNNTEST_ONLY_PUBLIC_API'],
    srcs = [
        "TestMain.cpp",
        ],
    deps = [
        "@android_ml_8_1//:cts_lib",
    ]
)

cc_test (
    name = "cts_q",
    copts = ["-Werror", "-fvisibility=hidden", '-std=c++1z', '-DNNTEST_ONLY_PUBLIC_API', '-D__ANDROID_API__=29'],
    srcs = [
        "TestMain.cpp",
        ],
    deps = [
        "@android_ml_q//:cts_lib",
        "@//third-party/nn-cts/android_stubs:android_stubs",
    ]
)

cc_test (
    name = "execution_async",
    copts = ["-Werror", "-fvisibility=hidden", '-DNNTEST_ONLY_PUBLIC_API'],
    linkopts = ["-pthread"],
    srcs = [
        "execution_async_test.cc",
        ],
    deps = ["//nnapi:nnapi"],
)
