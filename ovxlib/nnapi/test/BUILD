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

