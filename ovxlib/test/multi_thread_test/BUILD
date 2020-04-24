filegroup(
    name = "lenet_data",
    srcs = ["lenet.data"]
)

cc_test (
    name = "multi_thread_test",
    copts = ["-Werror", "-fvisibility=hidden", '-DNNTEST_ONLY_PUBLIC_API'],
    linkopts = ["-pthread"],
    visibility = ['//visibility:public'],
    srcs = [
        "multi_thread_test.cc",
        ],
    deps = [
        "@OVXLIB//:ovxlib",
    ],
    args = ["$(location :lenet_data)"],
    data = [":lenet_data"],
)
