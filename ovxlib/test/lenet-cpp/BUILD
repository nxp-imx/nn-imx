
filegroup(
    name = "data",
    srcs = ["none_quantized.export.data"]
)

filegroup(
    name = "img",
    srcs = ["0.jpg"]
)

cc_test(
    name = "lenet-cpp",
    srcs = ["vnn_pre_process.cpp", "vnn_post_process.cpp", "main.cpp", "vnn_lenet.cpp"] + ["vnn_global.h", "vnn_lenet.h", "vnn_post_process.h", "vnn_pre_process.h"] + ["//:ovxlib"],
    deps = [
        "//third-party/jpeg-9b:libjpeg",
        "//:ovxlib",
        "@VIV_SDK//:VIV_SDK_LIB",
    ],
    visibility = ['//visibility:public'],

    args = [
        "$(location :data)",
        "$(location :img)"
    ],
    data = ["@VIV_SDK//:VIV_EXT_HDR"] + [
        ":data",
        ":img"
    ],
)
