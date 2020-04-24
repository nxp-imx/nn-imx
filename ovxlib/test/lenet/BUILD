
filegroup(
    name = "data",
    srcs = ["none_quantized.export.data"]
)

filegroup(
    name = "img",
    srcs = ["0.jpg"]
)

cc_test(
    name = "lenet",
    srcs = ["vnn_pre_process.c", "vnn_post_process.c", "main.c", "vnn_lenet.c"] + ["vnn_global.h", "vnn_lenet.h", "vnn_post_process.h", "vnn_pre_process.h"] + ["//:ovxlib"],
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
