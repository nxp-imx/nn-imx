# AUTO GENERATED FILE, BUILD AND RUN IN OVXLIB

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs =
        [
        "GEMM_api.h",
        "GEMM_api.c",
        "gemm_sample.c",
        ],
)

cc_binary(
    name = "sample",
    srcs = [":srcs"] + ["//:ovxlib"],
    deps = [
        "//third-party/jpeg-9b:libjpeg",
        "//:ovxlib",
        "@VIV_SDK//:VIV_SDK_LIB",
    ],
)
