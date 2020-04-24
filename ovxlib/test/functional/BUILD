
cc_test (
    name = "hashmap_test",
    copts = ["-Werror"],
    linkopts = ["-pthread"],
    visibility = ['//visibility:public'],
    srcs = [
        "hashmap.c",
        ] + ["//:ovxlib"],
    deps = [
            "//:ovxlib",
    ],
)

cc_test (
    name = "quantize_multiplier_test",
    copts = ["-Werror"],
    visibility = ['//visibility:public'],
    srcs = [
        "quantize_multiplier.c",
        ] + ["//:ovxlib"],
    deps = [
            "//:ovxlib",
    ],
)

cc_test (
    name = "data_convert",
    copts = ["-Werror"],
    visibility = ['//visibility:public'],
    srcs = [
        "data_convert.cc",
        ] + ["//:ovxlib"],
    deps = [
            "//:ovxlib",
    ],
)
