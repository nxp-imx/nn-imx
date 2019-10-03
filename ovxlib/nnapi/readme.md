
the old version NN Adapter and android HAL is deprecate. And relative feature
is based on libovxlib.so instead of libOpenVX.so, and refined in ovxlib pacakge.

the build step as below:

* LINUX
1. build vxc driver
2. build libovxlib.so
    ```
    cd ${ovxlib_dir}
    make -f makefile.linux  %% with your costom toolchain
    ```
3. build libneuralnetworks.so
    ```
    cd ${ovxlib_dir}/nnapi
    make -f makefile.linux  %% with your costom toolchain
    ```


* ANDROID

1. build step:
    1. build driver as usual.
    2. export AQROOT=/dirver/path/

        **note**: in android 9.0, driver path is relative path, but
        in android 8.0 or 8.1, driver path should be absolute path.
        for example:

        in android 9.0:

            dirver_dir path: {ANDROID_TOP_DIR}/viv_drv/driver}
            set env: export AQROOT=viv_drv/driver

        in android 8.0:

            dirver_dir path: ANDROID_TOP_DIR/viv_drv/driver}
            set env: export AQROOT=ANDROID_TOP_DIR/viv_drv/driver

    3. copy ovxlib directory to anywhere in {ANDROID_TOP_DIR}
    4. build libovxlib.so and hal server
        cd {ovxlib_dir}
        mm

    **note**:the server name has been changed to
        ```android.hardware.neuralnetworks@1.1-service-vsi-npu-server```

2. setup step:
    1. modified /system/etc/vintf/manifest.xml in chips.
    add:
    ```
    <hal format="hidl">
        <name>android.hardware.neuralnetworks</name>
        <transport>hwbinder</transport>
        <version>1.1</version>
        <interface>
            <name>IDevice</name>
            <instance>vsi-npu-server</instance>
        </interface>
        <fqname>@1.1::IDevice/vsi-npu</fqname>
    </hal>
    ```
    and then reboot the broad.

    2. adb push driver as usual, please include libovxlib.so

    3. set up android server and run CTS.