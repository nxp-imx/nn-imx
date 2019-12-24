#/bin/bash
# This script is used to copy android nnrt binaries and header files
# from android out folder to vendor/nxp/fsl-proprietary.
# And auto generate a commit which contains current branch and commit info.
# Pre-condition:
# Need source/lunch under android source code root directory firstly.
# Need setup a branch to track remote branch and local branch name must be
# same as remote one.
# This script assmues every commit contains "Signed-off-by:***".

function update_lib()
{
    if [ -e "$SRC_PWD/lib/$1" ];then
        cp $SRC_PWD/lib/$1   $DST_PWD/gpu-viv/lib/$1 && \
        echo "update the file $DST_PWD/gpu-viv/lib/$1 based on $SRC_PWD/lib/$1"
    else
        echo "$SRC_PWD/lib/$1 not exits!"
    fi

    if [ -e "$SRC_PWD/lib64/$1" ];then
        cp $SRC_PWD/lib64/$1 $DST_PWD/gpu-viv/lib64/$1 && \
        echo "update the file $DST_PWD/gpu-viv/lib64/$1 based on $SRC_PWD/lib64/$1"
    else
        echo "$SRC_PWD/lib64/$1 not exits!"
    fi
}

function update_hw_service()
{
    if [ -e "$SRC_PWD/bin/hw/$1" ];then
        cp $SRC_PWD/bin/hw/$1   $DST_PWD/gpu-viv/hal/$1 && \
        echo "update the file $DST_PWD/gpu-viv/hal/$1 based on $SRC_PWD/bin/hw/$1"
    else
        echo "$SRC_PWD/bin/hw/$1 not exits!"
    fi
}

function update_file()
{
    cp -af $DRIVER_PWD/$1  $DST_PWD/$2
}

# Once there's anything need to be copied, need add it here.
function update_all()
{
    update_lib libovxlib.so
    update_lib libnnrt.so
    update_hw_service android.hardware.neuralnetworks@1.2-service-vsi-npu-server
    update_file "nn_runtime/android_hal/VsiDriver.rc" "gpu-viv/hal/"
}

# Auto generate a commit which contains current branch and commit info.
# Commit message format is as below:
# MA-**** ****
# Source git: nnrt-imx.git
# Source branch: remotes/origin/****
# Source commit: ****
function auto_commit()
{
    COMMIT_ID=`git rev-parse HEAD`
    COMMIT_BRANCH=`git rev-parse --abbrev-ref HEAD`
    TEMP_FILE=`pwd`/temp.txt
    git log -1 --pretty='%B' > ${TEMP_FILE}
    sed -i "s/Change-Id:.*//g" ${TEMP_FILE}
    sed -i "s/Signed-off-by.*/Source git: nnrt-imx.git\nSource branch: remotes\/origin\/${COMMIT_BRANCH}\nSource commit: ${COMMIT_ID}/g" ${TEMP_FILE}
    sed -i "s/ \[#imx-.*\]//g" ${TEMP_FILE}

    cd $DST_PWD/
    git add *
    git commit -s -F ${TEMP_FILE}
    rm -rf ${TEMP_FILE}
}

if [ -z $OUT ] && [ -z $ANDROID_BUILD_TOP ];then
    echo "Env is not ready! Please go to android source code root directory:"
    echo "source build/envsetup.sh"
    echo "lunch"
    exit
else
    SRC_PWD=$OUT/vendor
    DST_PWD=$ANDROID_BUILD_TOP/vendor/nxp/fsl-proprietary
    DRIVER_PWD=`pwd`
    update_all
    auto_commit
fi

