echo "Please set your compile toolchain first"
export AQROOT=`pwd`
export CC="ccache $CC"
export CXX="ccache $CXX"
export CPP="ccache $CPP"
. /opt/fsl-imx-internal-xwayland/5.4-zeus/environment-setup-aarch64-poky-linux
make install -j8
