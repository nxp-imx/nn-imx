
OUT_BIN=/lhome/dandan.he/android/rk3328_box_sdk/out/target/product/rk3328_box/system/bin/android.hardware.neuralnetworks\@1.0-service-ovx-driver
OUT=$1

mkdir $OUT

date

ls -l $OUT_BIN

echo ""
echo "cp $OUT_BIN $OUT"
echo ""

date


cp $OUT_BIN $OUT

chmod a+x $OUT

adb push $OUT_BIN /vendor/misc/hw
adb shell chmod a+x /vendor/misc/hw/
adb shell ls -l /vendor/misc/hw

ls -l $OUT
