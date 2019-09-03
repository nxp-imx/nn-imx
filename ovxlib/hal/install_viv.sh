OUT=$1
ANDROID_OUT=/lhome/dandan.he/android/rk3328_box_sdk/out/target/product/rk3328_box

mkdir $OUT

date 

ls -l $ANDROID_OUT/system/lib64/libEmulator.so
ls -l $ANDROID_OUT/system/lib64/libOpenCL.so
ls -l $ANDROID_OUT/system/lib64/libGAL.so
ls -l $ANDROID_OUT/system/lib64/libVSC.so
ls -l $ANDROID_OUT/system/lib64/libVivanteOpenCL.so
ls -l $ANDROID_OUT/system/lib64/libOpenVX.so
ls -l $ANDROID_OUT/system/lib64/libOpenVXU.so
#ls -l $ANDROID_OUT/system/lib64/libnn.so
ls -l $ANDROID_OUT/system/lib64/libOpenVXC.so
ls -l $ANDROID_OUT/system/lib64/libLLVM_viv.so
ls -l $ANDROID_OUT/system/lib64/libCLC.so


echo "cp $ANDROID_OUT/system/lib64/libEmulator.so $OUT/"
cp $ANDROID_OUT/system/lib64/libEmulator.so $OUT/

echo "cp $ANDROID_OUT/system/lib64/libOpenCL.so $OUT/"
cp $ANDROID_OUT/system/lib64/libOpenCL.so $OUT/

echo "cp $ANDROID_OUT/system/lib64/libGAL.so $OUT/"
cp $ANDROID_OUT/system/lib64/libGAL.so $OUT/

echo "cp $ANDROID_OUT/system/lib64/libVSC.so $OUT/"
cp $ANDROID_OUT/system/lib64/libVSC.so $OUT/

echo "cp $ANDROID_OUT/system/lib64/libVivanteOpenCL.so $OUT/"
cp $ANDROID_OUT/system/lib64/libVivanteOpenCL.so $OUT/

echo "cp $ANDROID_OUT/system/lib64/libOpenVX.so $OUT/"
cp $ANDROID_OUT/system/lib64/libOpenVX.so $OUT/

echo "cp $ANDROID_OUT/system/lib64/libOpenVXU.so $OUT/"
cp $ANDROID_OUT/system/lib64/libOpenVXU.so $OUT/

#echo "cp $ANDROID_OUT/system/lib64/libnn.so $OUT/"
#cp $ANDROID_OUT/system/lib64/libnn.so $OUT/

echo "cp $ANDROID_OUT/system/lib64/libOpenVXC.so $OUT/"
cp $ANDROID_OUT/system/lib64/libOpenVXC.so $OUT/

echo "cp $ANDROID_OUT/system/lib64/libLLVM_viv.so $OUT/"
cp $ANDROID_OUT/system/lib64/libLLVM_viv.so $OUT/

echo "cp $ANDROID_OUT/system/lib64/libCLC.so $OUT/"
cp $ANDROID_OUT/system/lib64/libCLC.so $OUT/

date

ls -rlt $OUT

echo ""
echo "Install driver to device ..."

adb push $OUT/* /system/lib64/


