SET SolutionDir=%1
SET ProjectDir=%2

SET SolutionDir=%SolutionDir:"=%
SET ProjectDir=%ProjectDir:"=%
SET DEST_FILE=%SolutionDir%ovxlib.path.user
SET TMP_FILE=%SolutionDir%ovxlib.path.user.tmp

ECHO %ProjectDir% > "%TMP_FILE%"
IF EXIST %DEST_FILE% (
    FC "%TMP_FILE%" "%DEST_FILE%" > nul 2>&1 || MOVE /Y "%TMP_FILE%" "%DEST_FILE%"
    DEL /F /Q "%TMP_FILE%" > nul 2>&1
) ELSE (
    MOVE /Y "%TMP_FILE%" "%DEST_FILE%"
)
SET DEST_FILE=%ProjectDir%include\vsi_nn_feature_config.h
SET TMP_FILE=%ProjectDir%include\vsi_nn_feature_config.h.tmp
SET FEATURE_CONFIG_SRC=%ProjectDir%vsi_feature_config
IF "%FEATURE_CONFIG_SRC%" NEQ "%FEATURE_CONFIG_SRC: =%" (
    ECHO "ERROR: The path of '%FEATURE_CONFIG_SRC%' contains spaces"
    EXIT /b -2
)

ECHO Auto generate feature config header...
ECHO /*****Auto generated header file, Please DO NOT modify manually!*****/> "%TMP_FILE%"
ECHO #ifndef _VSI_NN_FEATURE_CONFIG_H>> "%TMP_FILE%"
ECHO #define _VSI_NN_FEATURE_CONFIG_H>> "%TMP_FILE%"
ECHO.>> "%TMP_FILE%"
FOR /f "delims=" %%i in (%FEATURE_CONFIG_SRC%) do (
    ECHO %%i>> "%TMP_FILE%"
)
ECHO.>> "%TMP_FILE%"
ECHO #endif>> "%TMP_FILE%"

IF EXIST %DEST_FILE% (
    ATTRIB -R %DEST_FILE%
    FC "%TMP_FILE%" "%DEST_FILE%" > nul 2>&1 || MOVE /Y "%TMP_FILE%" "%DEST_FILE%"
    DEL /F /Q "%TMP_FILE%" > nul 2>&1
) ELSE (
    MOVE /Y "%TMP_FILE%" "%DEST_FILE%"
)
ECHO Generate feature config header to %DEST_FILE% successfully.
