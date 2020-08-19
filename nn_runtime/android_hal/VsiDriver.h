/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#ifndef ANDROID_ML_NN_VSI_DRIVER_H
#define ANDROID_ML_NN_VSI_DRIVER_H

#include "VsiDevice.h"
#include "HalInterfaces.h"
#include "Utils.h"
#include "CustomizeOpSupportList.h"
#include "WeightMd5CheckingConfig.h"

#include <sys/system_properties.h>
#include <android-base/logging.h>
#include <hidl/LegacySupport.h>
#include <thread>
#include <string>


namespace android {
namespace nn {
namespace vsi_driver {

static const std::vector<std::string> weight_md5 = {
    // vgg_float
    "0F308490FCD1009E922B340CEA5EE65B",
    "0C67D9A824D0C46D253395D7000A845B",
    "B547994B34A10F08847B6990CBDC1B90",
    "FE9981A325CE7451DF79F7CB801747DD",
    "909DDB66694AAE5F7E66E6A563596593",
    "80718DD8E3EE6050CC071D550AEEBFB2",
    "E8BCFC59FDAC1A6E2CA5F1723F8A05EC",
    "3A8B9BDB468724F536FEAF026CDA4EB4",
    "5DE10DD65C6510674B663FCFBBF13687",
    "1004D68474FC61490AFA267B21F70BBF",
    "470DDD89D86245603E3F6DBDBED89C0A",
    "16040F69CDCBF85F0E1BFE07296D48A6",
    "BE7D6E99C5C91D65412D4DEFF685481A",
    "024EEF3ABEF288D11B31C7395BF0596B",
    "3108C3EF41CAF3402B51199F4FBF7983",
    "FFEDD3DD01A32A7F2318DEEB46DA7C62",
    "8CDED202E872B97734E44DFFD3060163",
    "69F8BE71509C2F3397E80110ECF79786",

    // vgg_quant
    "3211BFCF380B68DE5D4D4DE31F455437",
    "9D3C4886E40CC739AA57277DA834AB49",
    "3194C1AE2328FDC6880C8C2AE264F1FB",
    "5C43BFF7AA9DEB2B172648351C6E7B6D",
    "2386297B3F240A1DC3E36FA49D87F423",
    "1DDB12DDA796A4BBAEB3B396534CFE5A",
    "E6CEE8FD56F5A1C9B1853892FA1D6291",
    "AB63F9D4BAC6D8370208AF841C4B44AD",
    "AB778BCC4862BF18BA3B44D695A908EE",
    "DD9F0E63DB950F963FBEA159FA4C419D",
    "0D7DEE3D17D149A60A121259C42DE167",
    "7880DE116BF1DCDCBFF1092C63BD4139",
    "7F645F5E4AA1BFB70CDDE718D52F7B22",
    "C7A9744B679A9D2069E489D43D279A19",
    "440BC68AA48D4126518559AE3500DF69",
    "8E4A887CF0111CED45DC61C0D22B5A90",
    "8A54333F7EA4418B8C512BBB9233E451",
    "8DAFD0FFBD3DD2C840B6C93EBBBAB1F8",

    // srgan_float
    "5639FA9397C762A043BAFBED8A370E21",
    "082433233B6B77115C343870590B95DE",
    "4968C2E9AD3834F3EB6ED8C955112B42",
    "F4D08991D32E979E0A3AFA256557944D",
    "3B6D51436689150B8DE8F555131B5B34",
    "62483F09B60BA3B034BF494C38A85744",
    "AE24503595F4886E3D1AE3E387F34AE1",
    "8AC390EA0D485377497E61B3D230473D",
    "1B06D274CA2A9F356798E288A88E030E",
    "1F069EC28BBBDE674F02AC477ED6D7BB",
    "759C66F03FB714FE6D9B13525A02485D",
    "C568475814E7BBE625C81B307C1710BE",
    "BE85256115D8B55FED35AD6ED6806845",
    "D1C80E5C8683296A46717152454ACC87",
    "76C6661E30AE8C8D4B7AB2274ABA850C",
    "0DB6266F34F400C7A835F8B1AB6ACA55",
    "03BE1B4718ED607E37945AB85C47CF82",
    "41A6A1721C168C4C53BE3F2BAE38B942",
    "02CA4DCF900795062EDD351B7CDE1388",
    "8A84982938B1D4E450EE2B56141B34B8",
    "CC4E867D4DDACCFA80BD5B74C6FFA6CF",
    "82E7E981C7918900017C1F110AD4C382",
    "1D774830EDF262F37AEB33B914072E29",
    "DAAB037FA3C0CC1803D202A87B2E57EB",
    "7B6305526A557D7FB7E33DEDFDFE4E9F",
    "803F5D3F0A9AFEBB368B56A4F19C0A0B",
    "49C4EC9A3F292178247FE00F2D5B3FA8",
    "E498F99F30CB5CC542542E32462E9427",
    "47500A4460B04469E089EEE6E62CDE0D",
    "B88909E466D941972480FD39255A2BBC",
    "04649EDF3FFE31EBA58810819799E686",
    "746203379C6B1D0C60A82F81DD970C6D",
    "B241E7F4A1B99A47F2BA8D862D063912",

    // srgan_quant
    "3423DDC5CB926340E969D292E134D925",
    "F69CFF631DF8E0ABD00D74746E6F632A",
    "1647B4DBC497DB3B6577ECBF30FE213C",
    "E5074ED2278F12426662873537CDB5B5",
    "E8DDA55D76A12F9FED3A58369549F011",
    "28FA4649726C67C7B802E5FC05A9CB12",
    "7D3F3DADC90AEA556A42316413F4E980",
    "2FAF4117D49322582D033F73BDBAC418",
    "EF5A0AC4C621CA15237583E4B3484829",
    "BEB3A2C1FCCA7F0D4CD5CAB837FDFCF1",
    "51F51381BD41578F47936945BE0A4CB5",
    "50644D314D5619999AB6D382E0E7D79C",
    "ABE8012830411B2778B7D667C4FF9F3B",
    "C0F625339B714EF77B29B510794ADA4E",
    "6F7BDB503C1783E101B3DE9A7484018B",
    "79D4E046A297719EB6B98FE6C68CB638",
    "EC794CF10B6862F2B094B03279247C7A",
    "32D3BBDAD59A15BCE01148DCC5360A0D",
    "4C6125FFF3559E48D675C85328C62851",
    "D8557035BF2AB9E3D132463FB40E578E",
    "B7E317A053EF6124B13A05201075382A",
    "3D0B82C19392153E7164A490DB726C6A",
    "5B1B6E8877BF9712087514DEACC00A4F",
    "D2DCBAD98D5D12400EBF5D52FDC0A815",
    "C472B0DACA4A3F6C86D29645AD8FC1DD",
    "93FF74C6117F1479B8586C1917F4DB44",
    "0ABB91DC2C5ED2DA0BAAF1E584B5B380",
    "EF02F13E1A3A59AA796631ACD241168F",
    "4B51E0F0C355727E2BBF36DF951B530E",
    "70CAFF45E026047BB9D9DFBCFC7C861F",
    "ADFA3C3AE51140A7C7651F4896434666",
    "CB093897870EBA1B55BDDE369FFA1071",
    "BABED20F5A1988EA4486510D0FFEAB6C",

    // dped_float
    "A59B0D480DFBA4FBC09D2F2EB1095668",
    "2998EC9356328DACF2F27F1BBA153470",
    "1A8E23645CF434F5FEC0E1BDC51CADC9",
    "B933FE40792D6B6867CF3D99275FC0CA",
    "A9734984FE40B695E511A83A030CCEF6",
    "169608F5079A780B66BAEF3C918B9F3D",
    "DC8991DA8C8CC4F52BA34209D35A77B8",
    "978F3BE63AB54755DD17E53E291E51CE",
    "8AD514778F1B76DF28AEA51EEB323A00",
    "4187083B2E58D0BDA414C128057D7333",

    // inception_v3_float
    "CA7BC170719FEC8FE04EC685FEBF8CA7",
    // inception_v3_quant
    "E494867C3D9B911F9C401531E4C97390",

    // inception_face_float
    "3C1812918369446168A511D2E2207599",
    // inception_face_quant
    "3071BB800BEE3652D7F38BEF08EE40DE",

    // icnet_float
    "31A7681CDBA02DEB562E4D3D9C367CB7",
    "0B1C4B3EE9E7D94B9A89A830F727D44B",
    "66379BC572694207D38B5351689934AB",
    "4D9A66570895DA7F324629B120353197",
    "6A8A44E021ACD9FFAFF69EB03C8CCB22",
    "6EB4A5B7FB4735A5817A970E154D5A7D",

    // srcnn_float
    "14EE8F7B5C4492159470CBD66CFE545A",

    // srcnn_quant
    "329F391220862B8CAB0AA46C6C440793",

    // srcnn_200-2000
    "A7C4E8420FF11F9A40E8EE324312FC9F",

    // mobilenet_v2_float
    "303C6EAA6EAEC4483B684E4D43AAD139",
    // mobilenet_v2_quant
    "0860871B70644D172A977254799C9306",

    // unet
    "637EA97C1E00783AEB6005005A0822D8",
};

class VsiDriver : public VsiDevice {
   public:
    VsiDriver() : VsiDevice("vsi-npu") {initalizeEnv();}
    Return<void> getCapabilities(getCapabilities_cb _hidl_cb) ;
    Return<void> getSupportedOperations(const V1_0::Model& model, V1_0::IDevice::getSupportedOperations_cb cb) ;

#if ANDROID_SDK_VERSION >= 28
    Return<void> getCapabilities_1_1(V1_1::IDevice::getCapabilities_1_1_cb _hidl_cb) ;
    Return<void> getSupportedOperations_1_1(const V1_1::Model& model,
                                                  V1_1::IDevice::getSupportedOperations_1_1_cb cb) ;
#endif

#if ANDROID_SDK_VERSION >= 29
    Return<void> getCapabilities_1_2(V1_2::IDevice::getCapabilities_1_2_cb _hidl_cb) ;
    Return<void> getSupportedOperations_1_2( const V1_2::Model& model,
                                                   V1_2::IDevice::getSupportedOperations_1_2_cb cb) ;
#endif
    static bool isSupportedOperation(const HalPlatform::Operation& operation,
                                     const HalPlatform::Model& model,
                                     std::string& not_support_reason);

    static const uint8_t* getOperandDataPtr( const HalPlatform::Model& model,
                                             const HalPlatform::Operand& hal_operand,
                                             VsiRTInfo &vsiMemory);

    bool isWeightMd5Matched(const HalPlatform::Operation& operation, const HalPlatform::Model& model);

   private:
   int32_t disable_float_feature_; // switch that float-type running on hal
   private:
    void initalizeEnv();

    template <typename T_model, typename T_getSupportOperationsCallback>
    Return<void> getSupportedOperationsBase(const T_model& model,
                                            T_getSupportOperationsCallback cb){
        LOG(INFO) << "getSupportedOperations";
        bool is_md5_matched = false;
        bool is_md5_check_env_set = false;
        char env[100] = {0};
        int32_t read_env = __system_property_get("WEIGHT_MD5_CHECK", env);
        if (read_env) {
            is_md5_check_env_set = (atoi(env) == 1 ? true : false);
        }
        if (validateModel(model)) {
            const size_t count = model.operations.size();
            std::vector<bool> supported(count, true);
            std::string notSupportReason = "";
            for (size_t i = 0; i < count; i++) {
                const auto& operation = model.operations[i];
                if (weight_md5_check || is_md5_check_env_set) {
                    if ((is_md5_matched = isWeightMd5Matched(operation, model))) break;
                }
                supported[i] = !IsOpBlocked(static_cast<int32_t>(operation.type)) &&
                               isSupportedOperation(operation, model, notSupportReason);
            }
            if (is_md5_matched) {
                LOG(INFO) << "Weight MD5 matched, reject the whole model.";
                for (size_t i = 0; i < count; ++i) {
                    supported[i] = false;
                }
            }
            LOG(INFO) << notSupportReason;
            cb(ErrorStatus::NONE, supported);
        } else {
            LOG(ERROR) << "invalid model";
            std::vector<bool> supported;
            cb(ErrorStatus::INVALID_ARGUMENT, supported);
        }
        LOG(INFO) << "getSupportedOperations exit";
        return Void();
    };

};

}
}
}
#endif
