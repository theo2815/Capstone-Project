// JNI wrapper over Fujifilm's Camera Control SDK (XAPI.so). Marshalling only; every
// function returns the SDK's long result (0 = complete). Compiled only in builds where
// mobile/fuji-sdk/ exists (see app/build.gradle.kts).
#include <jni.h>
#include <dlfcn.h>
#include <cstring>
#include "XAPI.H"

typedef long (*init_fn)(long);
typedef long (*exit_fn)();
typedef long (*open_fn)(long, unsigned char*, void*);
typedef long (*handle_fn)(void*);
typedef long (*errno_fn)(void*, long*, long*);
typedef long (*devinfo_fn)(void*, void*);
typedef long (*getlong_fn)(void*, long*);
typedef long (*setlong_fn)(void*, long);
typedef long (*imginfo_fn)(void*, void*);
typedef long (*readimg_fn)(void*, unsigned char*, unsigned long);

static void* lib;
static init_fn xInit; static exit_fn xExit; static open_fn xOpen; static handle_fn xClose;
static errno_fn xErr; static devinfo_fn xDevInfo; static getlong_fn xMode; static getlong_fn xGetRec;
static setlong_fn xSetRec; static imginfo_fn xImgInfo; static readimg_fn xReadImg; static handle_fn xDelImg;

#define SYM(var, name) var = (decltype(var)) dlsym(lib, name); if (!var) return -1;

extern "C" JNIEXPORT jlong JNICALL
Java_com_quickpitik_mobile_data_usb_fuji_FujiSdkNative_nativeLoad(JNIEnv*, jobject) {
    if (!lib) lib = dlopen("XAPI.so", RTLD_LAZY);
    if (!lib) return -1;
    SYM(xInit, "XSDK_Init") SYM(xExit, "XSDK_Exit") SYM(xOpen, "XSDK_SetUSBDeviceHandle") SYM(xClose, "XSDK_Close")
    SYM(xErr, "XSDK_GetErrorNumber") SYM(xDevInfo, "XSDK_GetDeviceInfo") SYM(xMode, "XSDK_GetMode")
    SYM(xGetRec, "XSDK_GetMediaRecord") SYM(xSetRec, "XSDK_SetMediaRecord") SYM(xImgInfo, "XSDK_ReadImageInfo")
    SYM(xReadImg, "XSDK_ReadImage") SYM(xDelImg, "XSDK_DeleteImage")
    return 0;
}
extern "C" JNIEXPORT jlong JNICALL
Java_com_quickpitik_mobile_data_usb_fuji_FujiSdkNative_nativeInit(JNIEnv*, jobject) { return xInit(0); }
extern "C" JNIEXPORT jlong JNICALL
Java_com_quickpitik_mobile_data_usb_fuji_FujiSdkNative_nativeExit(JNIEnv*, jobject) { return xExit(); }
extern "C" JNIEXPORT jlong JNICALL
Java_com_quickpitik_mobile_data_usb_fuji_FujiSdkNative_nativeOpen(JNIEnv* env, jobject, jint fd, jbyteArray raw, jlongArray out) {
    jbyte* bytes = env->GetByteArrayElements(raw, nullptr);
    void* camera = nullptr;
    long res = xOpen(fd, reinterpret_cast<unsigned char*>(bytes), &camera);
    env->ReleaseByteArrayElements(raw, bytes, JNI_ABORT);
    jlong handle = reinterpret_cast<jlong>(camera);
    env->SetLongArrayRegion(out, 0, 1, &handle);
    return res;
}
extern "C" JNIEXPORT jlong JNICALL
Java_com_quickpitik_mobile_data_usb_fuji_FujiSdkNative_nativeClose(JNIEnv*, jobject, jlong h) { return xClose(reinterpret_cast<void*>(h)); }
extern "C" JNIEXPORT jlong JNICALL
Java_com_quickpitik_mobile_data_usb_fuji_FujiSdkNative_nativeError(JNIEnv* env, jobject, jlong h, jlongArray out) {
    long api = 0, err = 0;
    long res = xErr(reinterpret_cast<void*>(h), &api, &err);
    jlong values[2] = { api, err };
    env->SetLongArrayRegion(out, 0, 2, values);
    return res;
}
extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_quickpitik_mobile_data_usb_fuji_FujiSdkNative_nativeDeviceInfo(JNIEnv* env, jobject, jlong h) {
    XSDK_DeviceInformation info;
    memset(&info, 0, sizeof(info));
    if (xDevInfo(reinterpret_cast<void*>(h), &info) != 0) return nullptr;
    const char* fields[4] = { info.strVendor, info.strProduct, info.strSerialNo, info.strFirmware };
    jobjectArray result = env->NewObjectArray(4, env->FindClass("java/lang/String"), nullptr);
    for (int i = 0; i < 4; i++) env->SetObjectArrayElement(result, i, env->NewStringUTF(fields[i]));
    return result;
}
extern "C" JNIEXPORT jlong JNICALL
Java_com_quickpitik_mobile_data_usb_fuji_FujiSdkNative_nativeGetLong(JNIEnv* env, jobject, jlong h, jint what, jlongArray out) {
    long value = 0;
    long res = (what == 0 ? xMode : xGetRec)(reinterpret_cast<void*>(h), &value);
    jlong v = value;
    env->SetLongArrayRegion(out, 0, 1, &v);
    return res;
}
extern "C" JNIEXPORT jlong JNICALL
Java_com_quickpitik_mobile_data_usb_fuji_FujiSdkNative_nativeSetMediaRecord(JNIEnv*, jobject, jlong h, jlong mode) {
    return xSetRec(reinterpret_cast<void*>(h), (long) mode);
}
extern "C" JNIEXPORT jlong JNICALL
Java_com_quickpitik_mobile_data_usb_fuji_FujiSdkNative_nativeReadImageInfo(JNIEnv* env, jobject, jlong h, jlongArray out, jobjectArray name) {
    XSDK_ImageInformation info;
    memset(&info, 0, sizeof(info));
    long res = xImgInfo(reinterpret_cast<void*>(h), &info);
    jlong values[2] = { info.lFormat, info.lDataSize };
    env->SetLongArrayRegion(out, 0, 2, values);
    env->SetObjectArrayElement(name, 0, env->NewStringUTF(info.strInternalName));
    return res;
}
extern "C" JNIEXPORT jlong JNICALL
Java_com_quickpitik_mobile_data_usb_fuji_FujiSdkNative_nativeReadImage(JNIEnv* env, jobject, jlong h, jbyteArray into) {
    jsize size = env->GetArrayLength(into);
    jbyte* bytes = env->GetByteArrayElements(into, nullptr);
    long res = xReadImg(reinterpret_cast<void*>(h), reinterpret_cast<unsigned char*>(bytes), (unsigned long) size);
    env->ReleaseByteArrayElements(into, bytes, 0);
    return res;
}
extern "C" JNIEXPORT jlong JNICALL
Java_com_quickpitik_mobile_data_usb_fuji_FujiSdkNative_nativeDeleteImage(JNIEnv*, jobject, jlong h) { return xDelImg(reinterpret_cast<void*>(h)); }
