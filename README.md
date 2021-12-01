## The Android demo of [Mediapipe Hand](https://google.github.io/mediapipe/solutions/hands)  infer by ncnn  

## Please enjoy the mediapipe hand demo on ncnn

You can try this APK demo https://pan.baidu.com/s/1ArAMH7uAic0cQJgOn-P-RQ pwd: jnrw  

https://github.com/Tencent/ncnn  
https://github.com/nihui/opencv-mobile
## palm model support:  
1.palm-lite  
2.palm-full  
## pose model support:  
1.hand-lite  
2.hand-full  

## how to build and run
### step1
https://github.com/Tencent/ncnn/releases

* Download ncnn-YYYYMMDD-android-vulkan.zip or build ncnn for android yourself
* Extract ncnn-YYYYMMDD-android-vulkan.zip into **app/src/main/jni** and change the **ncnn_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step2
https://github.com/nihui/opencv-mobile

* Download opencv-mobile-XYZ-android.zip
* Extract opencv-mobile-XYZ-android.zip into **app/src/main/jni** and change the **OpenCV_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step3
* Open this project with Android Studio, build it and enjoy!

## screenshot  
![](result.gif)  

## reference  
1.https://github.com/google/mediapipe  
2.https://github.com/nihui/ncnn-android-nanodet

