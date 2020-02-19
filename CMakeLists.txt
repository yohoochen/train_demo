cmake_minimum_required(VERSION 3.0)
project(ctdet_trt)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# pose lib
set(CMAKE_PREFIX_PATH /home/nvidia/demo/libtorch1.1)
find_package(Torch REQUIRED)

add_definitions(-std=c++11)
#set(QMAKE_CXXFLAGS "-std=c++11")
SET( CUDA_HOME /usr/local/cuda )
SET( TensorRT_HOME /usr/include/aarch64-linux-gnu )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast" )

#include_directories(./include/ ${CUDA_HOME}/include/ ${TensorRT_HOME}/include/ )
include_directories(./include/ ${CUDA_HOME}/include/ )
link_directories(/usr/include/aarch64-linux-gnu ${CUDA_HOME}/lib64 )
link_directories(/home/nvidia/demo/libtorch1.1)

# 设置要链接的库文件
SET( LIB_PROJECT glog boost_system protobuf boost_thread gflags libnvonnxparser.so  libnvinfer_plugin.so  libnvparsers.so libnvonnxparser_runtime.so libnvinfer.so cudart)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

file(GLOB CPP_SRC
        src/Object_detect.cpp
        src/ObjProcess.cpp)
file(GLOB CU_SRC src/ObjectLayer.cu)

cuda_add_executable(buildEngine example/buildEngine.cpp ${CPP_SRC} ${CU_SRC})
target_link_libraries(buildEngine ${OpenCV_LIBS} ${LIB_PROJECT})

#cuda_add_executable(runDet example/runDet.cpp ${CPP_SRC} ${CU_SRC})
#target_link_libraries(runDet ${OpenCV_LIBS} ${LIB_PROJECT})

file(GLOB_RECURSE SOURCES
        include/human_pose_estimator.h
        include/peak.h
        src/peak.cpp
        src/human_pose_estimator.cpp)

cuda_add_executable(pose_test example/main.cpp ${SOURCES} ${CPP_SRC} ${CU_SRC})
target_link_libraries(pose_test ${TORCH_LIBRARIES} thor ${OpenCV_LIBS} ${LIB_PROJECT})
set_property(TARGET pose_test PROPERTY CXX_STANDARD 11)
