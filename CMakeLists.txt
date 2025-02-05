cmake_minimum_required(VERSION 3.10)
project(HD CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
    -D_GNU_SOURCE
    -Wall
    -O2
)

set(Torch_DIR "/home/jetson/.local/lib/python3.10/site-packages/torch/share/cmake/Torch")#change if needed

find_package(Torch REQUIRED)

include_directories(/usr/include/openncc)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(HD helmet_detection.cpp)

set(TARGET_LIB /usr/lib/openncc/libOpenNCC_native.so)
set(OTHER_LIBS rt dl usb-1.0 pthread)

target_link_libraries(HD
    PRIVATE
        ${TORCH_LIBRARIES}
        ${OPENCV_LIBRARIES}
        ${OTHER_LIBS}
        ${TARGET_LIB}
)
