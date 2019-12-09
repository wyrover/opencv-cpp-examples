cmake_minimum_required(VERSION 3.0.0)

# cmake 工程名，不影响 vs 工程名
project(opencv-cpp-examples VERSION 0.1.0)

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")

if(CMAKE_CONFIGURATION_TYPES)
set(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE})
endif()

# 编译库输出路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# cmake 附加模块
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# 设置 cmake 搜索头文件路径
set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "${PROJECT_SOURCE_DIR}/include")

# 设置 cmake 搜索库路径
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "${PROJECT_SOURCE_DIR}/lib")

# set(CMAKE_GENERATOR_TOOLSET "v110_xp" CACHE STRING "CMAKE_GENERATOR_TOOLSET" FORCE)
# set(CMAKE_VS_PLATFORM_TOOLSET "v110_xp" CACHE STRING "CMAKE_VS_PLATFORM_TOOLSET" FORCE)

# set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "CMAKE_BUILD_TYPE" FORCE)
option(DEBUG_CONSOLE "是否启用调试控制台" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")


# 通用的 include 路径
#include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/123" "${CMAKE_CURRENT_SOURCE_DIR}/456" )

if(MSVC)
    foreach(flag
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
        endif()
    endforeach()

    foreach(flag
            CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
            CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
        if(${flag} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
        endif()
    endforeach()

    # 预定义宏
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)
endif()


add_subdirectory(src)




