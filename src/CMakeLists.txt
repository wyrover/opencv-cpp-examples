# 源码根路径
set(INCROOT ${PROJECT_SOURCE_DIR}/src)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src)



#add_definitions(-DWIN32 -D_WINDOWS -D_USRDLL)
#
IF(NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE "Release")
ENDIF(NOT CMAKE_BUILD_TYPE)

message(${CMAKE_BUILD_TYPE})

if( CMAKE_BUILD_TYPE MATCHES "Debug")	
	add_definitions(
		-DDEBUG
   )    
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
   add_definitions(        
       -DNDEBUG
   )      
   #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi /sdl /Oi /GL /EHa /Gy")
endif()
#
#if (DEBUG_CONSOLE)
#    add_definitions(
#        -DDEBUG_CONSOLE
#    )
#endif()


# 源码路径
#file(GLOB_RECURSE SRCS_G "${SRCROOT}/*.cpp")


#set(TMP_SRC 
#    "${SRCROOT}/lib_base.c"
#    "${SRCROOT}/lib_math.c"
#    "${SRCROOT}/lib_bit.c"
#    "${SRCROOT}/lib_string.c"
#    "${SRCROOT}/lib_table.c"
#    "${SRCROOT}/lib_io.c"
#    "${SRCROOT}/lib_os.c"
#    "${SRCROOT}/lib_package.c"
#    "${SRCROOT}/lib_debug.c"
#    "${SRCROOT}/lib_jit.c"
#    "${SRCROOT}/lib_ffi.c"    
#)
##source_group("游戏登录\\魔兽争霸对战平台" FILES ${TMP_SRC})
#list(APPEND SRC ${TMP_SRC})



#message(${SRC})







# 预编译头
#if(MSVC)	
#	set (PRECOMPILED_SOURCE ${SRC})
#	list(REMOVE_ITEM PRECOMPILED_SOURCE ${SRCROOT}/zhw/stdafx.cpp)
#	set (PRECOMPILED_BINARY ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pch)
#	set_source_files_properties (${PRECOMPILED_SOURCE} PROPERTIES COMPILE_FLAGS "/Yu\"stdafx.h\" /Fp\"${PRECOMPILED_BINARY}\"" OBJECT_DEPENDS "${PRECOMPILED_BINARY}")
#	set_source_files_properties (${SRCROOT}/zhw/stdafx.cpp PROPERTIES COMPILE_FLAGS "/Yc\"stdafx.h\" /Fp\"${PRECOMPILED_BINARY}\"" OBJECT_OUTPUTS "${PRECOMPILED_BINARY}")
#endif (MSVC)


#if(WIN32)    
#    add_definitions(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_STDIO_INLINE=__declspec\(dllexport\)__inline)    
#    
#endif()

# vs 工程名

# add_executable(minilua ${SRCROOT}/host/minilua.c)
# target_compile_definitions(minilua PRIVATE "-D_CRT_SECURE_NO_DEPRECATE" "-D_CRT_STDIO_INLINE=__declspec\(dllexport\)__inline")
# add_custom_command(TARGET minilua POST_BUILD COMMAND minilua ${SRCROOT}\\..\\dynasm\\dynasm.lua -LN -D WIN -D JIT -D FFI -o
#     ${SRCROOT}\\host\\buildvm_arch.h 
#     ${SRCROOT}\\vm_x86.dasc
# )


# add_executable(buildvm 
#     ${SRCROOT}/host/buildvm.c
#     ${SRCROOT}/host/buildvm_asm.c
#     ${SRCROOT}/host/buildvm_fold.c
#     ${SRCROOT}/host/buildvm_lib.c
#     ${SRCROOT}/host/buildvm_peobj.c
# )
# target_compile_definitions(buildvm PRIVATE "-D_CRT_SECURE_NO_DEPRECATE" "-D_CRT_STDIO_INLINE=__declspec\(dllexport\)__inline")
# target_compile_options(buildvm PRIVATE "/arch:SSE2")
# target_include_directories(buildvm PRIVATE 
#     ${INCROOT}
#     ${INCROOT}/../dynasm
# )

# add_custom_command(TARGET buildvm POST_BUILD COMMAND buildvm -m peobj -o ${SRCROOT}/lj_vm.obj)
# add_custom_command(TARGET buildvm POST_BUILD COMMAND buildvm -m bcdef -o 
#     ${SRCROOT}/lj_bcdef.h 
#     ${SRCROOT}/lib_base.c 
#     ${SRCROOT}/lib_math.c 
#     ${SRCROOT}/lib_bit.c 
#     ${SRCROOT}/lib_string.c 
#     ${SRCROOT}/lib_table.c 
#     ${SRCROOT}/lib_io.c 
#     ${SRCROOT}/lib_os.c 
#     ${SRCROOT}/lib_package.c 
#     ${SRCROOT}/lib_debug.c 
#     ${SRCROOT}/lib_jit.c 
#     ${SRCROOT}/lib_ffi.c
# )

# add_custom_command(TARGET buildvm POST_BUILD COMMAND buildvm -m ffdef -o 
#     ${SRCROOT}/lj_ffdef.h 
#     ${SRCROOT}/lib_base.c 
#     ${SRCROOT}/lib_math.c 
#     ${SRCROOT}/lib_bit.c 
#     ${SRCROOT}/lib_string.c 
#     ${SRCROOT}/lib_table.c 
#     ${SRCROOT}/lib_io.c 
#     ${SRCROOT}/lib_os.c 
#     ${SRCROOT}/lib_package.c 
#     ${SRCROOT}/lib_debug.c 
#     ${SRCROOT}/lib_jit.c 
#     ${SRCROOT}/lib_ffi.c
# )

# add_custom_command(TARGET buildvm POST_BUILD COMMAND buildvm -m libdef -o
#     ${SRCROOT}/lj_libdef.h 
#     ${SRCROOT}/lib_base.c 
#     ${SRCROOT}/lib_math.c 
#     ${SRCROOT}/lib_bit.c 
#     ${SRCROOT}/lib_string.c 
#     ${SRCROOT}/lib_table.c 
#     ${SRCROOT}/lib_io.c 
#     ${SRCROOT}/lib_os.c 
#     ${SRCROOT}/lib_package.c 
#     ${SRCROOT}/lib_debug.c 
#     ${SRCROOT}/lib_jit.c 
#     ${SRCROOT}/lib_ffi.c
# )

# add_custom_command(TARGET buildvm POST_BUILD COMMAND buildvm -m recdef -o 
#     ${SRCROOT}/lj_recdef.h 
#     ${SRCROOT}/lib_base.c 
#     ${SRCROOT}/lib_math.c 
#     ${SRCROOT}/lib_bit.c 
#     ${SRCROOT}/lib_string.c 
#     ${SRCROOT}/lib_table.c 
#     ${SRCROOT}/lib_io.c 
#     ${SRCROOT}/lib_os.c 
#     ${SRCROOT}/lib_package.c 
#     ${SRCROOT}/lib_debug.c 
#     ${SRCROOT}/lib_jit.c 
#     ${SRCROOT}/lib_ffi.c
# )

# add_custom_command(TARGET buildvm POST_BUILD COMMAND buildvm -m vmdef -o 
#     ${SRCROOT}/jit/vmdef.lua 
#     ${SRCROOT}/lib_base.c 
#     ${SRCROOT}/lib_math.c 
#     ${SRCROOT}/lib_bit.c 
#     ${SRCROOT}/lib_string.c 
#     ${SRCROOT}/lib_table.c 
#     ${SRCROOT}/lib_io.c 
#     ${SRCROOT}/lib_os.c 
#     ${SRCROOT}/lib_package.c 
#     ${SRCROOT}/lib_debug.c 
#     ${SRCROOT}/lib_jit.c 
#     ${SRCROOT}/lib_ffi.c
    
# )

# add_custom_command(TARGET buildvm POST_BUILD COMMAND buildvm -m folddef -o 
#     ${SRCROOT}/lj_folddef.h 
#     ${SRCROOT}/lj_opt_fold.c
# )




# add_library(lua51 SHARED 
#     ${SRCROOT}/lj_alloc.c
#     ${SRCROOT}/lj_api.c
#     ${SRCROOT}/lj_asm.c
#     ${SRCROOT}/lj_bc.c
#     ${SRCROOT}/lj_bcread.c
#     ${SRCROOT}/lj_bcwrite.c
#     ${SRCROOT}/lj_buf.c
#     ${SRCROOT}/lj_carith.c
#     ${SRCROOT}/lj_ccall.c
#     ${SRCROOT}/lj_ccallback.c
#     ${SRCROOT}/lj_cconv.c
#     ${SRCROOT}/lj_cdata.c
#     ${SRCROOT}/lj_char.c
#     ${SRCROOT}/lj_clib.c
#     ${SRCROOT}/lj_cparse.c
#     ${SRCROOT}/lj_crecord.c
#     ${SRCROOT}/lj_ctype.c
#     ${SRCROOT}/lj_debug.c
#     ${SRCROOT}/lj_dispatch.c
#     ${SRCROOT}/lj_err.c
#     ${SRCROOT}/lj_ffrecord.c
#     ${SRCROOT}/lj_func.c
#     ${SRCROOT}/lj_gc.c
#     ${SRCROOT}/lj_gdbjit.c
#     ${SRCROOT}/lj_ir.c
#     ${SRCROOT}/lj_lex.c
#     ${SRCROOT}/lj_lib.c
#     ${SRCROOT}/lj_load.c
#     ${SRCROOT}/lj_mcode.c
#     ${SRCROOT}/lj_meta.c
#     ${SRCROOT}/lj_obj.c
#     ${SRCROOT}/lj_opt_dce.c
#     ${SRCROOT}/lj_opt_fold.c
#     ${SRCROOT}/lj_opt_loop.c
#     ${SRCROOT}/lj_opt_mem.c
#     ${SRCROOT}/lj_opt_narrow.c
#     ${SRCROOT}/lj_opt_sink.c
#     ${SRCROOT}/lj_opt_split.c
#     ${SRCROOT}/lj_parse.c
#     ${SRCROOT}/lj_profile.c
#     ${SRCROOT}/lj_record.c
#     ${SRCROOT}/lj_snap.c
#     ${SRCROOT}/lj_state.c
#     ${SRCROOT}/lj_str.c
#     ${SRCROOT}/lj_strfmt.c
#     ${SRCROOT}/lj_strfmt_num.c
#     ${SRCROOT}/lj_strscan.c
#     ${SRCROOT}/lj_tab.c
#     ${SRCROOT}/lj_trace.c
#     ${SRCROOT}/lj_udata.c
#     ${SRCROOT}/lj_vmevent.c
#     ${SRCROOT}/lj_vmmath.c
#     ${SRCROOT}/lib_aux.c
#     ${SRCROOT}/lib_base.c    
#     ${SRCROOT}/lib_bit.c
#     ${SRCROOT}/lib_debug.c
#     ${SRCROOT}/lib_ffi.c
#     ${SRCROOT}/lib_init.c
#     ${SRCROOT}/lib_io.c
#     ${SRCROOT}/lib_jit.c
#     ${SRCROOT}/lib_math.c
#     ${SRCROOT}/lib_os.c
#     ${SRCROOT}/lib_package.c
#     ${SRCROOT}/lib_string.c
#     ${SRCROOT}/lib_table.c
#     ${SRCROOT}/lj_vm.obj
# )


# target_compile_definitions(lua51 PRIVATE "_CRT_SECURE_NO_DEPRECATE" "_CRT_STDIO_INLINE=__declspec\(dllexport\)__inline" "LUA_BUILD_AS_DLL")
# target_compile_options(lua51 PRIVATE "/arch:SSE2")

# #cl /nologo /c /O2 /W3 /D_CRT_SECURE_NO_DEPRECATE /D_CRT_STDIO_INLINE=__declspec(dllexport)__inline /arch:SSE2 /MD /DLUA_BUILD_AS_DLL lj_*.c lib_*.c

# add_executable(luajit ${SRCROOT}/luajit.c)
# target_compile_definitions(luajit PRIVATE "_CRT_SECURE_NO_DEPRECATE" "_CRT_STDIO_INLINE=__declspec\(dllexport\)__inline")
# target_compile_options(luajit PRIVATE "/arch:SSE2")
# target_link_libraries(luajit lua51)





# 特定 vs 工程头文件路径

# if( CMAKE_BUILD_TYPE MATCHES "Debug")	

#    target_link_directories(zhw PRIVATE
#         D:/downloads/4.1.2/opencv-4.1.2/build/install/x86/vc15/staticlib
#    )

# #    set_target_properties(zhw PROPERTIES LINK_FLAGS "/NODEFAULTLIB:\"LIBCMT\" /SAFESEH:NO")  

# #    string(REPLACE "/" "\\" bin_dir ${CMAKE_BINARY_DIR})        
# #    string(REPLACE "/" "\\" cmake_dir ${PROJECT_SOURCE_DIR})    
# #    add_custom_command(TARGET zhw POST_BUILD COMMAND xcopy /Y /E /R /F "${bin_dir}\\bin\\Debug\\zhw.dll" ${cmake_dir}\\build\\bin\\)
# elseif(CMAKE_BUILD_TYPE MATCHES "Release")

# #    target_link_directories(zhw PRIVATE
# #        ${SRCROOT}/../../../core/lib/win32/release
# #        ${SRCROOT}/../../../ui/lib/win32/release
# #        ${SRCROOT}/Import/lib
# #        ${SRCROOT}/../../yangyong/lib/win32    
# #        ${SRCROOT}/Import/bin/Common/TXSSO
# #        ${SRCROOT}/Import/bin/driver/Libs
# #        ${SRCROOT}/../3rdparty/boost/lib/win32
# #        ${SRCROOT}/../3rdparty/EasyHook/lib
# #    )    
   
# #    set_target_properties(zhw PROPERTIES LINK_FLAGS "/NODEFAULTLIB:\"LIBCMTD\" /SAFESEH:NO")  

# #    string(REPLACE "/" "\\" bin_dir ${CMAKE_BINARY_DIR})
# #    string(REPLACE "/" "\\" cmake_dir ${PROJECT_SOURCE_DIR})  
# #    add_custom_command(TARGET zhw POST_BUILD COMMAND xcopy /Y /E /R /F "${bin_dir}\\bin\\Release\\zhw.dll" ${cmake_dir}\\build\\bin\\)
# endif()








# 特定 vs 工程头文件路径



set_property(GLOBAL PROPERTY USE_FOLDERS ON)

function(snippet number)
    add_executable(snippet${number} ${PROJECT_SOURCE_DIR}/src/main.cpp)
    target_compile_definitions(snippet${number} PRIVATE SNIPPET${number})
    #target_link_libraries(snippet${number} pthread)

    if( CMAKE_BUILD_TYPE MATCHES "Debug")	
        target_include_directories(snippet${number} PRIVATE 
            D:/downloads/4.1.2/opencv-4.1.2/build/install/include   
        )    


        target_link_libraries(
            snippet${number} 
            opencv_world412d.lib
            opencv_img_hash412d.lib
#            ade.lib
#            IlmImfd.lib
#            ippicvmt.lib
#            ippiwd.lib
#            ittnotifyd.lib
#            libjasperd.lib
#            libjpeg-turbod.lib
#            libpngd.lib
#            libprotobufd.lib
#            libtiffd.lib
#            libwebpd.lib
#            quircd.lib
#            zlibd.lib
#            mf
#            mfplat
#            mfreadwrite
#            mfuuid
#            shlwapi
#            wmcodecdspuuid
        )


        target_link_directories(snippet${number} PRIVATE
            D:/downloads/4.1.2/opencv-4.1.2/build/install/x86/vc15/debug/lib
        )

    elseif(CMAKE_BUILD_TYPE MATCHES "Release")
        target_include_directories(snippet${number} PRIVATE 
            D:/downloads/4.1.2/opencv-4.1.2/build/install/include   
        )    


        target_link_libraries(
            snippet${number} 
            opencv_world412.lib
            opencv_img_hash412.lib
            ade.lib
            IlmImf.lib
            ippicvmt.lib
            ippiw.lib
            ittnotify.lib
            libjasper.lib
            libjpeg-turbo.lib
            libpng.lib
            libprotobuf.lib
            libtiff.lib
            libwebp.lib
            quirc.lib
            zlib.lib
            mf
            mfplat
            mfreadwrite
            mfuuid
            shlwapi
            wmcodecdspuuid
        )


        target_link_directories(snippet${number} PRIVATE
            D:/downloads/4.1.2/opencv-4.1.2/build/install/x86/vc15/release/staticlib
        )
    endif()


    

    set_property(TARGET snippet${number} PROPERTY FOLDER "opencv-cpp-examples")

endfunction(snippet)

function(get_num num result_var)
    set(result "error")
    string(LENGTH ${num} length)
    if (${length} STREQUAL "1")                
        set(result "00${num}")
    elseif (${length} STREQUAL "2")
        set(result "0${num}")        
    endif()   

    set(${result_var} "${result}" PARENT_SCOPE)
endfunction()


# 循环生成工程
foreach(i RANGE 1 70)
    get_num(${i} number)
    snippet(${number})    
endforeach(i)




# set_property(TARGET buildvm PROPERTY FOLDER "LuaJIT")
# set_property(TARGET lua51 PROPERTY FOLDER "LuaJIT")
# set_property(TARGET luajit PROPERTY FOLDER "LuaJIT")


#add_library(lua51 SHARED ${SRC})

