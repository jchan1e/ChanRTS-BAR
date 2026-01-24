### Generic native Skirmish AI config
#

set(mySourceDirRel         "src") # Common values are "" or "src"
set(additionalSources      "")
set(additionalCompileFlags "-std=c++17 -Wno-deprecated-declarations -D_GLIBCXX_USE_CXX11_ABI=1 -DTORCH_DISABLE_ASSERTS")

# Find LibTorch
find_package(Torch QUIET)
if(NOT Torch_FOUND)
    # Try to find LibTorch in common locations
    set(CMAKE_PREFIX_PATH 
        "/usr/local/libtorch"
        "/usr/local/libtorch/share/cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/share/cmake"
        ${CMAKE_PREFIX_PATH}
    )
    find_package(Torch QUIET)
endif()

if(Torch_FOUND)
    message(STATUS "Found LibTorch at: ${TORCH_INSTALL_PREFIX}")
    set(additionalLibraries    ${Cpp_AIWRAPPER_TARGET} CUtils ${TORCH_LIBRARIES})
    # Set C++ standard required by PyTorch
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    
    # Add aggressive compatibility flags for ROCm LibTorch 2.7.1 + GCC 13
    set(additionalCompileFlags "${additionalCompileFlags} -D_GLIBCXX_USE_CXX11_ABI=1 -DTORCH_API_INCLUDE_EXTENSION_H -fpermissive -Wno-narrowing -DCHANRTS_ROCM_COMPAT")
    
    # Fix CUDA/ROCm linker issues if present
    if(TORCH_CUDA_VERSION OR DEFINED ENV{ROCM_PATH})
        message(STATUS "LibTorch compiled with CUDA/ROCm")
        # Override problematic linker flags
        set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${CMAKE_CXX_COMPILER})
        # Remove incompatible linker flags
        string(REPLACE "-Wl,--icf=all" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
        string(REPLACE "-fuse-ld=gold" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
        # Use standard ld for CUDA/ROCm linking
        set(CMAKE_LINKER "/usr/bin/ld")
    endif()
else()
    message(WARNING "LibTorch not found! ChanRTS will be built without ML support.")
    message(STATUS "To install LibTorch:")
    message(STATUS "  wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip")
    message(STATUS "  unzip libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip")
    message(STATUS "  sudo mv libtorch /usr/local/")
    set(additionalLibraries    ${Cpp_AIWRAPPER_TARGET} CUtils)
    add_definitions(-DCHANRTS_NO_LIBTORCH)
    message(STATUS "Building ChanRTS without LibTorch support due to compatibility issues")
endif()


if    (BUILD_Cpp_AIWRAPPER)
	include_directories(BEFORE ${Cpp_AIWRAPPER_INCLUDE_DIRS})
	configure_native_skirmish_ai(mySourceDirRel additionalSources additionalCompileFlags additionalLibraries)
else  (BUILD_Cpp_AIWRAPPER)
	message ("warning: ChanRTS AI will not be built! (missing Cpp Wrapper)")
endif (BUILD_Cpp_AIWRAPPER)
