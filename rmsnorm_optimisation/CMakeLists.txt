cmake_minimum_required(VERSION 3.16)
project(rms_kernel_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# OV_ROOT:        path to an OpenVINO repo root that has already been built.
# KERNEL_SRC_DIR: path to directory containing the rms_kernel.cpp/.hpp
#                 variant to compile from source.
#
# Example:
#   cmake -B build -DOV_ROOT=/openvino -DKERNEL_SRC_DIR=/bench/variants/baseline
# ---------------------------------------------------------------------------
if(NOT DEFINED OV_ROOT)
    message(FATAL_ERROR "OV_ROOT must be set to the OpenVINO repo root (e.g. -DOV_ROOT=...)")
endif()
if(NOT DEFINED KERNEL_SRC_DIR)
    message(FATAL_ERROR "KERNEL_SRC_DIR must be set to the kernel variant directory")
endif()

get_filename_component(OV_ROOT "${OV_ROOT}" ABSOLUTE)
get_filename_component(KERNEL_SRC_DIR "${KERNEL_SRC_DIR}" ABSOLUTE)
message(STATUS "OV_ROOT        = ${OV_ROOT}")
message(STATUS "KERNEL_SRC_DIR = ${KERNEL_SRC_DIR}")

# ---------------------------------------------------------------------------
# Derive paths
# ---------------------------------------------------------------------------
set(CPU_PLUGIN_SRC  "${OV_ROOT}/src/plugins/intel_cpu/src")
set(CPU_PLUGIN_3P   "${OV_ROOT}/src/plugins/intel_cpu/thirdparty")
set(OV_BUILD        "${OV_ROOT}/build")
set(OV_BIN          "${OV_ROOT}/bin/intel64/Release")
set(OV_TBB          "${OV_ROOT}/temp/Linux_x86_64/tbb")

# ---------------------------------------------------------------------------
# Minimal set of plugin .o files needed by jit_rms_kernel.
# rms_kernel.cpp.o is NOT included - we compile it from KERNEL_SRC_DIR.
# ---------------------------------------------------------------------------
set(OBJ_DIR "${OV_BUILD}/src/plugins/intel_cpu/CMakeFiles/openvino_intel_cpu_plugin.dir")
set(PLUGIN_OBJECTS
    "${OBJ_DIR}/src/nodes/kernels/x64/jit_kernel_base.cpp.o"
    "${OBJ_DIR}/src/emitters/plugin/x64/jit_emitter.cpp.o"
    "${OBJ_DIR}/src/emitters/plugin/x64/jit_load_store_emitters.cpp.o"
    "${OBJ_DIR}/src/emitters/utils.cpp.o"
)
foreach(_obj IN LISTS PLUGIN_OBJECTS)
    if(NOT EXISTS "${_obj}")
        message(FATAL_ERROR "Required object file not found: ${_obj}\nIs the OV build complete?")
    endif()
endforeach()

if(NOT EXISTS "${KERNEL_SRC_DIR}/rms_kernel.cpp")
    message(FATAL_ERROR "Kernel source not found: ${KERNEL_SRC_DIR}/rms_kernel.cpp")
endif()
message(STATUS "Using 4 plugin .o files + kernel from ${KERNEL_SRC_DIR}")

# ---------------------------------------------------------------------------
# Create include override so the benchmark code's
#   #include "nodes/kernels/x64/rms_kernel.hpp"
# resolves to the variant's header (same layout as the compiled kernel).
# ---------------------------------------------------------------------------
set(INCLUDE_OVERRIDE "${CMAKE_BINARY_DIR}/include_override")
file(MAKE_DIRECTORY "${INCLUDE_OVERRIDE}/nodes/kernels/x64")
configure_file(
    "${KERNEL_SRC_DIR}/rms_kernel.hpp"
    "${INCLUDE_OVERRIDE}/nodes/kernels/x64/rms_kernel.hpp"
    COPYONLY
)

# ---------------------------------------------------------------------------
# Shared include directories
# ---------------------------------------------------------------------------
set(BENCH_INCLUDE_DIRS
    "${INCLUDE_OVERRIDE}"
    "${KERNEL_SRC_DIR}"
    "${CPU_PLUGIN_SRC}"
    "${CPU_PLUGIN_SRC}/nodes/kernels/x64"
    "${OV_ROOT}/src/core/include"
    "${OV_ROOT}/src/core/dev_api"
    "${OV_ROOT}/src/core/reference/include"
    "${OV_ROOT}/src/core/shape_inference/include"
    "${OV_ROOT}/src/core/xml_util/include"
    "${OV_ROOT}/src/inference/include"
    "${OV_ROOT}/src/inference/dev_api"
    "${OV_ROOT}/src/common/transformations/include"
    "${OV_ROOT}/src/common/low_precision_transformations/include"
    "${OV_ROOT}/src/common/snippets/include"
    "${OV_ROOT}/src/common/conditional_compilation/include"
    "${OV_ROOT}/src/common/itt/include"
    "${OV_ROOT}/src/common/util/include"
    "${OV_ROOT}/src/common/shutdown/include"
    "${OV_ROOT}/src/frontends/common/include"
    "${OV_ROOT}/src/frontends/onnx/frontend/include"
    "${OV_ROOT}/src/frontends/paddle/include"
    "${OV_ROOT}/src/frontends/pytorch/include"
    "${OV_ROOT}/src/frontends/tensorflow/include"
    "${OV_ROOT}/src/frontends/tensorflow_lite/include"
    "${OV_ROOT}/thirdparty/pugixml/src"
    "${OV_ROOT}/thirdparty/ittapi/ittapi/src/ittnotify"
    "${OV_ROOT}/thirdparty/ittapi/ittapi/include"
)

set(BENCH_SYSTEM_INCLUDE_DIRS
    "${CPU_PLUGIN_3P}/onednn/src"
    "${CPU_PLUGIN_3P}/onednn/include"
    "${CPU_PLUGIN_3P}/onednn/src/../include"
    "${CPU_PLUGIN_3P}/onednn/third_party"
    "${OV_BUILD}/src/plugins/intel_cpu/thirdparty/onednn/include"
    "${CPU_PLUGIN_3P}/mlas/inc"
    "${CPU_PLUGIN_3P}/mlas/lib"
    "${OV_TBB}/include"
    "${OV_ROOT}/thirdparty/xbyak"
)

set(BENCH_COMPILE_DEFS XBYAK64 OV_CPU_WITH_DNNL)

set(BENCH_LINK_LIBS
    ${PLUGIN_OBJECTS}
    "${OV_BIN}/libopenvino_onednn_cpu.a"
    "${OV_BIN}/libopenvino.so"
    "${OV_TBB}/lib/libtbb.so.12"
    dl
    pthread
)

set(BENCH_RPATH "${OV_BIN};${OV_TBB}/lib")

# ---------------------------------------------------------------------------
# Benchmark target
# ---------------------------------------------------------------------------
add_executable(rms_kernel_bench
    rms_kernel_bench.cpp
    "${KERNEL_SRC_DIR}/rms_kernel.cpp"
)

target_include_directories(rms_kernel_bench PRIVATE ${BENCH_INCLUDE_DIRS})
target_include_directories(rms_kernel_bench SYSTEM PRIVATE ${BENCH_SYSTEM_INCLUDE_DIRS})
target_compile_definitions(rms_kernel_bench PRIVATE ${BENCH_COMPILE_DEFS})
target_link_libraries(rms_kernel_bench PRIVATE ${BENCH_LINK_LIBS})
set_target_properties(rms_kernel_bench PROPERTIES BUILD_RPATH "${BENCH_RPATH}")
