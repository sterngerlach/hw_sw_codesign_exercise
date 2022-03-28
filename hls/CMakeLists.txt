
# CMakeLists.txt

include(CMakeParseArguments)

cmake_minimum_required(VERSION 3.16)

project(book_chapter_exercise CXX)

# Check that Vitis HLS is available (settings64.sh is sourced)
find_program(VITIS_HLS_SOURCED vitis_hls)

if (NOT VITIS_HLS_SOURCED)
  message(FATAL_ERROR "vitis_hls command is not available")
endif()

# Include directories
set(TOYNET_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src)

# Working directory
set(TOYNET_WORK_DIR ${PROJECT_SOURCE_DIR}/work)

if (NOT EXISTS ${TOYNET_WORK_DIR})
  message(FATAL_ERROR "Working directory does not exist: "
          ${TOYNET_WORK_DIR})
else()
  message(STATUS "Working directory: ${TOYNET_WORK_DIR}")
endif()

# Root directory for Vitis HLS 2020.2
set(VITIS_HLS_ROOT_DIR "/tools/Xilinx/Vitis_HLS/2020.2"
    CACHE STRING "Root directory for Vitis HLS 2020.2")
# Include directories for Vitis HLS 2020.2
set(VITIS_HLS_INCLUDE_DIRS ${VITIS_HLS_ROOT_DIR}/include)

if (NOT EXISTS ${VITIS_HLS_ROOT_DIR})
  message(FATAL_ERROR "Root directory for Vitis HLS 2020.2 does not exist: "
          ${VITIS_HLS_ROOT_DIR})
else()
  message(STATUS "Root directory for Vitis HLS 2020.2: "
          ${VITIS_HLS_ROOT_DIR})
endif()

if (NOT EXISTS ${VITIS_HLS_INCLUDE_DIRS})
  message(FATAL_ERROR "Include directory for Vitis HLS 2020.2 does not exist: "
          ${VITIS_HLS_INCLUDE_DIRS})
else()
  message(STATUS "Include directory for Vitis HLS 2020.2: "
          ${VITIS_HLS_INCLUDE_DIRS})
endif()

# Target board name (Xilinx ZCU104, Avnet Ultra96v2)
set(TARGET_BOARD "zcu104" CACHE STRING
    "Target device (zcu104, ultra96v2, pynqz2)")

# Target device name
if (${TARGET_BOARD} STREQUAL "zcu104")
  set(TARGET_DEVICE "xczu7ev-ffvc1156-2-e")
  message(STATUS "Target board: ${TARGET_BOARD}")
  message(STATUS "Target device: ${TARGET_DEVICE}")
elseif (${TARGET_BOARD} STREQUAL "ultra96v2")
  set(TARGET_DEVICE "xczu3eg-sbva484-1-e")
  message(STATUS "Target board: ${TARGET_BOARD}")
  message(STATUS "Target device: ${TARGET_DEVICE}")
elseif (${TARGET_BOARD} STREQUAL "pynqz2")
  set(TARGET_DEVICE "xc7z020-clg400-1")
  message(STATUS "Target board: ${TARGET_BOARD}")
  message(STATUS "Target device: ${TARGET_DEVICE}")
else()
  message(FATAL_ERROR "Unexpected target board: ${TARGET_BOARD}")
endif()

# Path to the Tcl scripts
set(VITIS_HLS_TCL_PATH ${PROJECT_SOURCE_DIR}/tcl/common.tcl)

if (NOT EXISTS ${VITIS_HLS_TCL_PATH})
  message(FATAL_ERROR "Tcl script does not exist: ${VITIS_HLS_TCL_PATH}")
else()
  message(STATUS "Tcl script found: ${VITIS_HLS_TCL_PATH}")
endif()

macro(prefix_options srcs option dsts)
  foreach(src ${srcs})
    list(APPEND ${dsts} "${option}${src}")
  endforeach()
endmacro()

function(hls_add_targets project_name top_function_name)
  cmake_parse_arguments(ARG "" "" "HLS_SRCS;TB_SRCS;CXXFLAGS" ${ARGN})

  set(include_dirs ${TOYNET_INCLUDE_DIRS})
  prefix_options("${include_dirs}" "-I" include_options)
  message(STATUS "Include directories: ${include_options}")

  set(library_dirs "")
  prefix_options("${library_dirs}" "-L" library_dir_options)
  prefix_options("${library_dirs}" "-Wl,-rpath," rpath_options)
  message(STATUS "Library directories: ${library_dir_options}")

  set(libraries "m" "pthread" "dl")
  prefix_options("${libraries}" "-l" library_options)
  message(STATUS "Library options: ${library_options}")

  set(hls_cxx_flags ${ARG_CXXFLAGS} ${include_options})
  string(REPLACE ";" " " hls_cxx_flags_str "${hls_cxx_flags}")

  set(hls_linker_flags ${library_dir_options}
      ${library_options} ${rpath_options})
  string(REPLACE ";" " " hls_linker_flags_str "${hls_linker_flags}")

  string(REPLACE ";" " " hls_srcs_str "${ARG_HLS_SRCS}")
  string(REPLACE ";" " " tb_srcs_str "${ARG_TB_SRCS}")
  message(STATUS "HLS source files: ${hls_srcs_str}")
  message(STATUS "HLS testbench files: ${tb_srcs_str}")

  list(LENGTH ARG_HLS_SRCS num_hls_srcs)
  list(LENGTH ARG_TB_SRCS num_tb_srcs)

  if (${num_tb_srcs} GREATER 0)
    message(STATUS "Appending a custom target: ${project_name}_csim")
    add_custom_target(${project_name}_csim
      COMMAND vitis_hls -f ${VITIS_HLS_TCL_PATH}
        "csim" ${project_name} ${top_function_name} ${TARGET_DEVICE}
        "cxx_flags=${hls_cxx_flags_str}" "${hls_srcs_str}"
        "linker_flags=${hls_linker_flags_str}" "${tb_srcs_str}"
      WORKING_DIRECTORY ${TOYNET_WORK_DIR})
  endif()

  if (${num_hls_srcs} GREATER 0)
    message(STATUS "Appending a custom target: ${project_name}_csynth")
    add_custom_target(${project_name}_csynth
      COMMAND vitis_hls -f ${VITIS_HLS_TCL_PATH}
        "csynth" ${project_name} ${top_function_name} ${TARGET_DEVICE}
        "cxx_flags=${hls_cxx_flags_str}" "${hls_srcs_str}"
      WORKING_DIRECTORY ${TOYNET_WORK_DIR})

    message(STATUS "Appending a custom target: ${project_name}_csynth_export")
    add_custom_target(${project_name}_csynth_export
      COMMAND vitis_hls -f ${VITIS_HLS_TCL_PATH}
        "csynth_export" ${project_name} ${top_function_name} ${TARGET_DEVICE}
        "cxx_flags=${hls_cxx_flags_str}" "${hls_srcs_str}"
      WORKING_DIRECTORY ${TOYNET_WORK_DIR})
  endif()
endfunction()

hls_add_targets(zcu104_toynet_naive InferenceNaive
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_naive.cpp
  TB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/tb/layer_test.cpp
  CXXFLAGS "-DBIT_WIDTH=32 -DINT_BIT_WIDTH=16")
hls_add_targets(zcu104_toynet_opt InferenceOpt
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt.cpp
  CXXFLAGS "-DBIT_WIDTH=32 -DINT_BIT_WIDTH=16")
hls_add_targets(zcu104_toynet_opt2 InferenceOpt2
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt2.cpp
  CXXFLAGS "-DBIT_WIDTH=32 -DINT_BIT_WIDTH=16")
hls_add_targets(zcu104_toynet_opt3 InferenceOpt3
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt3.cpp
  CXXFLAGS "-DBIT_WIDTH=32 -DINT_BIT_WIDTH=16")

hls_add_targets(zcu104_toynet_opt2_24 InferenceOpt2
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt2.cpp
  CXXFLAGS "-DBIT_WIDTH=24 -DINT_BIT_WIDTH=12")
hls_add_targets(zcu104_toynet_opt2_16 InferenceOpt2
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt2.cpp
  CXXFLAGS "-DBIT_WIDTH=16 -DINT_BIT_WIDTH=8")
hls_add_targets(zcu104_toynet_opt2_8 InferenceOpt2
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt2.cpp
  CXXFLAGS "-DBIT_WIDTH=8 -DINT_BIT_WIDTH=4")

hls_add_targets(zcu104_toynet_opt3_24 InferenceOpt3
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt3.cpp
  CXXFLAGS "-DBIT_WIDTH=24 -DINT_BIT_WIDTH=12")
hls_add_targets(zcu104_toynet_opt3_16 InferenceOpt3
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt3.cpp
  CXXFLAGS "-DBIT_WIDTH=16 -DINT_BIT_WIDTH=8")
hls_add_targets(zcu104_toynet_opt3_14 InferenceOpt3
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt3.cpp
  CXXFLAGS "-DBIT_WIDTH=14 -DINT_BIT_WIDTH=7")
hls_add_targets(zcu104_toynet_opt3_12 InferenceOpt3
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt3.cpp
  CXXFLAGS "-DBIT_WIDTH=12 -DINT_BIT_WIDTH=6")
hls_add_targets(zcu104_toynet_opt3_11 InferenceOpt3
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt3.cpp
  CXXFLAGS "-DBIT_WIDTH=11 -DINT_BIT_WIDTH=5")
hls_add_targets(zcu104_toynet_opt3_10 InferenceOpt3
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt3.cpp
  CXXFLAGS "-DBIT_WIDTH=10 -DINT_BIT_WIDTH=5")
  hls_add_targets(zcu104_toynet_opt3_9 InferenceOpt3
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt3.cpp
  CXXFLAGS "-DBIT_WIDTH=9 -DINT_BIT_WIDTH=4")
hls_add_targets(zcu104_toynet_opt3_8 InferenceOpt3
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_opt3.cpp
  CXXFLAGS "-DBIT_WIDTH=8 -DINT_BIT_WIDTH=4")

hls_add_targets(zcu104_empty InferenceEmpty
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_empty.cpp
  CXXFLAGS "-DBIT_WIDTH=32 -DINT_BIT_WIDTH=16")
