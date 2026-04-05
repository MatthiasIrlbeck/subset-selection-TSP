cmake_minimum_required(VERSION 3.16)
project(aldous_tsp LANGUAGES CXX)

if(NOT UNIX)
  message(FATAL_ERROR "aldous_tsp currently requires a POSIX-like platform (fork/waitpid/kill/chdir).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
  endif()
endif()

option(ALDOUS_TSP_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(ALDOUS_TSP_ENABLE_NATIVE "Enable -march=native on supported compilers" OFF)

find_package(Threads REQUIRED)
include(CTest)

add_executable(aldous_tsp
    src/main.cpp
    src/tsp_solver.cpp
    src/oracle.cpp
    src/subset_seed.cpp
    src/subset_pool.cpp
    src/subset_smallp.cpp
    src/subset_highp.cpp
    src/subset_search.cpp
    src/run_cli.cpp
    src/run_cli_common.cpp
    src/run_cli_parse.cpp
    src/run_cli_report.cpp
    src/run_cli_self_test.cpp
    src/run_cli_orchestrate.cpp
    include/config.hpp
    include/problem.hpp
    include/tour.hpp
    include/oracle.hpp
    include/core.hpp
    include/tsp_solver.hpp
    include/subset_solver.hpp
    include/run_cli.hpp
    src/subset_solver_internal.hpp
    src/run_cli_internal.hpp
)

target_include_directories(aldous_tsp PRIVATE include)
target_link_libraries(aldous_tsp PRIVATE Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  if(ALDOUS_TSP_ENABLE_WARNINGS)
    target_compile_options(aldous_tsp PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  if(ALDOUS_TSP_ENABLE_NATIVE)
    target_compile_options(aldous_tsp PRIVATE -march=native)
    message(STATUS "ALDOUS_TSP_ENABLE_NATIVE=ON; enabling -march=native")
  endif()
endif()


if(BUILD_TESTING)
  add_test(NAME aldous_tsp_self_test COMMAND $<TARGET_FILE:aldous_tsp> --self-test)
  set_tests_properties(aldous_tsp_self_test PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    LABELS self-test
    TIMEOUT 180
  )
endif()
