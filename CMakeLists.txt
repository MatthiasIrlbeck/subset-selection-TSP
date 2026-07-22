cmake_minimum_required(VERSION 3.20)

project(aldous_tsp VERSION 0.10.0 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

option(ALDOUS_TSP_BUILD_CLI "Build the aldous_tsp command-line executable" ON)
option(ALDOUS_TSP_BUILD_TESTS "Build C++ tests" ON)
option(ALDOUS_TSP_BUILD_EXAMPLES "Build library usage examples" OFF)
option(ALDOUS_TSP_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(ALDOUS_TSP_ENABLE_WERROR "Treat compiler warnings as errors" OFF)
option(ALDOUS_TSP_ENABLE_NATIVE "Enable -march=native on supported compilers" OFF)
option(ALDOUS_TSP_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ALDOUS_TSP_ENABLE_TSAN "Enable ThreadSanitizer (GNU/Clang on supported platforms)" OFF)
option(ALDOUS_TSP_ENABLE_CLANG_TIDY "Run clang-tidy when available" OFF)
option(ALDOUS_TSP_ENABLE_PYTHON_TESTS "Enable optional Python-driven benchmark/parity CTest smoke tests" OFF)
option(ALDOUS_TSP_LOW_MEMORY_BUILD "Compile dense translation units with low optimization to reduce compiler memory" OFF)
option(ALDOUS_TSP_BUILD_FUZZERS "Build dedicated libFuzzer targets (Clang only)" OFF)
option(ALDOUS_TSP_FUZZ_ENABLE_ASAN "Combine AddressSanitizer with libFuzzer (disable only for incompatible local runtimes)" ON)

if(ALDOUS_TSP_ENABLE_SANITIZERS AND ALDOUS_TSP_ENABLE_TSAN)
  message(FATAL_ERROR "Address/Undefined sanitizers and ThreadSanitizer require separate builds")
endif()
if(ALDOUS_TSP_ENABLE_TSAN AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(FATAL_ERROR "ALDOUS_TSP_ENABLE_TSAN requires GCC or Clang")
endif()
if(ALDOUS_TSP_ENABLE_TSAN AND ALDOUS_TSP_BUILD_FUZZERS)
  message(FATAL_ERROR "ThreadSanitizer and libFuzzer targets require separate builds")
endif()

if(ALDOUS_TSP_LOW_MEMORY_BUILD)
  set(ALDOUS_TSP_OPTIMIZATION_PROFILE "low-memory")
else()
  set(ALDOUS_TSP_OPTIMIZATION_PROFILE "release-optimized")
endif()

if(ALDOUS_TSP_LOW_MEMORY_BUILD)
  set(ALDOUS_TSP_EFFECTIVE_OPTIMIZATION_LEVEL "O0")
elseif(CMAKE_CONFIGURATION_TYPES)
  set(ALDOUS_TSP_EFFECTIVE_OPTIMIZATION_LEVEL "config-dependent")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(ALDOUS_TSP_EFFECTIVE_OPTIMIZATION_LEVEL "O2")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(ALDOUS_TSP_EFFECTIVE_OPTIMIZATION_LEVEL "O2")
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
  set(ALDOUS_TSP_EFFECTIVE_OPTIMIZATION_LEVEL "Os")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ALDOUS_TSP_EFFECTIVE_OPTIMIZATION_LEVEL "compiler-default")
else()
  set(ALDOUS_TSP_EFFECTIVE_OPTIMIZATION_LEVEL "unknown")
endif()

find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter QUIET)

if(Python3_Interpreter_FOUND)
  add_custom_target(aldous_tsp_generate_options
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_options.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Regenerating option-derived C++/schema/documentation surfaces"
    VERBATIM)
  add_custom_target(aldous_tsp_check_generated_options
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_options.py --check
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking generated option surfaces are up to date"
    VERBATIM)
endif()

# A native posix_spawn working-directory action avoids even the short, safe
# shell trampoline on platforms that provide the non-portable extension. Keep a
# portable argv-only shell fallback for other POSIX implementations.
if(WIN32)
  set(ALDOUS_TSP_HAVE_POSIX_SPAWN_CHDIR_NP OFF)
else()
  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(
    posix_spawn_file_actions_addchdir_np
    "spawn.h"
    ALDOUS_TSP_HAVE_POSIX_SPAWN_CHDIR_NP)
endif()

if(ALDOUS_TSP_ENABLE_PYTHON_TESTS)
  find_program(Python3_EXECUTABLE NAMES python3 python REQUIRED)
endif()

execute_process(
  COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse --verify HEAD
  OUTPUT_VARIABLE ALDOUS_TSP_GIT_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT ALDOUS_TSP_GIT_COMMIT)
  set(ALDOUS_TSP_GIT_COMMIT "unknown")
endif()

if(CMAKE_CONFIGURATION_TYPES)
  set(ALDOUS_TSP_CONFIGURED_BUILD_TYPE "multi-config")
else()
  set(ALDOUS_TSP_CONFIGURED_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
endif()
set(ALDOUS_TSP_CMAKE_GENERATOR "${CMAKE_GENERATOR}")
set(ALDOUS_TSP_CMAKE_VERSION "${CMAKE_VERSION}")
string(TOUPPER "${CMAKE_BUILD_TYPE}" ALDOUS_TSP_BUILD_TYPE_UPPER)
set(ALDOUS_TSP_CXX_FLAGS_CONFIGURED "${CMAKE_CXX_FLAGS}")
set(ALDOUS_TSP_CXX_FLAGS_EFFECTIVE_CONFIGURED "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${ALDOUS_TSP_BUILD_TYPE_UPPER}}")
set(ALDOUS_TSP_CORE_TARGET_COMPILE_OPTIONS_CONFIGURED "")
set(ALDOUS_TSP_CLI_TARGET_COMPILE_OPTIONS_CONFIGURED "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  if(ALDOUS_TSP_ENABLE_WARNINGS)
    string(APPEND ALDOUS_TSP_CORE_TARGET_COMPILE_OPTIONS_CONFIGURED " -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wold-style-cast")
    string(APPEND ALDOUS_TSP_CLI_TARGET_COMPILE_OPTIONS_CONFIGURED " -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wold-style-cast")
  endif()
  if(ALDOUS_TSP_ENABLE_WERROR)
    string(APPEND ALDOUS_TSP_CORE_TARGET_COMPILE_OPTIONS_CONFIGURED " -Werror")
    string(APPEND ALDOUS_TSP_CLI_TARGET_COMPILE_OPTIONS_CONFIGURED " -Werror")
  endif()
  if(ALDOUS_TSP_ENABLE_NATIVE)
    string(APPEND ALDOUS_TSP_CORE_TARGET_COMPILE_OPTIONS_CONFIGURED " -march=native")
  endif()
  string(APPEND ALDOUS_TSP_CORE_TARGET_COMPILE_OPTIONS_CONFIGURED " Release:-O2")
  if(ALDOUS_TSP_ENABLE_SANITIZERS)
    string(APPEND ALDOUS_TSP_CORE_TARGET_COMPILE_OPTIONS_CONFIGURED " -fsanitize=address,undefined -fno-omit-frame-pointer")
  elseif(ALDOUS_TSP_ENABLE_TSAN)
    string(APPEND ALDOUS_TSP_CORE_TARGET_COMPILE_OPTIONS_CONFIGURED " -fsanitize=thread -fno-omit-frame-pointer")
  endif()
  if(ALDOUS_TSP_LOW_MEMORY_BUILD)
    string(APPEND ALDOUS_TSP_CORE_TARGET_COMPILE_OPTIONS_CONFIGURED " low-memory-source:-O0 -g0")
    string(APPEND ALDOUS_TSP_CLI_TARGET_COMPILE_OPTIONS_CONFIGURED " low-memory-target:-O0 -g0")
  endif()
elseif(MSVC)
  if(ALDOUS_TSP_ENABLE_WARNINGS)
    string(APPEND ALDOUS_TSP_CORE_TARGET_COMPILE_OPTIONS_CONFIGURED " /W4")
    string(APPEND ALDOUS_TSP_CLI_TARGET_COMPILE_OPTIONS_CONFIGURED " /W4")
  endif()
  if(ALDOUS_TSP_ENABLE_WERROR)
    string(APPEND ALDOUS_TSP_CORE_TARGET_COMPILE_OPTIONS_CONFIGURED " /WX")
    string(APPEND ALDOUS_TSP_CLI_TARGET_COMPILE_OPTIONS_CONFIGURED " /WX")
  endif()
endif()
string(STRIP "${ALDOUS_TSP_CORE_TARGET_COMPILE_OPTIONS_CONFIGURED}" ALDOUS_TSP_CORE_TARGET_COMPILE_OPTIONS_CONFIGURED)
string(STRIP "${ALDOUS_TSP_CLI_TARGET_COMPILE_OPTIONS_CONFIGURED}" ALDOUS_TSP_CLI_TARGET_COMPILE_OPTIONS_CONFIGURED)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/aldous_tsp/version.hpp
  @ONLY
)

add_library(aldous_tsp_core
  src/geometry.cpp
  src/generated_options_core.cpp
  src/memory.cpp
  src/validation.cpp
  src/exact_subset.cpp
  src/instance.cpp
  src/tour.cpp
  src/solver_moves.cpp
  src/solver_spatial.cpp
  src/solver_construction.cpp
  src/solver_local_search.cpp
  src/solver_exchange.cpp
  src/solver_lns.cpp
  src/solver_ejection_chain.cpp
  src/solver_pair_exchange.cpp
  src/solver_path_relink.cpp
  src/solver_seeds.cpp
  src/solver_subset.cpp
  src/solver_tsp.cpp
  src/oracle.cpp
  src/lower_bound.cpp
  src/results.cpp
  src/results_io.cpp
  src/solver_facade.cpp
  src/experiment.cpp
)
add_library(aldous_tsp::core ALIAS aldous_tsp_core)
set_target_properties(aldous_tsp_core PROPERTIES EXPORT_NAME core)

target_compile_features(aldous_tsp_core PUBLIC cxx_std_17)
if(MSVC)
  # MSVC reports __cplusplus as 199711L unless told otherwise, which breaks the
  # build-metadata schema check (cplusplus >= 201703). /Zc:__cplusplus makes
  # the macro truthful. (Found by the first MSVC production run, 2026-07-08.)
  target_compile_options(aldous_tsp_core PUBLIC /Zc:__cplusplus)
endif()
target_include_directories(aldous_tsp_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(aldous_tsp_core PUBLIC Threads::Threads)
# Dense solver files can be compiled at low optimization in constrained builders,
# but normal Release builds keep production optimization enabled.
if(ALDOUS_TSP_LOW_MEMORY_BUILD AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  set_source_files_properties(
    src/solver_moves.cpp
  src/solver_spatial.cpp
    src/solver_construction.cpp
    src/solver_local_search.cpp
    src/solver_exchange.cpp
  src/solver_lns.cpp
  src/solver_ejection_chain.cpp
  src/solver_pair_exchange.cpp
  src/solver_path_relink.cpp
    src/solver_seeds.cpp
    src/solver_subset.cpp
    src/solver_tsp.cpp
    src/generated_options_cli.cpp
    src/cli_parse.cpp
    src/cli_values.cpp
    src/cli_summary.cpp
    src/cli_self_test.cpp
    src/cli_main.cpp
    tests/test_01_metadata_geometry.cpp
    tests/test_02_tour_exact.cpp
    tests/test_03_solver.cpp
    tests/test_04_restarts.cpp
    tests/test_05_neighborhoods_oracle.cpp
    tests/test_06_tsp_results.cpp
    PROPERTIES COMPILE_OPTIONS "-O0;-g0"
  )
endif()

string(REPLACE ";" " " ALDOUS_TSP_CXX_FLAGS_STR "${ALDOUS_TSP_CXX_FLAGS_EFFECTIVE_CONFIGURED}")
target_compile_definitions(aldous_tsp_core PRIVATE
  ALDOUS_TSP_BUILD_TYPE="$<CONFIG>"
  ALDOUS_TSP_BUILD_ENABLE_NATIVE=$<IF:$<BOOL:${ALDOUS_TSP_ENABLE_NATIVE}>,"ON","OFF">
  ALDOUS_TSP_BUILD_ENABLE_SANITIZERS=$<IF:$<OR:$<BOOL:${ALDOUS_TSP_ENABLE_SANITIZERS}>,$<BOOL:${ALDOUS_TSP_ENABLE_TSAN}>>,"ON","OFF">
  ALDOUS_TSP_BUILD_ENABLE_TSAN=$<IF:$<BOOL:${ALDOUS_TSP_ENABLE_TSAN}>,"ON","OFF">
  ALDOUS_TSP_BUILD_ENABLE_WARNINGS=$<IF:$<BOOL:${ALDOUS_TSP_ENABLE_WARNINGS}>,"ON","OFF">
  ALDOUS_TSP_BUILD_ENABLE_WERROR=$<IF:$<BOOL:${ALDOUS_TSP_ENABLE_WERROR}>,"ON","OFF">
  ALDOUS_TSP_BUILD_LOW_MEMORY=$<IF:$<BOOL:${ALDOUS_TSP_LOW_MEMORY_BUILD}>,"ON","OFF">
  ALDOUS_TSP_BUILD_CXX_FLAGS="${ALDOUS_TSP_CXX_FLAGS_STR}"
  ALDOUS_TSP_HAVE_POSIX_SPAWN_CHDIR_NP=$<BOOL:${ALDOUS_TSP_HAVE_POSIX_SPAWN_CHDIR_NP}>
)

set(ALDOUS_TSP_WARNING_FLAGS)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  list(APPEND ALDOUS_TSP_WARNING_FLAGS -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wold-style-cast)
  if(ALDOUS_TSP_ENABLE_WERROR)
    list(APPEND ALDOUS_TSP_WARNING_FLAGS -Werror)
  endif()
  if(ALDOUS_TSP_ENABLE_NATIVE)
    target_compile_options(aldous_tsp_core PRIVATE -march=native)
  endif()
  # The solver translation unit is intentionally algorithm-dense; using -O2 in
  # Release keeps compile-time memory bounded while preserving the production
  # optimization profile expected by users.
  target_compile_options(aldous_tsp_core PRIVATE $<$<CONFIG:Release>:-O2>)
  if(ALDOUS_TSP_ENABLE_SANITIZERS)
    target_compile_options(aldous_tsp_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(aldous_tsp_core PUBLIC -fsanitize=address,undefined)
  elseif(ALDOUS_TSP_ENABLE_TSAN)
    target_compile_options(aldous_tsp_core PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(aldous_tsp_core PUBLIC -fsanitize=thread)
  endif()
elseif(MSVC)
  list(APPEND ALDOUS_TSP_WARNING_FLAGS /W4)
  if(ALDOUS_TSP_ENABLE_WERROR)
    list(APPEND ALDOUS_TSP_WARNING_FLAGS /WX)
  endif()
endif()

set(ALDOUS_TSP_TARGET_COMPILE_OPTIONS_LIST)
set(ALDOUS_TSP_SOURCE_COMPILE_OPTIONS_LIST)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  if(ALDOUS_TSP_ENABLE_NATIVE)
    list(APPEND ALDOUS_TSP_TARGET_COMPILE_OPTIONS_LIST -march=native)
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND ALDOUS_TSP_TARGET_COMPILE_OPTIONS_LIST -O2)
  endif()
  if(ALDOUS_TSP_ENABLE_SANITIZERS)
    list(APPEND ALDOUS_TSP_TARGET_COMPILE_OPTIONS_LIST -fsanitize=address,undefined -fno-omit-frame-pointer)
  elseif(ALDOUS_TSP_ENABLE_TSAN)
    list(APPEND ALDOUS_TSP_TARGET_COMPILE_OPTIONS_LIST -fsanitize=thread -fno-omit-frame-pointer)
  endif()
  if(ALDOUS_TSP_LOW_MEMORY_BUILD)
    list(APPEND ALDOUS_TSP_SOURCE_COMPILE_OPTIONS_LIST -O0 -g0)
  endif()
endif()
if(ALDOUS_TSP_ENABLE_WARNINGS)
  list(APPEND ALDOUS_TSP_TARGET_COMPILE_OPTIONS_LIST ${ALDOUS_TSP_WARNING_FLAGS})
  target_compile_options(aldous_tsp_core PRIVATE ${ALDOUS_TSP_WARNING_FLAGS})
endif()
string(REPLACE ";" " " ALDOUS_TSP_TARGET_COMPILE_OPTIONS_STR "${ALDOUS_TSP_TARGET_COMPILE_OPTIONS_LIST}")
string(REPLACE ";" " " ALDOUS_TSP_SOURCE_COMPILE_OPTIONS_STR "${ALDOUS_TSP_SOURCE_COMPILE_OPTIONS_LIST}")
string(STRIP "${ALDOUS_TSP_CXX_FLAGS_STR} ${ALDOUS_TSP_TARGET_COMPILE_OPTIONS_STR} ${ALDOUS_TSP_SOURCE_COMPILE_OPTIONS_STR}" ALDOUS_TSP_EFFECTIVE_COMPILE_OPTIONS_STR)
target_compile_definitions(aldous_tsp_core PRIVATE
  ALDOUS_TSP_BUILD_TARGET_COMPILE_OPTIONS="${ALDOUS_TSP_TARGET_COMPILE_OPTIONS_STR}"
  ALDOUS_TSP_BUILD_SOURCE_COMPILE_OPTIONS="${ALDOUS_TSP_SOURCE_COMPILE_OPTIONS_STR}"
  ALDOUS_TSP_BUILD_EFFECTIVE_COMPILE_OPTIONS="${ALDOUS_TSP_EFFECTIVE_COMPILE_OPTIONS_STR}"
)

if(ALDOUS_TSP_ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    set_target_properties(aldous_tsp_core PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  endif()
endif()

if(ALDOUS_TSP_BUILD_CLI)
  add_library(aldous_tsp_cli_lib
    src/generated_options_cli.cpp
    src/cli_parse.cpp
    src/cli_values.cpp
    src/cli_summary.cpp
    src/cli_self_test.cpp
    src/cli_main.cpp
  )
  add_library(aldous_tsp::cli ALIAS aldous_tsp_cli_lib)
  set_target_properties(aldous_tsp_cli_lib PROPERTIES EXPORT_NAME cli)
  target_compile_features(aldous_tsp_cli_lib PUBLIC cxx_std_17)
  target_include_directories(aldous_tsp_cli_lib
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  target_link_libraries(aldous_tsp_cli_lib PUBLIC aldous_tsp::core)
  if(ALDOUS_TSP_LOW_MEMORY_BUILD AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(aldous_tsp_cli_lib PRIVATE -O0 -g0)
  endif()
  if(ALDOUS_TSP_ENABLE_WARNINGS)
    target_compile_options(aldous_tsp_cli_lib PRIVATE ${ALDOUS_TSP_WARNING_FLAGS})
  endif()

  add_executable(aldous_tsp apps/aldous_tsp_main.cpp)
  target_link_libraries(aldous_tsp PRIVATE aldous_tsp::cli)
endif()

if(ALDOUS_TSP_BUILD_FUZZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    message(FATAL_ERROR "ALDOUS_TSP_BUILD_FUZZERS requires Clang/libFuzzer")
  endif()
  if(NOT ALDOUS_TSP_BUILD_CLI)
    message(FATAL_ERROR "ALDOUS_TSP_BUILD_FUZZERS requires ALDOUS_TSP_BUILD_CLI=ON for fuzz_cli")
  endif()

  if(ALDOUS_TSP_FUZZ_ENABLE_ASAN)
    set(ALDOUS_TSP_FUZZ_SANITIZERS "address,undefined")
  else()
    set(ALDOUS_TSP_FUZZ_SANITIZERS "undefined")
  endif()
  set(ALDOUS_TSP_FUZZ_INSTRUMENTATION
    -fsanitize=fuzzer-no-link,${ALDOUS_TSP_FUZZ_SANITIZERS}
    -fno-omit-frame-pointer
    -gline-tables-only)
  target_compile_options(aldous_tsp_core PRIVATE ${ALDOUS_TSP_FUZZ_INSTRUMENTATION})
  target_compile_options(aldous_tsp_cli_lib PRIVATE ${ALDOUS_TSP_FUZZ_INSTRUMENTATION})
  target_link_options(aldous_tsp_core PUBLIC -fsanitize=${ALDOUS_TSP_FUZZ_SANITIZERS})

  function(aldous_tsp_add_fuzzer target source)
    add_executable(${target} ${source})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_include_directories(${target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/fuzz
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_options(${target} PRIVATE
      -fsanitize=fuzzer,${ALDOUS_TSP_FUZZ_SANITIZERS}
        -fno-omit-frame-pointer
      -gline-tables-only)
    target_link_options(${target} PRIVATE -fsanitize=fuzzer,${ALDOUS_TSP_FUZZ_SANITIZERS})
    target_link_libraries(${target} PRIVATE ${ARGN})
  endfunction()

  aldous_tsp_add_fuzzer(fuzz_cli fuzz/fuzz_cli.cpp aldous_tsp::cli)
  aldous_tsp_add_fuzzer(fuzz_instance fuzz/fuzz_instance.cpp aldous_tsp::core)
  aldous_tsp_add_fuzzer(fuzz_tour fuzz/fuzz_tour.cpp aldous_tsp::core)
  aldous_tsp_add_fuzzer(fuzz_solver fuzz/fuzz_solver.cpp aldous_tsp::core)
  add_custom_target(aldous_tsp_fuzzers DEPENDS
    fuzz_cli fuzz_instance fuzz_tour fuzz_solver)

  if(BUILD_TESTING)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fuzz-artifacts)
    set(_aldous_fuzz_smoke_corpus_root
      ${CMAKE_CURRENT_BINARY_DIR}/fuzz-smoke-corpus)
    file(REMOVE_RECURSE ${_aldous_fuzz_smoke_corpus_root})
    foreach(_fuzzer IN ITEMS cli instance tour solver)
      file(MAKE_DIRECTORY
        ${_aldous_fuzz_smoke_corpus_root}/${_fuzzer})
      file(COPY
        ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/corpus/${_fuzzer}/
        DESTINATION ${_aldous_fuzz_smoke_corpus_root}/${_fuzzer})
      add_test(NAME fuzz_${_fuzzer}_smoke
        COMMAND fuzz_${_fuzzer}
          -runs=250
          -print_funcs=0
          -timeout=15
          -rss_limit_mb=2048
          -artifact_prefix=${CMAKE_CURRENT_BINARY_DIR}/fuzz-artifacts/
          ${_aldous_fuzz_smoke_corpus_root}/${_fuzzer})
      set_tests_properties(fuzz_${_fuzzer}_smoke PROPERTIES
        LABELS fuzz-smoke
        TIMEOUT 90)
    endforeach()
  endif()
endif()


# Some constrained CI/container images use Ninja rules that do not create nested
# object-file directories before emitting compiler depfiles. Creating the known
# target object directories here is harmless on normal generators and keeps
# source-package builds reproducible across those environments.
foreach(_aldous_obj_dir IN ITEMS
  CMakeFiles/aldous_tsp_core.dir/src
  CMakeFiles/aldous_tsp_cli_lib.dir/src
  CMakeFiles/aldous_tsp.dir/apps
  CMakeFiles/aldous_tsp_test_core.dir/tests
)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${_aldous_obj_dir}")
endforeach()

if(ALDOUS_TSP_BUILD_EXAMPLES)
  add_executable(aldous_tsp_library_usage examples/library_usage.cpp)
  target_link_libraries(aldous_tsp_library_usage PRIVATE aldous_tsp::core)
endif()

if(BUILD_TESTING AND Python3_Interpreter_FOUND)
  add_test(NAME generated_options_check
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_options.py --check)
  set_tests_properties(generated_options_check PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30)
endif()

if(BUILD_TESTING AND ALDOUS_TSP_BUILD_TESTS)
  add_executable(aldous_tsp_test_core
    tests/test_main.cpp
    tests/test_support.cpp
    tests/test_01_metadata_geometry.cpp
    tests/test_02_tour_exact.cpp
    tests/test_03_solver.cpp
    tests/test_04_restarts.cpp
    tests/test_05_neighborhoods_oracle.cpp
    tests/test_06_tsp_results.cpp
  )
  target_link_libraries(aldous_tsp_test_core PRIVATE aldous_tsp::core)
  # Absolute path to the tests directory so the oracle round-trip test can find
  # the reference LKH stand-in regardless of the working directory.
  target_compile_definitions(aldous_tsp_test_core PRIVATE
    ALDOUS_TSP_TESTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests")
  # White-box access to internal search kernels (path relink step selection).
  target_include_directories(aldous_tsp_test_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  if(ALDOUS_TSP_LOW_MEMORY_BUILD AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(aldous_tsp_test_core PRIVATE -O0 -g0)
  endif()
  if(ALDOUS_TSP_ENABLE_WARNINGS)
    target_compile_options(aldous_tsp_test_core PRIVATE ${ALDOUS_TSP_WARNING_FLAGS})
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(aldous_tsp_test_core PRIVATE $<$<CONFIG:Release>:-O1>)
  elseif(MSVC)
    target_compile_options(aldous_tsp_test_core PRIVATE $<$<CONFIG:Release>:/O1>)
  endif()
  foreach(_core_group IN ITEMS 01_metadata_geometry 02_tour_exact 03_solver 04_restarts 05_neighborhoods_oracle 06_tsp_results)
    add_test(NAME core_${_core_group} COMMAND aldous_tsp_test_core test_${_core_group}.cpp)
    set_tests_properties(core_${_core_group} PROPERTIES TIMEOUT 180)
  endforeach()

  if(ALDOUS_TSP_BUILD_CLI)
    add_test(NAME cli_self_test COMMAND aldous_tsp --self-test)
    set_tests_properties(cli_self_test PROPERTIES TIMEOUT 90)
    add_test(NAME cli_quick_smoke COMMAND aldous_tsp --N 24 --instances 1 --threads 1 --p-values 0.25,1.0 --sa-iters 5 --restarts 1 --tsp-restarts 1 --tsp-ils 2 --final-exhaustive-k 80 --pair-exchange-passes 0 --ruin-recreate-rounds 0 --path-relink-top 0 --output quick-smoke.json --force)
    set_tests_properties(cli_quick_smoke PROPERTIES TIMEOUT 120)
    if(NOT WIN32)
      # The fake oracle is a POSIX shell script; the oracle plumbing itself is
      # covered by the C++ parser tests on all platforms.
      add_test(NAME cli_fake_oracle_smoke COMMAND aldous_tsp --N 20 --instances 1 --threads 1 --p-values 1.0 --sa-iters 0 --restarts 1 --tsp-restarts 1 --tsp-ils 0 --oracle lkh --lkh-path ${CMAKE_CURRENT_SOURCE_DIR}/tests/fake_lkh.sh --oracle-min-k 17 --oracle-max-k 50 --output fake-oracle-smoke.json --force)
      set_tests_properties(cli_fake_oracle_smoke PROPERTIES TIMEOUT 120)
    endif()
    if(ALDOUS_TSP_ENABLE_PYTHON_TESTS)
      add_test(NAME backend_parity_smoke COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/benchmark_parity.py --exe $<TARGET_FILE:aldous_tsp> --out-dir ${CMAKE_CURRENT_BINARY_DIR}/parity-smoke)
      set_tests_properties(backend_parity_smoke PROPERTIES TIMEOUT 180)
      add_test(NAME original_parity_compat_smoke COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/benchmark_parity.py --exe $<TARGET_FILE:aldous_tsp> --baseline-exe ${CMAKE_CURRENT_SOURCE_DIR}/tests/fake_original_cli.py --baseline-kind original --suite original-compat-smoke --out-dir ${CMAKE_CURRENT_BINARY_DIR}/parity-original-compat-smoke)
      set_tests_properties(original_parity_compat_smoke PROPERTIES TIMEOUT 240)
      add_test(NAME real_oracle_smoke_optional COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/oracle_real_smoke.py --exe $<TARGET_FILE:aldous_tsp> --out-dir ${CMAKE_CURRENT_BINARY_DIR}/real-oracle-smoke)
      set_tests_properties(real_oracle_smoke_optional PROPERTIES TIMEOUT 240)
      add_test(NAME cli_regressions COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_regressions.py $<TARGET_FILE:aldous_tsp> ${CMAKE_CURRENT_SOURCE_DIR})
      set_tests_properties(cli_regressions PROPERTIES TIMEOUT 240)
      add_test(NAME schema_validation COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/schema_validation.py $<TARGET_FILE:aldous_tsp> ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/schema-validation)
      set_tests_properties(schema_validation PROPERTIES TIMEOUT 180)
      add_test(NAME option_metadata_parity COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/options_metadata_test.py ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_FILE:aldous_tsp>)
      set_tests_properties(option_metadata_parity PROPERTIES TIMEOUT 90)
      add_test(NAME hotpath_benchmark_smoke COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/benchmark_hotpaths.py --baseline-exe $<TARGET_FILE:aldous_tsp> --candidate-exe $<TARGET_FILE:aldous_tsp> --suite smoke --repetitions 1 --out-dir ${CMAKE_CURRENT_BINARY_DIR}/hotpath-benchmark-smoke --check --max-wall-ratio 5.0 --max-quality-regression 0.0)
      set_tests_properties(hotpath_benchmark_smoke PROPERTIES TIMEOUT 240)
      add_test(NAME exhaustive_policy_benchmark_smoke COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/benchmark_exhaustive_policy.py --exe $<TARGET_FILE:aldous_tsp> --out-dir ${CMAKE_CURRENT_BINARY_DIR}/exhaustive-policy-smoke --N 72 --instances 1 --p-values 0.25,1.0 --sa-iters 20 --restarts 1 --tsp-restarts 1 --tsp-ils 4 --final-exhaustive-k 96 --check)
      set_tests_properties(exhaustive_policy_benchmark_smoke PROPERTIES TIMEOUT 180)
      add_test(NAME install_tree_regressions COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/install_tree_regressions.py ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
      set_tests_properties(install_tree_regressions PROPERTIES TIMEOUT 300)
      add_test(NAME validation_report_render COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/render_validation_report.py --validation-dir ${CMAKE_CURRENT_SOURCE_DIR}/validation_runs --output ${CMAKE_CURRENT_BINARY_DIR}/known_good_benchmarks.rendered.md)
      set_tests_properties(validation_report_render PROPERTIES TIMEOUT 60)
      add_test(NAME validation_artifacts_schema COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/validation_artifacts_schema.py ${CMAKE_CURRENT_SOURCE_DIR})
      set_tests_properties(validation_artifacts_schema PROPERTIES TIMEOUT 120)
      add_test(NAME schema_13_to_14_migration COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/schema_migration_test.py ${CMAKE_CURRENT_SOURCE_DIR})
      set_tests_properties(schema_13_to_14_migration PROPERTIES TIMEOUT 120)
      add_test(NAME quality_canary COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/quality_canary.py $<TARGET_FILE:aldous_tsp> ${CMAKE_CURRENT_BINARY_DIR}/quality-canary)
      set_tests_properties(quality_canary PROPERTIES TIMEOUT 300)
      add_test(NAME extrapolation_self_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/extrapolate_fpN.py --self-test)
      set_tests_properties(extrapolation_self_test PROPERTIES TIMEOUT 60)
      add_test(NAME campaign_analysis_self_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/analyze_campaign.py --self-test)
      set_tests_properties(campaign_analysis_self_test PROPERTIES TIMEOUT 120)
      add_test(NAME convergence_study_self_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/convergence_study.py --self-test)
      add_test(NAME policy_tuning_self_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/tune_search_policy.py --self-test)
      set_tests_properties(convergence_study_self_test PROPERTIES TIMEOUT 60)
    endif()
  endif()
endif()

install(TARGETS aldous_tsp_core
  EXPORT aldous_tspTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(ALDOUS_TSP_BUILD_CLI)
  install(TARGETS aldous_tsp aldous_tsp_cli_lib
    EXPORT aldous_tspTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()
install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERN "cli.hpp" EXCLUDE
)
if(ALDOUS_TSP_BUILD_CLI)
  install(FILES include/aldous_tsp/cli.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aldous_tsp)
endif()
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated/aldous_tsp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT aldous_tspTargets NAMESPACE aldous_tsp:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aldous_tsp)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/aldous_tspConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/aldous_tspConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aldous_tsp
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/aldous_tspConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/aldous_tspConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/aldous_tspConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aldous_tsp
)
