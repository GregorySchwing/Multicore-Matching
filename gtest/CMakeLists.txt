include_directories(${CUDA_INCLUDE_DIR} ${TBB_INCLUDE_DIR} ${MATCH_SOURCE_DIR}/src/ ${MATCH_SOURCE_DIR}/DotWriter/lib )

set(test_src
  src/run_all.cc
  src/tree_arithmetic.cc
  #src/matrix_mult_test.cc
)

link_directories(${Boost_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIRS} ${MATCH_SOURCE_DIR}/DotWriter/lib/)
add_executable(test_deepgreen ${DOT_LIBRARY_HEADERS} ${DOT_LIBRARY_SOURCES} ${test_src} ${MATCH_SOURCE_DIR}/src/ ${MATCH_SOURCE_DIR}/DotWriter/lib)
set_target_properties(test_deepgreen PROPERTIES
CUDA_ARCHITECTURES "35;37;60;70;80")
target_compile_options(test_deepgreen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

target_link_libraries(test_deepgreen gtest_main)
target_link_libraries(test_deepgreen ${CUDA_LIBRARY} cuda)
target_link_libraries(test_deepgreen ${TBB_LIBRARY})
target_link_libraries(test_deepgreen ${Boost_LIBRARIES})
target_link_libraries(test_deepgreen ${CURSES_LIBRARIES})
target_link_libraries(test_deepgreen ${TBB_IMPORTED_TARGETS})
target_link_libraries(test_deepgreen matcher)
target_link_libraries(test_deepgreen gtest )

add_test(
  NAME tests
  COMMAND test_deepgreen
)