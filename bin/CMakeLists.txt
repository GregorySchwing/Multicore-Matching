#include_directories(${CUDA_INCLUDE_DIR} ${TBB_INCLUDE_DIRS} ${MATCH_SOURCE_DIR}/src/ ${MATCH_SOURCE_DIR}/DotWriter/lib )
include_directories(${CUDA_INCLUDE_DIR} ${MATCH_SOURCE_DIR}/src/ ${MATCH_SOURCE_DIR}/DotWriter/lib )

file (GLOB DOT_LIBRARY_SOURCES ${MATCH_SOURCE_DIR}/DotWriter/lib/*.cpp)
file (GLOB DOT_LIBRARY_HEADERS ${MATCH_SOURCE_DIR}/DotWriter/lib/*.h)



cuda_add_executable(gpustat gpustat.cu)
target_link_libraries(gpustat ${CUDA_LIBRARY} cuda)

find_package(Boost COMPONENTS iostreams program_options REQUIRED)

link_directories(${Boost_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIRS} ${MATCH_SOURCE_DIR}/DotWriter/lib/)
add_executable(match match.cpp ${DOT_LIBRARY_HEADERS} ${DOT_LIBRARY_SOURCES})
set_target_properties(match PROPERTIES
CUDA_ARCHITECTURES "35;37;60;70;80")
target_compile_options(match PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)
#target_compile_options(match PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_37,code=sm_37>)
target_link_libraries(match ${CUDA_LIBRARY} cuda)
#target_link_libraries(match ${TBB_LIBRARIES})
target_link_libraries(match ${Boost_LIBRARIES})
target_link_libraries(match ${CURSES_LIBRARIES})
#target_link_libraries(match ${TBB_IMPORTED_TARGETS})
target_link_libraries(match matcher)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G --keep")

link_directories(${Boost_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIRS} ${MATCH_SOURCE_DIR}/DotWriter/lib/ ${CURSES_INCLUDE_DIR})
add_executable(cover cover.cpp ${DOT_LIBRARY_HEADERS} ${DOT_LIBRARY_SOURCES})
set_target_properties(cover PROPERTIES
CUDA_ARCHITECTURES "35;37;60;70;80"
#CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(matcher PROPERTIES
CUDA_ARCHITECTURES "35;37;60;70;80"
#CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(cover PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -g -G --keep
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)
#target_compile_options(cover PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_37,code=sm_37>)
target_link_libraries(cover ${CUDA_LIBRARY} cuda)
target_link_libraries(cover ${CURSES_LIBRARY})
target_link_libraries(cover curses)
target_link_libraries(cover ${CURSES_LIBRARIES})
target_link_libraries(cover ${NCURSES_LIBRARIES})

#target_link_libraries(cover ${TBB_IMPORTED_TARGETS})
target_link_libraries(cover ${Boost_LIBRARIES})
target_link_libraries(cover matcher)

add_executable(graphstat graphstat.cpp)
target_link_libraries(graphstat ${Boost_LIBRARIES})
