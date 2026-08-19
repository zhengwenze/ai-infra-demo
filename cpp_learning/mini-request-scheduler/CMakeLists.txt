cmake_minimum_required(VERSION 3.14)
project(mini-request-scheduler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 源文件
set(SOURCES
    src/main.cpp
    src/blocking_queue.cpp
    src/timer.cpp
)

# 可执行文件
add_executable(mini-scheduler ${SOURCES})

# 线程库
find_package(Threads REQUIRED)
target_link_libraries(mini-scheduler PRIVATE Threads::Threads)

include(CTest)
if(BUILD_TESTING)
    add_executable(blocking-queue-test
        tests/blocking_queue_test.cpp
        src/blocking_queue.cpp
    )
    target_link_libraries(blocking-queue-test PRIVATE Threads::Threads)
    add_test(NAME blocking-queue-multiple-waiters COMMAND blocking-queue-test)
    set_tests_properties(blocking-queue-multiple-waiters PROPERTIES TIMEOUT 5)
endif()
