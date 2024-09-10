set(CMAKE_PREFIX_PATH "/usr/local/lib/cmake")
# 查找 Catch2
find_package(Catch2 3 REQUIRED)
include(Catch)

# 收集测试源文件
file(GLOB_RECURSE TEST_SOURCES "*.cpp" PROPERTIES LANGUAGE CUDA)

# 为每个测试源文件创建一个独立的测试可执行文件
foreach(TEST_SOURCE ${TEST_SOURCES})
    # 获取文件名（不包括扩展名）
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    # 创建测试可执行文件
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    # 链接主项目库和 Catch2
    target_link_libraries(${TEST_NAME} PRIVATE ${CMAKE_PROJECT_NAME}_static Catch2::Catch2WithMain)
    
    # 包含项目的 include 目录
    target_include_directories(${TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    
    # 注册测试
    catch_discover_tests(${TEST_NAME})

    install(TARGETS ${TEST_NAME}
            RUNTIME DESTINATION test)
endforeach()
install(DIRECTORY test_data/ DESTINATION test/test_data)
# 添加测试
include(CTest)
