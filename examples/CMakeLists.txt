file(GLOB EXAMPLE_SOURCES "*.cu")
# 为每个示例创建可执行文件
foreach(example_source ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_source} NAME_WE)
    add_executable(${example_name} ${example_source})
    target_link_libraries(${example_name} PRIVATE ${PROJECT_NAME}_static)
    
    # 设置包含目录
    target_include_directories(${example_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
endforeach()
