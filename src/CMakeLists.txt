aux_source_directory(. SRC_LIST)
add_library(klt_lib ${SRC_LIST})
target_link_libraries(klt_lib ${OpenCV_LIBS})