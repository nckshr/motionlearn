add_library(utils src/utils.cpp)

add_library(NN src/BasicNN.cpp)
target_link_libraries(NN utils)

add_executable(test_nn test_nn.cpp)
target_link_libraries(test_nn NN utils)

#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/train.csv ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
