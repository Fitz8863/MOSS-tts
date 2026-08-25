add_executable(moss-tts-cli main.cpp)
target_link_libraries(moss-tts-cli PRIVATE moss-tts)
target_include_directories(moss-tts-cli PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
set_target_properties(moss-tts-cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
