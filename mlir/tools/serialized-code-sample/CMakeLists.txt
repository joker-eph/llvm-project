

add_executable(
  binary-loader-test
  BinLoader.cpp
)

target_link_libraries(
  binary-loader-test
  PRIVATE
  LLVMOrcJIT
)
