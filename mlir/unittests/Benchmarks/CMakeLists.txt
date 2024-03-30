add_mlir_dialect(TestBenchDialect testbench)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_benchmark(MLIR_IR_Benchmark
 CreateOps.cpp
 TestBenchDialect.cpp
 Walk.cpp

 DEPENDS
 MLIRTestBenchDialectIncGen
)

target_link_libraries(MLIR_IR_Benchmark
  PRIVATE
  MLIRIR
  LLVMCore
  )
