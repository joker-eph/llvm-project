add_mlir_dialect(TestBenchDialect testbench)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBS
  ${dialect_libs}
  MLIRIR
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRTransformUtils
  LLVMCore
)
add_benchmark(MLIR_IR_Benchmark
  Attributes.cpp
  CreateOps.cpp
  GreedyRewriter.cpp
  LoopUnrolling.cpp
  ParserPrinter.cpp
  TestBenchDialect.cpp
  Walk.cpp

 DEPENDS
  MLIRTestBenchDialectIncGen
  ${LIBS}
)

target_link_libraries(MLIR_IR_Benchmark
 PRIVATE
  ${LIBS}
)
