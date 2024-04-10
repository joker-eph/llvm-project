add_mlir_dialect(TestBenchDialect testbench)

add_mlir_interface(TestBenchInterface)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBS
  ${dialect_libs}
  MLIRIR
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRTransformUtils
  LLVMAnalysis
  LLVMCore
)
add_benchmark(MLIR_IR_Benchmark
  Analysis.cpp
  Attributes.cpp
  ConstantFolding.cpp
  Cloning.cpp
  CreateOps.cpp
  DialectConversion.cpp
  GreedyRewriter.cpp
  Interfaces.cpp
  LoopUnrolling.cpp
  ParserPrinter.cpp
  TestBenchDialect.cpp
  Walk.cpp

 DEPENDS
  MLIRTestBenchDialectIncGen
  MLIRTestBenchInterfaceIncGen
  ${LIBS}
)

target_link_libraries(MLIR_IR_Benchmark
 PRIVATE
  ${LIBS}
)
