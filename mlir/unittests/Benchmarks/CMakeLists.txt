add_benchmark(MLIR_IR_Benchmark
 CreateOps.cpp
)

target_link_libraries(MLIR_IR_Benchmark
  PRIVATE
  MLIRIR
  )
