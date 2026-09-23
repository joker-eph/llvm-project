if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT MSVC)
  # Avoid SLP optimization work across thousands of generated functions.
  # Limit inlining work while retaining the performance of the dialect hooks.
  set_source_files_properties(OpenMPDialect.cpp PROPERTIES COMPILE_OPTIONS
    "-fno-slp-vectorize;-mllvm;-inline-threshold=150")
endif()

add_mlir_dialect_library(MLIROpenMPDialect
  OpenMPDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  DEPENDS
  omp_gen
  MLIROpenMPOpsIncGen
  MLIROpenMPOpsInterfacesIncGen
  MLIROpenMPTypeInterfacesIncGen

  LINK_COMPONENTS
  TargetParser

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIROpenACCMPCommon
  )
