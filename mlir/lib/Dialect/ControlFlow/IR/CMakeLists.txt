add_mlir_dialect_library(MLIRControlFlowDialect
  ControlFlowOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ControlFlow/IR

  DEPENDS
  MLIRControlFlowOpsIncGen

  # ControlFlowOps.cpp includes generated Bufferization interfaces.
  HEADER_LIBS
  MLIRBufferizationDialect

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRUBDialect
  )
