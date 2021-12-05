add_subdirectory(PDL)
add_subdirectory(PDLInterp)

set(LLVM_OPTIONAL_SOURCES
  Traits.cpp
)

add_mlir_library(MLIRDialect
  Traits.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect

  LINK_LIBS PUBLIC
  MLIRIR
  )
