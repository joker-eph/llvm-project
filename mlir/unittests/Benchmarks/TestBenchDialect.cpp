#include "TestBenchDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

TestBenchDialect *sdfsdf;

#include "TestBenchInterface.cpp.inc"

#include "TestBenchDialectDialect.cpp.inc"

void TestBenchDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TestBenchDialect.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "TestBenchDialect.cpp.inc"
