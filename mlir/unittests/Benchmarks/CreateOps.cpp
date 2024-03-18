//===- AdaptorTest.cpp - Adaptor unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <memory>

#include "benchmark/benchmark.h"

using namespace mlir;
namespace {
class CreateOps : public benchmark::Fixture {
public:
  void SetUp(::benchmark::State& state) final {
    ctx = std::make_unique<MLIRContext>();
    ctx->allowUnregisteredDialects();
    unknownLoc = UnknownLoc::get(ctx.get());
    block = std::make_unique<Block>();
  }

  void TearDown(::benchmark::State& state) final {
    block.reset();
    ctx.reset();
  }
  std::unique_ptr<MLIRContext> ctx;
  std::unique_ptr<Block> block;
  UnknownLoc unknownLoc;

};
}



BENCHMARK_DEFINE_F(CreateOps, simple)(benchmark::State& state) {
  for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
      OperationState opState(unknownLoc, "foo");
      Operation::create(opState);
    }
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(CreateOps, simple)->Ranges({{10, 10*1000*1000}})->Complexity();

BENCHMARK_DEFINE_F(CreateOps, hoistedOpState)(benchmark::State& state) {
  OperationState opState(unknownLoc, "foo");
  for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j)
      Operation::create(opState);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(CreateOps, hoistedOpState)->Ranges({{10, 10*1000*1000}})->Complexity();

BENCHMARK_DEFINE_F(CreateOps, withInsert)(benchmark::State& state) {
  for (auto _ : state) {
    OperationState opState(unknownLoc, "foo");
    for (int j = 0; j < state.range(0); ++j)
      block->push_back(Operation::create(opState));
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(CreateOps, withInsert)->Ranges({{10, 10*1000*1000}})->Complexity();



BENCHMARK_MAIN();

