//===- Attributes.cpp - Benchmark Attribtues Creation -------------------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "TestBenchDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/InitLLVM.h"

#include <memory>

#include "benchmark/benchmark.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace {

class AttributesBench : public benchmark::Fixture {
public:
  void SetUp(::benchmark::State &state) final {
    const char *cmd = "bench";
    const char **argv = &cmd;
    int argc = 1;
    // Init LLVM to get backtraces on crash
    static llvm::InitLLVM initOnce(argc, argv);

    ctx = std::make_unique<MLIRContext>();
    ctx->allowUnregisteredDialects();
    unknownLoc = UnknownLoc::get(ctx.get());
    moduleOp = OpBuilder(ctx.get()).create<ModuleOp>(unknownLoc);
  }

  void TearDown(::benchmark::State &state) final {
    moduleOp.release()->erase();
    ctx.reset();
  }
  std::unique_ptr<MLIRContext> ctx;
  OwningOpRef<ModuleOp> moduleOp;
  UnknownLoc unknownLoc;
};
} // namespace

BENCHMARK_DEFINE_F(AttributesBench, sameString)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
      StringAttr::get(ctx.get(), std::to_string(0));
    }
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, sameString)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(AttributesBench, newString)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
      StringAttr::get(ctx.get(), std::to_string(j));
    }
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, newString)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(AttributesBench, sameStringNoThreading)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  ctx->disableMultithreading();
  for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
      StringAttr::get(ctx.get(), std::to_string(0));
    }
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, sameStringNoThreading)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(AttributesBench, newStringNoThreading)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  ctx->disableMultithreading();
  for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
      StringAttr::get(ctx.get(), std::to_string(j));
    }
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, newStringNoThreading)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);
