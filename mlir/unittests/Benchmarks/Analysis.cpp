//===- Analysis.cpp - Benchmark Attribtues Creation ---------------------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "TestBenchDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/InitLLVM.h"

#include <memory>

#include "benchmark/benchmark.h"

using namespace mlir;

void mlirBenchmarkInitLLVM(int argc, const char **argv) {
  static llvm::InitLLVM initOnce(argc, argv);
}

namespace {

class Analysis : public benchmark::Fixture {
public:
  void SetUp(::benchmark::State &state) final {
    const char *cmd = "bench";
    const char **argv = &cmd;
    int argc = 1;
    // Init LLVM to get backtraces on crash
    mlirBenchmarkInitLLVM(argc, argv);

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

BENCHMARK_DEFINE_F(Analysis, symbolTable)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  auto b = OpBuilder::atBlockBegin(moduleOp->getBody());
  for (int j = 0; j < state.range(0); ++j)
    b.create<SymbolOp>(unknownLoc, b.getStringAttr("symbol_" + Twine(j)));
  for (auto _ : state)
    SymbolTable symtab(*moduleOp);
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(Analysis, symbolTable)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);
