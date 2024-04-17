//===- CreateOps.cpp - Benchmark Op Creation ----------------------------- ===//
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
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/InitLLVM.h"

#include <memory>

#include "benchmark/benchmark.h"

using namespace mlir;
void mlirBenchmarkInitLLVM(int argc, const char **argv);
namespace {
class CreateOps : public benchmark::Fixture {
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
    block = std::make_unique<Block>();
  }

  void TearDown(::benchmark::State &state) final {
    block.reset();
    ctx.reset();
  }
  std::unique_ptr<MLIRContext> ctx;
  std::unique_ptr<Block> block;
  UnknownLoc unknownLoc;
};
} // namespace

BENCHMARK_DEFINE_F(CreateOps, simple)(benchmark::State &state) {
  for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
      OperationState opState(unknownLoc, "testbench.empty");
      Operation::create(opState);
    }
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(CreateOps, simple)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(CreateOps, hoistedOpState)(benchmark::State &state) {
  OperationState opState(unknownLoc, "testbench.empty");
  for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j)
      Operation::create(opState);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(CreateOps, hoistedOpState)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(CreateOps, withInsert)(benchmark::State &state) {
  for (auto _ : state) {
    OperationState opState(unknownLoc, "testbench.empty");
    for (int j = 0; j < state.range(0); ++j)
      block->push_back(Operation::create(opState));
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(CreateOps, withInsert)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(CreateOps, simpleRegistered)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b(ctx.get());
  for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
      b.create<EmptyOp>(unknownLoc);
    }
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(CreateOps, simpleRegistered)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(CreateOps, withInsertRegistered)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b(ctx.get());
  b.setInsertionPoint(block.get(), block->begin());
  for (auto _ : state) {
    OperationState opState(unknownLoc, "foo");
    for (int j = 0; j < state.range(0); ++j)
      b.create<EmptyOp>(unknownLoc);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(CreateOps, withInsertRegistered)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(CreateOps, llvm_withInsertRegistered)
(benchmark::State &state) {
  llvm::LLVMContext ctx;
  auto module = std::make_unique<llvm::Module>("MyModule", ctx);
  auto *fTy = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
                                      /*isVarArg=*/false);
  auto *func = llvm::Function::Create(fTy, llvm::Function::ExternalLinkage, "",
                                      module.get());
  auto *block = llvm::BasicBlock::Create(ctx, "", func);
  llvm::IRBuilder<> builder(block);

  for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j)
      builder.CreateUnreachable();
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(CreateOps, llvm_withInsertRegistered)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_MAIN();
