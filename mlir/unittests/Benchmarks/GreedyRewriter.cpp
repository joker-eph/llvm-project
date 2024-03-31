//===- GreedyRewriter.cpp - Benchmark Op Traveral -------------------------
//===//
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
#include "mlir/InitAllDialects.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/InitLLVM.h"

#include <memory>

#include "benchmark/benchmark.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace {

class GreedyRewriter : public benchmark::Fixture {
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

BENCHMARK_DEFINE_F(GreedyRewriter, empty)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  for (int j = 0; j < state.range(0); ++j) {
    b.create<EmptyOp>(unknownLoc);
  }
  RewritePatternSet patterns(ctx.get());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  for (auto _ : state) {
    (void)applyPatternsAndFoldGreedily(moduleOp.get(), frozenPatterns);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(GreedyRewriter, empty)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(GreedyRewriter, withCanonicalizationPatterns)
(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  DialectRegistry registry;
  registerAllDialects(registry);
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  for (int j = 0; j < state.range(0); ++j) {
    b.create<EmptyOp>(unknownLoc);
  }
  RewritePatternSet patterns(ctx.get());
  for (auto *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, ctx.get());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  for (auto _ : state) {
    (void)applyPatternsAndFoldGreedily(moduleOp.get(), frozenPatterns);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(GreedyRewriter, withCanonicalizationPatterns)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);


BENCHMARK_DEFINE_F(GreedyRewriter, withPatterns)
(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  for (int j = 0; j < 1000; ++j) {
    b.create<EmptyOp>(unknownLoc);
  }
  RewritePatternSet patterns(ctx.get());
  for (int j = 0; j < state.range(0); ++j) {
    LogicalResult (*implFn)(EmptyOp, PatternRewriter &rewriter) = [] (EmptyOp, PatternRewriter &) { return failure(); };
    patterns.add<EmptyOp>(implFn);
  }
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  for (auto _ : state) {
    (void)applyPatternsAndFoldGreedily(moduleOp.get(), frozenPatterns);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(GreedyRewriter, withPatterns)
    ->Ranges({{1, 1 * 1000 * 1000}})
    ->Complexity();
