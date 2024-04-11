//===- Walk.cpp - Benchmark Op Traveral ---------------------------------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "TestBenchDialect.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <memory>

#include "benchmark/benchmark.h"

using namespace mlir;
void mlirBenchmarkInitLLVM(int argc, const char **argv);
namespace {

class ParserPrinter : public benchmark::Fixture {
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

  void populateTestModule(int num) {
    OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
    for (int j = 0; j < num; ++j) {
      Operation *op;
      if (j % 2) {
        op = b.create<EmptyOp>(unknownLoc);
      } else {
        op = b.create<OpWithRegion>(unknownLoc);
      }
      // Add some attributes.
      if (j % 2) {
        op->setAttr("some_attr", b.getI32IntegerAttr(j));
      } else {
        op->setAttr("another_attr", b.getStringAttr(Twine("hello") + Twine(j)));
      }
    }
  }

  std::unique_ptr<MLIRContext> ctx;
  OwningOpRef<ModuleOp> moduleOp;
  UnknownLoc unknownLoc;
};
} // namespace

BENCHMARK_DEFINE_F(ParserPrinter, parseTextualIR)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  populateTestModule(state.range(0));
  std::string data;
  llvm::raw_string_ostream os(data);
  AsmState asmState(*moduleOp, OpPrintingFlags());
  moduleOp->print(os, asmState);
  llvm::SourceMgr sourceMgr;
  std::unique_ptr<llvm::MemoryBuffer> buf =
      llvm::MemoryBuffer::getMemBufferCopy(data);
  sourceMgr.AddNewSourceBuffer(std::move(buf), SMLoc());
  for (auto _ : state) {
    MLIRContext parserContext;
    parserContext.loadDialect<TestBenchDialect>();
    ParserConfig config(&parserContext, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> result = parseSourceFile<ModuleOp>(sourceMgr, config);
    benchmark::DoNotOptimize(result.get());
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(ParserPrinter, parseTextualIR)
    ->Ranges({{10, 1 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(ParserPrinter, parseBytecodeIR)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  populateTestModule(state.range(0));
  std::string data;
  llvm::raw_string_ostream os(data);
  (void)writeBytecodeToFile(*moduleOp, os);
  std::unique_ptr<llvm::MemoryBuffer> buf =
      llvm::MemoryBuffer::getMemBufferCopy(data);
  for (auto _ : state) {
    MLIRContext parserContext;
    parserContext.loadDialect<TestBenchDialect>();
    ParserConfig config(&parserContext, /*verifyAfterParse=*/false);
    Block owningBlock;
    LogicalResult result = readBytecodeFile(*buf, &owningBlock, config);
    benchmark::DoNotOptimize(result);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(ParserPrinter, parseBytecodeIR)
    ->Ranges({{10, 1 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(ParserPrinter, printTextualIR)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  populateTestModule(state.range(0));
  // Print the module once so we know how much memory it will use. Pre-allocate
  // when measuring to not count the cost of reallocations.
  std::string data;
  llvm::raw_string_ostream os(data);
  AsmState asmState(*moduleOp, OpPrintingFlags());
  moduleOp->print(os, asmState);
  for (auto _ : state) {
    std::string benchData;
    llvm::raw_string_ostream os(benchData);
    benchData.reserve(data.size());
    AsmState asmState(*moduleOp, OpPrintingFlags());
    moduleOp->print(os, asmState);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(ParserPrinter, printTextualIR)
    ->Ranges({{10, 1 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(ParserPrinter, printBytecodeIR)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  populateTestModule(state.range(0));
  std::string data;
  llvm::raw_string_ostream os(data);
  // Print the module once so we know how much memory it will use. Pre-allocate
  // when measuring to not count the cost of reallocations.
  (void)writeBytecodeToFile(*moduleOp, os);
  for (auto _ : state) {
    std::string benchData;
    llvm::raw_string_ostream os(benchData);
    benchData.reserve(data.size());
    (void)writeBytecodeToFile(*moduleOp, os);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(ParserPrinter, printBytecodeIR)
    ->Ranges({{10, 1 * 1000 * 1000}})
    ->Complexity(benchmark::oN);
