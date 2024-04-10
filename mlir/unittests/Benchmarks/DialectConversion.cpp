//===- DialectConversion.cpp - Benchmark  ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Transforms/DialectConversion.h"
#include "TestBenchDialect.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

#include <memory>
#include <unistd.h>

#include "benchmark/benchmark.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

using namespace mlir;
namespace {

class DialectConversion : public benchmark::Fixture {
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

  void populateTestModule(int num) {
    moduleOp->getBody()->erase();
    moduleOp->getBodyRegion().push_back(new Block);
    ctx->loadDialect<arith::ArithDialect>();
    ctx->loadDialect<scf::SCFDialect>();
    ctx->loadDialect<func::FuncDialect>();
    ctx->loadDialect<TestBenchDialect>();

    OpBuilder moduleB = OpBuilder::atBlockBegin(moduleOp->getBody());
    auto intTy = IndexType::get(ctx.get());
    auto fTy = FunctionType::get(ctx.get(), {}, intTy);
    auto funcOp = moduleB.create<func::FuncOp>(unknownLoc, "folding", fTy);
    funcOp.addEntryBlock();
    ImplicitLocOpBuilder funcB = ImplicitLocOpBuilder::atBlockBegin(
        unknownLoc, &funcOp.getBody().front());
    Value workingSet[2];
    workingSet[0] = funcB.create<arith::ConstantIndexOp>(13);
    workingSet[1] = funcB.create<arith::ConstantIndexOp>(7907);
    for (int j = 0; j < num; ++j) {
      auto addOp = funcB.create<arith::AddIOp>(workingSet[0], workingSet[1]);
      auto subOp = funcB.create<arith::SubIOp>(workingSet[1], workingSet[0]);
      workingSet[0] = addOp;
      workingSet[1] = subOp;
      if (j % 2)
        std::swap(workingSet[0], workingSet[0]);
    }
    funcB.create<func::ReturnOp>(ValueRange{workingSet[0]});
    if (failed(verify(moduleOp.get()))) {
      llvm::errs() << "Verifier failed " << __FILE__ << ":" << __LINE__ << "\n";
      exit(-1);
    }
  }

  std::unique_ptr<MLIRContext> ctx;
  OwningOpRef<ModuleOp> moduleOp;
  UnknownLoc unknownLoc;
};
} // namespace

BENCHMARK_DEFINE_F(DialectConversion, noPatterns)(benchmark::State &state) {
  int testSize = state.range(0);
  populateTestModule(0);

  ConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  LLVMTypeConverter typeConverter(ctx.get());
  RewritePatternSet tempPatterns(ctx.get());
  FrozenRewritePatternSet patterns(std::move(tempPatterns));

  for (auto _ : state) {
    state.PauseTiming();
    populateTestModule(testSize);
    state.ResumeTiming();
    if (failed(applyPartialConversion(moduleOp.get(), target, patterns))) {
      llvm::errs() << "Conversion failed " << __FILE__ << ":" << __LINE__
                   << "\n";
      exit(-1);
    }
  }
  // exit(1);
  // llvmFunc->dump();
  if (failed(verify(moduleOp.get()))) {
    llvm::errs() << "Verifier failed " << __FILE__ << ":" << __LINE__ << "\n";
    exit(-1);
  }
  int countOps = 0;
  moduleOp->walk([&](Operation *op) { ++countOps; });
  // moduleOp->dump();
  int expectedOps = 5 + testSize * 2;
  if (countOps != expectedOps) {
    llvm::errs() << "Got " << countOps << " operation and expected "
                 << expectedOps << "\n";
    exit(1);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(DialectConversion, noPatterns)
    ->Ranges({{1, 1 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(DialectConversion, toLLVM)(benchmark::State &state) {
  int testSize = state.range(0);
  DialectRegistry registry;
  registry.insert<LLVM::LLVMDialect>();
  cf::registerConvertControlFlowToLLVMInterface(registry);
  arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  ctx->appendDialectRegistry(registry);
  populateTestModule(0);

  RewritePatternSet tempPatterns(ctx.get());
  ConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  LLVMTypeConverter typeConverter(ctx.get());
  // Normal mode: Populate all patterns from all dialects that implement the
  // interface.
  for (Dialect *dialect : ctx->getLoadedDialects()) {
    // First time we encounter this dialect: if it implements the interface,
    // let's populate patterns !
    auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
    if (!iface)
      continue;
    iface->populateConvertToLLVMConversionPatterns(target, typeConverter,
                                                   tempPatterns);
  }
    FrozenRewritePatternSet patterns(std::move(tempPatterns));

    for (auto _ : state) {
      state.PauseTiming();
      populateTestModule(testSize);
      if (failed(verify(moduleOp.get()))) {
        llvm::errs() << "Verifier failed " << __FILE__ << ":" << __LINE__
                     << "\n";
        exit(-1);
      }
      PassManager pm(ctx.get());
      pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
      if (failed(pm.run(*moduleOp))) {
        llvm::errs() << "failed to run the PM " << __FILE__ << ":" << __LINE__
                     << "\n";
        exit(-1);
      }
      if (failed(verify(moduleOp.get()))) {
        llvm::errs() << "Verifier failed " << __FILE__ << ":" << __LINE__
                     << "\n";
        exit(-1);
      }
      state.ResumeTiming();
      if (failed(applyPartialConversion(moduleOp.get(), target, patterns))) {
        llvm::errs() << "Conversion failed " << __FILE__ << ":" << __LINE__
                     << "\n";
        exit(-1);
      }
    }
    // exit(1);
    // llvmFunc->dump();
    if (failed(verify(moduleOp.get()))) {
      llvm::errs() << "Verifier failed " << __FILE__ << ":" << __LINE__ << "\n";
      exit(-1);
    }
    int countOps = 0;
    moduleOp->walk([&](Operation *op) { ++countOps; });
    // moduleOp->dump();
    int expectedOps = 5 + testSize * 2;
    if (countOps != expectedOps) {
      llvm::errs() << "Got " << countOps << " operation and expected "
                   << expectedOps << "\n";
      exit(1);
    }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(DialectConversion, toLLVM)
    ->Ranges({{1, 1 * 1000 * 1000}})
    ->Complexity(benchmark::oN);
