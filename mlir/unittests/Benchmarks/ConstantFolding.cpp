//===- ConstantFolding.cpp - Benchmark  ---------------------------------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/ConstantFolding.h"
#include "TestBenchDialect.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
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
#include "mlir/Transforms/FoldUtils.h"
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Transforms/Utils/Local.h"

#include <memory>
#include <unistd.h>

#include "benchmark/benchmark.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

using namespace mlir;
namespace {

class ConstantFolding : public benchmark::Fixture,
                        public RewriterBase::Listener {
public:
  SmallVector<Operation *> existingConstants;
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override {
    existingConstants.push_back(op);
  }
  void notifyOperationErased(Operation *op) override {
    auto *it = llvm::find(existingConstants, op);
    if (it != existingConstants.end())
      existingConstants.erase(it);
  }

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

BENCHMARK_DEFINE_F(ConstantFolding, folding)(benchmark::State &state) {
  ctx->disableMultithreading();
  int testSize = state.range(0);
  for (auto _ : state) {
    state.PauseTiming();
    moduleOp->getBody()->erase();
    moduleOp->getBodyRegion().push_back(new Block);
    populateTestModule(testSize);
    auto funcOp = cast<func::FuncOp>(moduleOp->getBody()->front());
    if (!funcOp) {
      llvm::errs() << "Missing funcOp in module\n";
      exit(-1);
    }
    SmallVector<OpFoldResult, 8> foldResults;
    DenseMap<Attribute, Operation *> constantsMap;
    state.ResumeTiming();

    for (Operation &op :
         llvm::make_early_inc_range(funcOp.getBody().front().getOperations())) {
      Attribute constValue;
      matchPattern(&op, m_Constant(&constValue));
      if (constValue) {
        auto it = constantsMap.insert({constValue, &op});
        if (!it.second) {
          op.replaceAllUsesWith(it.first->getSecond());
          op.erase();
        }
        continue;
      }
      foldResults.clear();
      if (succeeded(op.fold(foldResults))) {
        // Check to see if the operation was just updated in place.
        if (foldResults.empty())
          continue;
        assert(foldResults.size() == op.getNumResults());
        auto *dialect = op.getDialect();
        OpBuilder b(&op);
        auto loc = op.getLoc();
        for (auto [cst, res] : llvm::zip(foldResults, op.getResults())) {
          if (cst.is<Value>()) {
            res.replaceAllUsesWith(cst.get<Value>());
            if (op.use_empty())
              op.erase();
            continue;
          }
          constValue = cst.get<Attribute>();
          auto it = constantsMap.find(constValue);
          if (it != constantsMap.end()) {
            res.replaceAllUsesWith(it->getSecond()->getResult(0));
            if (op.use_empty())
              op.erase();
            continue;
          }

          auto type = res.getType();
          // Ask the dialect to materialize a constant operation for this value.
          if (auto *constOp =
                  dialect->materializeConstant(b, constValue, type, loc)) {
            res.replaceAllUsesWith(constOp->getResult(0));
            if (op.use_empty())
              op.erase();
            auto it = constantsMap.insert({constValue, constOp});
            if (!it.second)
              llvm::report_fatal_error("Fatal");
          }
        }
      }
    }
    // By the time we are done, we may have simplified a bunch of code,
    // leaving around dead constants. Check for them now and remove them.
    for (auto it : constantsMap) {
      if (it.getSecond()->use_empty())
        it.getSecond()->erase();
    }
    state.PauseTiming();
    // funcOp->dump();

    if (failed(verify(moduleOp.get()))) {
      llvm::errs() << "Verifier failed " << __FILE__ << ":" << __LINE__ << "\n";
      exit(-1);
    }
    state.ResumeTiming();
  }
  // moduleOp->dump();
  // exit(-1);
  int countOps = 0;
  moduleOp->walk([&](Operation *op) { ++countOps; });
  int expectedOps = 4; // module + func + constant + return
  if (countOps != expectedOps) {
    llvm::errs() << "Got " << countOps << " operation and expected "
                 << expectedOps << "\n";
    exit(1);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(ConstantFolding, folding)
    ->Ranges({{1, 10000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(ConstantFolding, opFolder)(benchmark::State &state) {
  ctx->disableMultithreading();
  int testSize = state.range(0);
  for (auto _ : state) {
    state.PauseTiming();
    moduleOp->getBody()->erase();
    moduleOp->getBodyRegion().push_back(new Block);
    populateTestModule(testSize);
    auto funcOp = cast<func::FuncOp>(moduleOp->getBody()->front());
    if (!funcOp) {
      llvm::errs() << "Missing funcOp in module\n";
      exit(-1);
    }

    existingConstants.clear();

    state.ResumeTiming();

    // Fold the constants in reverse so that the last generated constants
    // from folding are at the beginning. This creates somewhat of a linear
    // ordering to the newly generated constants that matches the operation
    // order and improves the readability of test cases.
    OperationFolder helper(funcOp->getContext(), /*listener=*/this);
    for (Operation &op :
         llvm::make_early_inc_range(funcOp.getBody().front().getOperations()))
      (void)helper.tryToFold(&op);

    // By the time we are done, we may have simplified a bunch of code,
    // leaving around dead constants. Check for them now and remove them.
    for (auto *cst : existingConstants) {
      if (cst->use_empty())
        cst->erase();
    }
    state.PauseTiming();

    if (failed(verify(moduleOp.get()))) {
      llvm::errs() << "Verifier failed " << __FILE__ << ":" << __LINE__ << "\n";
      exit(-1);
    }
    state.ResumeTiming();
  }
  int countOps = 0;
  moduleOp->walk([&](Operation *op) { ++countOps; });
  int expectedOps = 4;
  if (countOps != expectedOps) {
    llvm::errs() << "Got " << countOps << " ops and expected " << expectedOps
                 << "\n";
    exit(1);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(ConstantFolding, opFolder)
    ->Ranges({{1, 10000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(ConstantFolding, llvm_folding)
(benchmark::State &state) {
  int testSize = state.range(0);
  llvm::LLVMContext llvmContext;
  DialectRegistry registry;
  registerConvertFuncToLLVMInterface(registry);
  // index::registerConvertIndexToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  arith::registerConvertArithToLLVMInterface(registry);
  registerLLVMDialectTranslation(registry);
  registerBuiltinDialectTranslation(registry);
  ctx->appendDialectRegistry(registry);
  populateTestModule(testSize);
  if (failed(verify(moduleOp.get()))) {
    llvm::errs() << "Verifier failed " << __FILE__ << ":" << __LINE__ << "\n";
    exit(-1);
  }
  PassManager pm(ctx.get());
  pm.addPass(createConvertToLLVMPass());
  if (failed(pm.run(*moduleOp))) {
    llvm::errs() << "failed to run the PM " << __FILE__ << ":" << __LINE__
                 << "\n";
    exit(-1);
  }
  if (failed(verify(moduleOp.get()))) {
    llvm::errs() << "Verifier failed " << __FILE__ << ":" << __LINE__ << "\n";
    exit(-1);
  }

  std::unique_ptr<llvm::Module> llvmMod;
  llvm::Function *llvmFunc{};
  // const char *args[] = {"bench", "-debug"};
  // const char **argv = args;
  // int argc = 2;
  // llvm::cl::ParseCommandLineOptions(argc, argv);
  for (auto _ : state) {
    state.PauseTiming();
    // moduleOp->dump();
    llvmMod = translateModuleToLLVMIR(moduleOp.get(), llvmContext);
    // llvmMod->dump();
    if (!llvmMod) {
      llvm::errs() << "Translate to LLVM failed " << __FILE__ << ":" << __LINE__
                   << "\n";
      exit(-1);
    }

    llvmFunc = llvmMod->getFunction("folding");
    if (!llvmFunc) {
      llvm::errs() << "Can't lookup function in LLVM module " << __FILE__ << ":"
                   << __LINE__ << "\n";
      exit(-1);
    }
    // llvmFunc->print(llvm::errs());
    // llvm::errs() << '\n';
    // exit(1);

    llvm::DataLayout DL(llvmMod.get());
    llvm::TargetLibraryInfoImpl TLII;
    llvm::TargetLibraryInfo TLI(TLII);
    state.ResumeTiming();
    for (llvm::Instruction &I :
         llvm::make_early_inc_range(llvmFunc->getEntryBlock())) {
      if (llvm::Constant *C = llvm::ConstantFoldInstruction(&I, DL, &TLI)) {
        // llvm::errs() << "IC: ConstFold to: " << *C << " from: " << I <<
        // '\n';
        I.replaceAllUsesWith(C);
        if (llvm::isInstructionTriviallyDead(&I, &TLI))
          I.eraseFromParent();
      }
    }
    // llvmFunc->print(llvm::errs());
    // llvm::errs() << '\n';
    // exit(1);
  }
  int countBlocks = 0;
  for ([[maybe_unused]] llvm::BasicBlock &B : *llvmFunc)
    countBlocks++;
  int expectedBlocks = 1;
  if (countBlocks != expectedBlocks) {
    llvm::errs() << "Got " << countBlocks << " blocks and expected "
                 << expectedBlocks << "\n";
    exit(1);
  }
  int countOps = 0;
  for ([[maybe_unused]] llvm::Instruction &I : llvmFunc->getEntryBlock()) {
    countOps++;
  }
  int expectedOp = 1;
  if (countOps != expectedOp) {
    llvm::errs() << "Got " << countBlocks << " ops and expected "
                 << expectedBlocks << "\n";
    exit(1);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(ConstantFolding, llvm_folding)
    ->Ranges({{1, 1000}})
    ->Complexity(benchmark::oN);
