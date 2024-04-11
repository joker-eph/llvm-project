//===- Interface.cpp - Benchmark ----------------------------------------- ===//
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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/InitLLVM.h"

#include <memory>

#include "benchmark/benchmark.h"

using namespace mlir;
void mlirBenchmarkInitLLVM(int argc, const char **argv);
namespace {

class InterfaceBench : public benchmark::Fixture {
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

BENCHMARK_DEFINE_F(InterfaceBench, vectorTraveralCallInterfaceMethod)
(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  SmallVector<Operation *> ops;
  for (int j = 0; j < state.range(0); ++j) {
    ops.push_back(b.create<OpWithInterface>(unknownLoc));
  }
  for (auto _ : state) {
    for (Operation *op : ops) {
      auto casted = dyn_cast<TestBenchInterface>(op);
      if (!casted) {
        llvm::errs() << "Cast failed unexpectedly " << __FILE__ << ":" << __LINE__ << "\n";
        exit(-1);
      }
      auto cost = casted.getCost();
      benchmark::DoNotOptimize(&cost);
    };
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(InterfaceBench, vectorTraveralCallInterfaceMethod)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);


static int64_t getCost(llvm::Instruction *op) {
    using namespace llvm;
  switch (op->getOpcode()) {
  // Terminators
  case Instruction::Ret:    return 42;
  case Instruction::Br:     return 13;
  case Instruction::Switch: return 18;
  case Instruction::IndirectBr: return 22;
  case Instruction::Invoke: return 123;
  case Instruction::Resume: return 64;
  case Instruction::Unreachable: return 16;
  case Instruction::CleanupRet: return 75;
  case Instruction::CatchRet: return 67;
  case Instruction::CatchPad: return 678;
  case Instruction::CatchSwitch: return 10;
  case Instruction::CallBr: return 49;

  // Standard unary operators...
  case Instruction::FNeg: return 493;

  // Standard binary operators...
  case Instruction::Add: return 452;
  case Instruction::FAdd: return 19;
  case Instruction::Sub: return 248;
  case Instruction::FSub: return 682;
  case Instruction::Mul: return 494;
  case Instruction::FMul: return 731;
  case Instruction::UDiv: return 673;
  case Instruction::SDiv: return 574;
  case Instruction::FDiv: return 104;
  case Instruction::URem: return 484;
  case Instruction::SRem: return 4893;
  case Instruction::FRem: return 4209;

  // Logical operators...
  case Instruction::And: return 433;
  case Instruction::Or : return 35;
  case Instruction::Xor: return 3461;

  // Memory instructions...
  case Instruction::Alloca:        return 1094;
  case Instruction::Load:          return 585;
  case Instruction::Store:         return 385;
  case Instruction::AtomicCmpXchg: return 290249;
  case Instruction::AtomicRMW:     return 5712;
  case Instruction::Fence:         return 423;
  case Instruction::GetElementPtr: return 43412;

  // Convert instructions...
  case Instruction::Trunc:         return 2490;
  case Instruction::ZExt:          return 245;
  case Instruction::SExt:          return 2453;
  case Instruction::FPTrunc:       return 902580;
  case Instruction::FPExt:         return 20459;
  case Instruction::FPToUI:        return 29053390;
  case Instruction::FPToSI:        return 2309530;
  case Instruction::UIToFP:        return 219532;
  case Instruction::SIToFP:        return 930582;
  case Instruction::IntToPtr:      return 814792;
  case Instruction::PtrToInt:      return 79242;
  case Instruction::BitCast:       return 14290;
  case Instruction::AddrSpaceCast: return 234534;

  // Other instructions...
  case Instruction::ICmp:           return 18042;
  case Instruction::FCmp:           return 208597;
  case Instruction::PHI:            return 240853;
  case Instruction::Select:         return 320835;
  case Instruction::Call:           return 1981;
  case Instruction::Shl:            return 1085;
  case Instruction::LShr:           return 24085;
  case Instruction::AShr:           return 430584;
  case Instruction::VAArg:          return 24805;
  case Instruction::ExtractElement: return 12476;
  case Instruction::InsertElement:  return 18063;
  case Instruction::ShuffleVector:  return 3909322;
  case Instruction::ExtractValue:   return 10858035;
  case Instruction::InsertValue:    return 198538035;
  case Instruction::LandingPad:     return 24095903;
  case Instruction::CleanupPad:     return 2493;
  case Instruction::Freeze:         return 130953;

  default: llvm::errs() << "Invalid opcode\n"; exit(-1);

  };
}

BENCHMARK_DEFINE_F(InterfaceBench, llvm_vectorTraveralCallInterfaceMethod)(benchmark::State &state) {
  llvm::LLVMContext ctx;
  auto module = std::make_unique<llvm::Module>("MyModule", ctx);
  auto *fTy = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
                                      /*isVarArg=*/false);
  auto *func = llvm::Function::Create(fTy, llvm::Function::ExternalLinkage, "",
                                      module.get());
  auto *block = llvm::BasicBlock::Create(ctx, "", func);
  llvm::IRBuilder<> builder(block);

  SmallVector<llvm::Instruction *> ops;
  for (int j = 0; j < state.range(0); ++j) {
    ops.push_back(builder.CreateUnreachable());
  }
  for (auto _ : state) {
    for (llvm::Instruction *op : ops) {
      auto cost = getCost(op);
      benchmark::DoNotOptimize(&cost);
    };
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(InterfaceBench, llvm_vectorTraveralCallInterfaceMethod)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);
