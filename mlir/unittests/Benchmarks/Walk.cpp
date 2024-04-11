//===- Walk.cpp - Benchmark Op Traveral ---------------------------------- ===//
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

class IRWalk : public benchmark::Fixture {
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

BENCHMARK_DEFINE_F(IRWalk, blockTraveral)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  for (int j = 0; j < state.range(0); ++j) {
    b.create<EmptyOp>(unknownLoc);
  }
  Block *block = moduleOp->getBody();
  for (auto _ : state) {
    for (Operation &op : *block) {
      benchmark::DoNotOptimize(&op);
    };
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(IRWalk, blockTraveral)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(IRWalk, vectorTraveral)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  SmallVector<Operation *> ops;
  for (int j = 0; j < state.range(0); ++j) {
    ops.push_back(b.create<EmptyOp>(unknownLoc));
  }
  for (auto _ : state) {
    for (Operation *op : ops) {
      benchmark::DoNotOptimize(op);
    };
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(IRWalk, vectorTraveral)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(IRWalk, simpleWalk)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  for (int j = 0; j < state.range(0); ++j) {
    b.create<EmptyOp>(unknownLoc);
  }
  Block *block = moduleOp->getBody();
  for (auto _ : state) {
    block->walk([](Operation *op) { benchmark::DoNotOptimize(&op); });
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(IRWalk, simpleWalk)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(IRWalk, filteredOps)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  for (int j = 0; j < state.range(0); ++j) {
    b.create<EmptyOp>(unknownLoc);
  }
  for (auto _ : state) {
    // This is filtering out all the ops.
    for (ModuleOp op : moduleOp->getBody()->getOps<ModuleOp>()) {
      benchmark::DoNotOptimize(op.getOperation());
    }
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(IRWalk, filteredOps)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(IRWalk, nestedRegion)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  for (int j = 0; j < state.range(0); ++j) {
    auto op = b.create<OpWithRegion>(unknownLoc);
    op.getBody().push_back(new Block);
    // b.setInsertionPoint(&op.getBody().front(), op.getBody().front().begin());
  }
  for (auto _ : state) {
    moduleOp->walk([](Operation *op) { benchmark::DoNotOptimize(&op); });
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(IRWalk, nestedRegion)
    ->RangeMultiplier(4)
    ->Ranges({{10, 8 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(IRWalk, llvm_blockTraversal)(benchmark::State &state) {
  llvm::LLVMContext ctx;
  auto module = std::make_unique<llvm::Module>("MyModule", ctx);
  auto *fTy = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
                                      /*isVarArg=*/false);
  auto *func = llvm::Function::Create(fTy, llvm::Function::ExternalLinkage, "",
                                      module.get());
  auto *block = llvm::BasicBlock::Create(ctx, "", func);
  llvm::IRBuilder<> builder(block);

  for (int j = 0; j < state.range(0); ++j) {
    builder.CreateUnreachable();
  }
  for (auto _ : state) {
    for (llvm::Instruction &op : *block) {
      benchmark::DoNotOptimize(&op);
    };
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(IRWalk, llvm_blockTraversal)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(IRWalk, vectorTraveralOpCastFail)
(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  SmallVector<Operation *> ops;
  for (int j = 0; j < state.range(0); ++j) {
    ops.push_back(b.create<EmptyOp>(unknownLoc));
  }
  for (auto _ : state) {
    for (Operation *op : ops) {
      auto casted = dyn_cast<OpWithRegion>(op);
      benchmark::DoNotOptimize(&casted);
    };
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(IRWalk, vectorTraveralOpCastFail)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(IRWalk, vectorTraveralOpCastSuccess)
(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  SmallVector<Operation *> ops;
  for (int j = 0; j < state.range(0); ++j) {
    ops.push_back(b.create<EmptyOp>(unknownLoc));
  }
  for (auto _ : state) {
    for (Operation *op : ops) {
      auto casted = dyn_cast<EmptyOp>(op);
      benchmark::DoNotOptimize(&casted);
    };
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(IRWalk, vectorTraveralOpCastSuccess)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(IRWalk, vectorTraveralWithoutInterfaceCastFail)
(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  SmallVector<Operation *> ops;
  for (int j = 0; j < state.range(0); ++j) {
    ops.push_back(b.create<EmptyOp>(unknownLoc));
  }
  for (auto _ : state) {
    for (Operation *op : ops) {
      auto iface = dyn_cast<RegionBranchOpInterface>(op);
      benchmark::DoNotOptimize(&iface);
    };
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(IRWalk, vectorTraveralWithoutInterfaceCastFail)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(IRWalk, vectorTraveralWithInterfaceCastFail)
(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  SmallVector<Operation *> ops;
  for (int j = 0; j < state.range(0); ++j) {
    ops.push_back(b.create<OpWithRegion>(unknownLoc));
  }
  for (auto _ : state) {
    for (Operation *op : ops) {
      auto iface = dyn_cast<CastOpInterface>(op);
      benchmark::DoNotOptimize(&iface);
    };
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(IRWalk, vectorTraveralWithInterfaceCastFail)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(IRWalk, vectorTraveralWithInterfaceCastSuccess)
(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  SmallVector<Operation *> ops;
  for (int j = 0; j < state.range(0); ++j) {
    ops.push_back(b.create<OpWithInterface>(unknownLoc));
  }
  for (auto _ : state) {
    for (Operation *op : ops) {
      auto iface = dyn_cast<TestBenchInterface>(op);
      if (!iface) {
        llvm::errs() << "Cast failed unexpectedly " << __FILE__ << ":" << __LINE__ << "\n";
        exit(-1);
      }
      benchmark::DoNotOptimize(&iface);
    };
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(IRWalk, vectorTraveralWithInterfaceCastSuccess)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(IRWalk, vectorTraveralOpTraitFail)
(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  SmallVector<Operation *> ops;
  for (int j = 0; j < state.range(0); ++j) {
    ops.push_back(b.create<EmptyOp>(unknownLoc));
  }
  for (auto _ : state) {
    for (Operation *op : ops) {
      bool hasTrait = op->hasTrait<OpTrait::SingleBlock>();
      benchmark::DoNotOptimize(&hasTrait);
    };
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(IRWalk, vectorTraveralOpTraitFail)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(IRWalk, vectorTraveralOpTraitSuccess)
(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  SmallVector<Operation *> ops;
  for (int j = 0; j < state.range(0); ++j) {
    ops.push_back(b.create<OpWithRegion>(unknownLoc));
  }
  for (auto _ : state) {
    for (Operation *op : ops) {
      bool hasTrait = op->hasTrait<OpTrait::SingleBlock>();
      benchmark::DoNotOptimize(&hasTrait);
    };
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(IRWalk, vectorTraveralOpTraitSuccess)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);
