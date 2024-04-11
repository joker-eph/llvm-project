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
#include "llvm/Support/ThreadPool.h"

#include <memory>

#include "benchmark/benchmark.h"

using namespace mlir;
void mlirBenchmarkInitLLVM(int argc, const char **argv);
namespace {

class AttributesBench : public benchmark::Fixture {
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

BENCHMARK_DEFINE_F(AttributesBench, sameString)(benchmark::State &state) {
  for (auto _ : state) {
    MLIRContext ctx;
    for (int j = 0; j < state.range(0); ++j)
      StringAttr::get(&ctx, Twine(0));
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, sameString)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(AttributesBench, newString)(benchmark::State &state) {
  for (auto _ : state) {
    MLIRContext ctx;
    for (int j = 0; j < state.range(0); ++j)
      StringAttr::get(&ctx, Twine(j));
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, newString)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(AttributesBench, sameStringNoThreading)
(benchmark::State &state) {
  for (auto _ : state) {
    MLIRContext ctx;
    ctx.disableMultithreading();
    for (int j = 0; j < state.range(0); ++j)
      StringAttr::get(&ctx, Twine(0));
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, sameStringNoThreading)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(AttributesBench, newStringNoThreading)
(benchmark::State &state) {
  for (auto _ : state) {
    MLIRContext ctx;
    ctx.disableMultithreading();
    for (int j = 0; j < state.range(0); ++j)
      StringAttr::get(&ctx, Twine(j));
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, newStringNoThreading)
    ->Ranges({{10, 10 * 1000 * 1000}})
    ->Complexity(benchmark::oN);

static constexpr int numThreadsToBench = 4;

BENCHMARK_DEFINE_F(AttributesBench, sameStringMultithreaded)
(benchmark::State &state) {
  for (auto _ : state) {
    MLIRContext ctx;
    llvm::ThreadPoolInterface &tp = ctx.getThreadPool();
    llvm::ThreadPoolTaskGroup tg(tp);

    for (int i = 0; i < numThreadsToBench; ++i) {
      tg.async([&]() {
        for (int i = 0; i < state.range(0); ++i)
          StringAttr::get(&ctx, Twine(0));
      });
    }
    tg.wait();
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, sameStringMultithreaded)
    ->Ranges({{1, 10 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(AttributesBench, newStringMultithreaded)
(benchmark::State &state) {
  for (auto _ : state) {
    MLIRContext ctx;
    llvm::ThreadPoolInterface &tp = ctx.getThreadPool();
    llvm::ThreadPoolTaskGroup tg(tp);

    for (int i = 0; i < numThreadsToBench; ++i) {
      tg.async([&]() {
        for (int i = 0; i < state.range(0); ++i)
          StringAttr::get(&ctx, Twine(i));
      });
    }
    tg.wait();
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, newStringMultithreaded)
    ->Ranges({{1, 10 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(AttributesBench, newStringEachMultithreaded)
(benchmark::State &state) {
  for (auto _ : state) {
    MLIRContext ctx;
    llvm::ThreadPoolInterface &tp = ctx.getThreadPool();
    llvm::ThreadPoolTaskGroup tg(tp);

    for (int j = 0; j < numThreadsToBench; ++j) {
      tg.async([&, j]() {
        for (int i = 0; i < state.range(0); ++i)
          StringAttr::get(&ctx, Twine(i) + "_" + Twine(j));
      });
    }
    tg.wait();
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, newStringEachMultithreaded)
    ->Ranges({{1, 10 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(AttributesBench, setAttrRaw)(benchmark::State &state) {
  OperationState opState(unknownLoc, "testbench.op");
  auto attr = StringAttr::get(&*ctx, "some_attr");
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  for (auto _ : state) {
    Operation *op = b.create(opState);
    for (int j = 0; j < state.range(0); ++j) {
      op->removeAttr("some_attr");
      op->setAttr("some_attr", attr);
    }
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, setAttrRaw)
    ->Ranges({{1, 100 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(AttributesBench, setAttrProp)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  auto attr = StringAttr::get(&*ctx, "some_attr");
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  auto op = b.create<OpWithAttr>(unknownLoc, attr);
  for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
      op.removeAttrAttr();
      op.setAttrAttr(attr);
    }
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, setAttrProp)
    ->Ranges({{1, 100 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(AttributesBench, setProp)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  auto attr = StringAttr::get(&*ctx, "some_attr");
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  auto op = b.create<OpWithAttr>(unknownLoc, attr);
  for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
      op.getProperties().attr = nullptr;
      op.getProperties().attr = attr;
    }
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, setProp)
    ->Ranges({{1, 1000 * 1000}})
    ->Complexity(benchmark::oN);

BENCHMARK_DEFINE_F(AttributesBench, setPropHoist)(benchmark::State &state) {
  ctx->loadDialect<TestBenchDialect>();
  auto attr = StringAttr::get(&*ctx, "some_attr");
  OpBuilder b = OpBuilder::atBlockBegin(moduleOp->getBody());
  auto op = b.create<OpWithAttr>(unknownLoc, attr);
  auto &props = op.getProperties();
  for (auto _ : state) {
    for (int j = 0; j < state.range(0); ++j) {
      props.attr = nullptr;
      props.attr = attr;
    }
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK_REGISTER_F(AttributesBench, setPropHoist)
    ->Ranges({{1, 1000 * 1000}})
    ->Complexity(benchmark::oN);
