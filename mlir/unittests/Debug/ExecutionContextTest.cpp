//===- ExecutionContextTest.cpp - Debug Execution Context first impl ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Debug/ExecutionContext.h"
#include "mlir/Debug/BreakpointManagers/TagBreakpointManager.h"
#include "llvm/ADT/MapVector.h"
#include "gmock/gmock.h"

using namespace mlir;
using namespace mlir::tracing;

namespace {
struct DebuggerAction : public ActionImpl<DebuggerAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DebuggerAction)
  static constexpr StringLiteral tag = "debugger-action";
};
struct OtherAction : public ActionImpl<OtherAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OtherAction)
  static constexpr StringLiteral tag = "other-action";
};
struct ThirdAction : public ActionImpl<ThirdAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThirdAction)
  static constexpr StringLiteral tag = "third-action";
};

void noOp() { return; }

TEST(ExecutionContext, ActionActiveStackTest) {

  std::vector<std::vector<StringRef>> expectedStacks = {
      {DebuggerAction::tag},
      {OtherAction::tag, DebuggerAction::tag},
      {ThirdAction::tag, OtherAction::tag, DebuggerAction::tag}};
  std::vector<StringRef> currentStack;

  auto generateStack = [&](const ActionActiveStack *backtrace) {
    currentStack.clear();
    auto *cur = backtrace;
    while (cur != nullptr) {
      currentStack.push_back(cur->getAction().getTag());
      cur = cur->getParent();
    }
    return currentStack;
  };

  auto checkStacks = [&](const std::vector<StringRef> &currentStack,
                         const std::vector<StringRef> &expectedStack) {
    if (currentStack.size() != expectedStack.size()) {
      return false;
    }
    bool areEqual = true;
    for (int i = 0; i < (int)currentStack.size(); ++i) {
      if (currentStack[i] != expectedStack[i]) {
        areEqual = false;
        break;
      }
    }
    return areEqual;
  };

  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Step, ExecutionContext::Step, ExecutionContext::Apply};
  int idx = 0;
  StringRef current;
  int currentDepth = -1;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    current = backtrace->getAction().getTag();
    currentDepth = backtrace->getDepth();
    generateStack(backtrace);
    return controlSequence[idx++];
  };

  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  std::vector<TagBreakpoint *> breakpoints;
  breakpoints.push_back(simpleManager.addBreakpoint(DebuggerAction::tag));
  breakpoints.push_back(simpleManager.addBreakpoint(OtherAction::tag));
  breakpoints.push_back(simpleManager.addBreakpoint(ThirdAction::tag));

  auto third = [&]() {
    EXPECT_EQ(current, ThirdAction::tag);
    EXPECT_EQ(currentDepth, 2);
    EXPECT_TRUE(checkStacks(currentStack, expectedStacks[2]));
    return noOp();
  };
  auto nested = [&]() {
    EXPECT_EQ(current, OtherAction::tag);
    EXPECT_EQ(currentDepth, 1);
    EXPECT_TRUE(checkStacks(currentStack, expectedStacks[1]));
    executionCtx(third, ThirdAction{});
    return noOp();
  };
  auto original = [&]() {
    EXPECT_EQ(current, DebuggerAction::tag);
    EXPECT_EQ(currentDepth, 0);
    EXPECT_TRUE(checkStacks(currentStack, expectedStacks[0]));
    executionCtx(nested, OtherAction{});
    return noOp();
  };

  executionCtx(original, DebuggerAction{});
}

TEST(ExecutionContext, DebuggerTest) {
  int match = 0;
  auto onBreakpoint = [&match](const ActionActiveStack *backtrace) {
    match++;
    return ExecutionContext::Skip;
  };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);

  executionCtx(noOp, DebuggerAction{});
  EXPECT_EQ(match, 0);

  Breakpoint *dbgBreakpoint = simpleManager.addBreakpoint(DebuggerAction::tag);
  executionCtx(noOp, DebuggerAction{});
  EXPECT_EQ(match, 1);

  dbgBreakpoint->disable();
  executionCtx(noOp, DebuggerAction{});
  EXPECT_EQ(match, 1);

  dbgBreakpoint->enable();
  executionCtx(noOp, DebuggerAction{});
  EXPECT_EQ(match, 2);

  executionCtx(noOp, OtherAction{});
  EXPECT_EQ(match, 2);

  // Not yet implemented.
  //  simpleManager.deleteBreakpoint(dbgBreakpoint);
  //  EXPECT_TRUE(manager.execute<DebuggerAction>(noOp));
  // EXPECT_EQ(match, 2);
}

TEST(ExecutionContext, ApplyTest) {
  std::vector<StringRef> tagSequence = {DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);

  executionCtx(callback, DebuggerAction{});
  EXPECT_EQ(counter, 1);
}

TEST(ExecutionContext, SkipTest) {
  std::vector<StringRef> tagSequence = {DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Skip};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    assert(false);
    return noOp();
  };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);

  executionCtx(callback, DebuggerAction{});
  EXPECT_EQ(counter, 1);
}

TEST(ExecutionContext, StepApplyTest) {
  std::vector<StringRef> tagSequence = {DebuggerAction::tag, OtherAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Step, ExecutionContext::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);
  auto nested = [&]() {
    EXPECT_EQ(counter, 2);
    return noOp();
  };
  auto original = [&]() {
    EXPECT_EQ(counter, 1);
    executionCtx(nested, OtherAction{});
    return noOp();
  };

  executionCtx(original, DebuggerAction{});
  EXPECT_EQ(counter, 2);
}

TEST(ExecutionContext, StepNothingInsideTest) {
  std::vector<StringRef> tagSequence = {DebuggerAction::tag,
                                        DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Step, ExecutionContext::Step};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);

  executionCtx(callback, DebuggerAction{});
  EXPECT_EQ(counter, 2);
}

TEST(ExecutionContext, NextTest) {
  std::vector<StringRef> tagSequence = {DebuggerAction::tag,
                                        DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Next, ExecutionContext::Next};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);

  executionCtx(callback, DebuggerAction{});
  EXPECT_EQ(counter, 2);
}

TEST(ExecutionContext, FinishTest) {
  std::vector<StringRef> tagSequence = {DebuggerAction::tag, OtherAction::tag,
                                        DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Step, ExecutionContext::Finish,
      ExecutionContext::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);
  auto nested = [&]() {
    EXPECT_EQ(counter, 2);
    return noOp();
  };
  auto original = [&]() {
    EXPECT_EQ(counter, 1);
    executionCtx(nested, OtherAction{});
    EXPECT_EQ(counter, 2);
    return noOp();
  };

  executionCtx(original, DebuggerAction{});
  EXPECT_EQ(counter, 3);
}

TEST(ExecutionContext, FinishBreakpointInNestedTest) {
  std::vector<StringRef> tagSequence = {OtherAction::tag, DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Finish, ExecutionContext::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(OtherAction::tag);

  auto nested = [&]() {
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  auto original = [&]() {
    EXPECT_EQ(counter, 0);
    executionCtx(nested, OtherAction{});
    EXPECT_EQ(counter, 1);
    return noOp();
  };

  executionCtx(original, DebuggerAction{});
  EXPECT_EQ(counter, 2);
}

TEST(ExecutionContext, FinishNothingBackTest) {
  std::vector<StringRef> tagSequence = {DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Finish};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);

  executionCtx(callback, DebuggerAction{});
  EXPECT_EQ(counter, 1);
}

TEST(ExecutionContext, EnableDisableBreakpointOnCallback) {

  std::vector<StringRef> tagSequence = {DebuggerAction::tag, ThirdAction::tag,
                                        OtherAction::tag, DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Apply, ExecutionContext::Finish,
      ExecutionContext::Finish, ExecutionContext::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };

  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);
  Breakpoint *toBeDisabled = simpleManager.addBreakpoint(OtherAction::tag);

  auto third = [&]() {
    EXPECT_EQ(counter, 2);
    return noOp();
  };
  auto nested = [&]() {
    EXPECT_EQ(counter, 1);
    executionCtx(third, ThirdAction{});
    EXPECT_EQ(counter, 2);
    return noOp();
  };
  auto original = [&]() {
    EXPECT_EQ(counter, 1);
    toBeDisabled->disable();
    simpleManager.addBreakpoint(ThirdAction::tag);
    executionCtx(nested, OtherAction{});
    EXPECT_EQ(counter, 3);
    return noOp();
  };

  executionCtx(original, DebuggerAction{});
  EXPECT_EQ(counter, 4);
}
} // namespace
