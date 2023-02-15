//===- GdbDebugExecutionContextHook.cpp - GDB for Debugger Support --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Debug/GdbDebugExecutionContextHook.h"

#include "mlir/Debug/BreakpointManagers/FileLineColLocBreakpointManager.h"
// #include "mlir/Debug/BreakpointManagers/RewritePatternBreakpointManager.h"
#include "mlir/Debug/BreakpointManagers/TagBreakpointManager.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Unit.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstddef>
#include <signal.h>

using namespace mlir;
using namespace mlir::tracing;

namespace {
struct DebuggerState {
  ExecutionContext::Control gdbControl = ExecutionContext::Apply;
  TagBreakpointManager tagBreakpointManager;
  FileLineColLocBreakpointManager fileLineColLocBreakpointManager;
  DenseMap<unsigned, Breakpoint *> breakpointIdsMap;
  const tracing::ActionActiveStack *actionActiveStack;
  IRUnit cursor;
};
} // namespace

static DebuggerState &getGlobalDebuggerState() {
  static thread_local DebuggerState debuggerState;
  return debuggerState;
}

extern "C" {
void mlirDebuggerSetControl(int controlOption) {
  getGlobalDebuggerState().gdbControl =
      static_cast<ExecutionContext::Control>(controlOption);
}

void mlirDebuggerEnableBreakpoint(breakpointHandle breakpoint) {
  reinterpret_cast<Breakpoint *>(breakpoint)->enable();
}

void mlirDebuggerDisableBreakpoint(breakpointHandle breakpoint) {
  reinterpret_cast<Breakpoint *>(breakpoint)->disable();
}

void mlirDebuggerPrintContext() {
  DebuggerState &state = getGlobalDebuggerState();
  if (!state.actionActiveStack) {
    llvm::errs() << "No active action.\n";
    return;
  }
  const ArrayRef<IRUnit> &units =
      state.actionActiveStack->getAction().getContextIRUnits();
  llvm::errs() << units.size() << " available IRUnits:\n";
  llvm::errs() << " (";
  interleaveComma(units, llvm::errs(),
                  [&](const IRUnit &unit) { printIRUnit(unit, llvm::errs()); });
  llvm::errs() << ")\n";
}

void mlirDebuggerPrintActionBacktrace(bool withContext) {
  DebuggerState &state = getGlobalDebuggerState();
  if (!state.actionActiveStack) {
    llvm::errs() << "No active action.\n";
    return;
  }
  state.actionActiveStack->print(llvm::errs(), withContext);
}

void mlirDebuggerPrintCursor() {
  printIRUnit(getGlobalDebuggerState().cursor, llvm::errs());
}

void mlirDebuggerCursorSelectIRUnitFromContext(int index) {
  auto &state = getGlobalDebuggerState();
  if (!state.actionActiveStack) {
    llvm::errs() << "No active MLIR Action stack\n";
    return;
  }
  ArrayRef<IRUnit> units =
      state.actionActiveStack->getAction().getContextIRUnits();
  if (index < 0 || index >= static_cast<int>(units.size())) {
    llvm::errs() << "Index invalid, bounds: [0, " << units.size()
                 << "] but got " << index << "\n";
    return;
  }
  state.cursor = units[index];
  printIRUnit(state.cursor, llvm::errs());
}

void mlirDebuggerCursorSelectParentIRUnit(irunitHandle irUnitPtr) {
  auto &state = getGlobalDebuggerState();
  if (!state.cursor) {
    llvm::errs() << "No active MLIR cursor, select from the context first\n";
    return;
  }
  IRUnit *unit = reinterpret_cast<IRUnit *>(irUnitPtr);
  if (auto *op = unit->dyn_cast<Operation *>()) {
    state.cursor = op->getBlock();
  } else if (auto *region = unit->dyn_cast<Region *>()) {

    state.cursor = region->getParentOp();
  } else if (auto *block = unit->dyn_cast<Block *>()) {
    state.cursor = block->getParent();
  } else {
    llvm::errs() << "Current cursor is not a valid IRUnit";
    return;
  }
  printIRUnit(state.cursor, llvm::errs());
}

void mlirDebuggerCursorSelectChildIRUnit(irunitHandle irUnitPtr, int index) {
  auto &state = getGlobalDebuggerState();
  if (!state.cursor) {
    llvm::errs() << "No active MLIR cursor, select from the context first\n";
    return;
  }
  IRUnit *unit = reinterpret_cast<IRUnit *>(irUnitPtr);
  if (auto *op = unit->dyn_cast<Operation *>()) {
    if (index < 0 || index >= static_cast<int>(op->getNumRegions())) {
      llvm::errs() << "Index invalid, op has " << op->getNumRegions()
                   << " but got " << index << "\n";
      return;
    }
    state.cursor = &op->getRegion(index);
  } else if (auto *region = unit->dyn_cast<Region *>()) {
    auto block = region->begin();
    int count = 0;
    while (block != region->end() && count != index) {
      ++block;
      ++count;
    }

    if (block == region->end()) {
      llvm::errs() << "Index invalid, region has " << count << " block but got "
                   << index << "\n";
      return;
    }
    state.cursor = &*block;
  } else if (auto *block = unit->dyn_cast<Block *>()) {
    auto op = block->begin();
    int count = 0;
    while (op != block->end() && count != index) {
      ++op;
      ++count;
    }

    if (op == block->end()) {
      llvm::errs() << "Index invalid, block has " << count
                   << "operations but got " << index << "\n";
      return;
    }
    state.cursor = &*op;
  } else {
    llvm::errs() << "Current cursor is not a valid IRUnit";
    return;
  }
  printIRUnit(state.cursor, llvm::errs());
}

void mlirDebuggerCursorSelectPreviousIRUnit(irunitHandle irUnitPtr) {
  auto &state = getGlobalDebuggerState();
  if (!state.cursor) {
    llvm::errs() << "No active MLIR cursor, select from the context first\n";
    return;
  }
  IRUnit *unit = reinterpret_cast<IRUnit *>(irUnitPtr);
  if (auto *op = unit->dyn_cast<Operation *>()) {
    state.cursor = op->getBlock();
  } else if (auto *region = unit->dyn_cast<Region *>()) {

    state.cursor = region->getParentOp();
  } else if (auto *block = unit->dyn_cast<Block *>()) {
    state.cursor = block->getParent();
  } else {
    llvm::errs() << "Current cursor is not a valid IRUnit";
    return;
  }
  printIRUnit(state.cursor, llvm::errs());
}

void mlirDebuggerCursorSelectNextIRUnit(irunitHandle irUnitPtr) {
  auto &state = getGlobalDebuggerState();
  if (!state.cursor) {
    llvm::errs() << "No active MLIR cursor, select from the context first\n";
    return;
  }
  IRUnit *unit = reinterpret_cast<IRUnit *>(irUnitPtr);
  if (auto *op = unit->dyn_cast<Operation *>()) {
    state.cursor = op->getBlock();
  } else if (auto *region = unit->dyn_cast<Region *>()) {

    state.cursor = region->getParentOp();
  } else if (auto *block = unit->dyn_cast<Block *>()) {
    state.cursor = block->getParent();
  } else {
    llvm::errs() << "Current cursor is not a valid IRUnit";
    return;
  }
  printIRUnit(state.cursor, llvm::errs());
}

breakpointHandle mlirDebuggerAddTagBreakpoint(const char *tag) {
  DebuggerState &state = getGlobalDebuggerState();
  Breakpoint *breakpoint =
      state.tagBreakpointManager.addBreakpoint(StringRef(tag, strlen(tag)));
  int breakpointId = state.breakpointIdsMap.size() + 1;
  state.breakpointIdsMap[breakpointId] = breakpoint;
  return reinterpret_cast<breakpointHandle>(breakpoint);
}

void mlirDebuggerAddRewritePatternBreakpoint(const char *patternNameInfo) {}

void mlirDebuggerAddFileLineColLocBreakpoint(const char *file, unsigned line,
                                             unsigned col) {
  getGlobalDebuggerState().fileLineColLocBreakpointManager.addBreakpoint(
      StringRef(file, strlen(file)), line, col);
}

} // extern "C"

static void preventLinkerDeadCodeElim() {
  static void *volatile sink;
  static bool initialized = [&]() {
    sink = (void *)mlirDebuggerSetControl;
    sink = (void *)mlirDebuggerEnableBreakpoint;
    sink = (void *)mlirDebuggerDisableBreakpoint;
    sink = (void *)mlirDebuggerPrintContext;
    sink = (void *)mlirDebuggerPrintActionBacktrace;
    sink = (void *)mlirDebuggerCursorSelectIRUnitFromContext;
    sink = (void *)mlirDebuggerCursorSelectParentIRUnit;
    sink = (void *)mlirDebuggerCursorSelectChildIRUnit;
    sink = (void *)mlirDebuggerCursorSelectPreviousIRUnit;
    sink = (void *)mlirDebuggerCursorSelectNextIRUnit;
    sink = (void *)mlirDebuggerAddTagBreakpoint;
    sink = (void *)mlirDebuggerAddRewritePatternBreakpoint;
    sink = (void *)mlirDebuggerAddFileLineColLocBreakpoint;
    sink = (void *)&sink;
    return true;
  }();
  (void)initialized;
}

__attribute__((noinline)) void mlirGdbBreakpointHook() {}

static tracing::ExecutionContext::Control
gdbCallBackFunction(const tracing::ActionActiveStack *actionStack) {
  preventLinkerDeadCodeElim();
  auto &state = getGlobalDebuggerState();
  state.actionActiveStack = actionStack;
  state.cursor = nullptr;
  // If no breakpoint was hit, just continue from there.
  if (!actionStack->getBreakpoint())
    return ExecutionContext::Control::Apply;
  // Invoke the breakpoint hook, the debugger is supposed to trap this.
  // The debugger controls the execution from there by invoking
  // `mlirDebuggerSetControl()`.
  getGlobalDebuggerState().gdbControl = ExecutionContext::Apply;
  mlirGdbBreakpointHook();
  return getGlobalDebuggerState().gdbControl;
}

void mlir::setupGdbDebugExecutionContextHook(
    tracing::ExecutionContext &executionContext) {
  executionContext.setCallback(gdbCallBackFunction, /*passThrough=*/true);
  DebuggerState &state = getGlobalDebuggerState();
  executionContext.addBreakpointManager(&state.fileLineColLocBreakpointManager);
  executionContext.addBreakpointManager(&state.tagBreakpointManager);
}
