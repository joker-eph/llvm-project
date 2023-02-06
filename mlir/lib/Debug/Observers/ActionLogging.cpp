//===- ExecutionContext.cpp - Debug Execution Context Support -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Debug/Observers/ActionLogging.h"
#include "mlir/Debug/BreakpointManager.h"
#include "mlir/IR/Action.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>
#include <thread>

using namespace mlir;
using namespace mlir::tracing;

//===----------------------------------------------------------------------===//
// ActionLogger
//===----------------------------------------------------------------------===//

static std::atomic<int64_t> thread_counter;
int64_t getThreadId() {
  thread_local int64_t tid = thread_counter++;
  return tid;
}

bool ActionLogger::shouldLog(const ActionActiveStack *action) {
  // If some condition was set, we ensured it is met before logging.
  if (breakpointManagers.empty())
    return true;
  return llvm::any_of(breakpointManagers,
                      [&](const BreakpointManager *manager) {
                        return manager->match(action->getAction());
                      });
}

void ActionLogger::beforeExecute(const ActionActiveStack *action,
                                 Breakpoint *breakpoint, bool willExecute) {
  if (!shouldLog(action))
    return;
  os << "[thread " << getThreadId() << "] ";
  if (willExecute)
    os << "begins ";
  else
    os << "skipping ";
  if (printBreakpoints) {
    if (breakpoint)
      os << "(on breakpoint: " << *breakpoint << ") ";
    else
      os << "(no breakpoint) ";
  }
  os << "Action ";
  if (printActions)
    action->getAction().print(os);
  else
    os << action->getAction().getTag();
  if (printIRUnits) {
    os << " (";
    interleaveComma(action->getAction().getContextIRUnits(), os,
                    [&](const IRUnit &unit) { printIRUnit(unit, os); });
    os << ")\n";
  }
}

void ActionLogger::afterExecute(const ActionActiveStack *action) {
  if (!shouldLog(action))
    return;
  os << "[thread " << getThreadId() << "] completed `"
     << action->getAction().getTag() << "`\n";
}
