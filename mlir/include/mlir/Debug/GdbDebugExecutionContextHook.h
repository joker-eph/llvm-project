//===- GdbDebugExecutionContextHook.h - GDB for Debugger Support *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of C API functions that are used by the debugger to
// interact with the ExecutionContext.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_GDBDEBUGEXECUTIONCONTEXTHOOK_H
#define MLIR_SUPPORT_GDBDEBUGEXECUTIONCONTEXTHOOK_H

#include "mlir-c/IR.h"
#include "mlir/Debug/ExecutionContext.h"
#include "llvm/Support/Compiler.h"

extern "C" {
struct breakpoint;
struct irunit;
typedef struct breakpoint *breakpointHandle;
typedef struct irunit *irunitHandle;

/// This is used by the debugger to control what to do after a breakpoint is
/// hit. See tracing::ExecutionContext::Control for more information.
void mlirDebuggerSetControl(int controlOption);

/// Enable the provided breakpoint.
void mlirDebuggerEnableBreakpoint(breakpointHandle breakpoint);

/// Disable the provided breakpoint.
void mlirDebuggerDisableBreakpoint(breakpointHandle breakpoint);

/// Print the available context for the current Action.
void mlirDebuggerPrintContext();

/// Print the current action backtrace.
void mlirDebuggerPrintActionBacktrace(bool withContext);

/// Print the provided IR unit.
void mlirDebuggerPrintIRUnit(irunitHandle);

/// Return the IR unit from the current context by ID.
void mlirDebuggerCursorSelectIRUnitFromContext(int index);

/// Return the parent IR unit of the provided IR unit, or nullptr if the IR unit
/// has no parent.
void mlirDebuggerCursorSelectParentIRUnit(irunitHandle);

/// Return the child IR unit at the provided index, or nullptr if the index is
/// out of bound. For example if the irunit is an operation, the children IR
/// units will be the operation's regions.
void lirDebuggerCursorSelectChildIRUnit(irunitHandle, int index);

/// Return the next IR unit logically in the IR. For example if the irunit is a
/// Region the next IR unit will be the next region in the parent operation or
/// nullptr if there is no next region.
void mlirDebuggerCursorSelectPreviousIRUnit(irunitHandle);

/// Return the previous IR unit logically in the IR. For example if the irunit
/// is a Region, the previous IR unit will be the previous region in the parent
/// operation or nullptr if there is no previous region.
void mlirDebuggerCursorSelectNextIRUnit(irunitHandle);

/// Add a breakpoint matching exactly the provided tag.
breakpointHandle mlirDebuggerAddTagBreakpoint(const char *tag);

/// Add a breakpoint matching a pattern by name.
void mlirDebuggerAddRewritePatternBreakpoint(const char *patternNameInfo);

/// Add a breakpoint matching a file, line and column.
void mlirDebuggerAddFileLineColLocBreakpoint(const char *file, unsigned line,
                                             unsigned col);

} // extern "C"

namespace mlir {
// Setup the GDB hooks as a callback on the provided ExecutionContext.
void setupGdbDebugExecutionContextHook(
    tracing::ExecutionContext &executionContext);

} // namespace mlir

#endif // MLIR_SUPPORT_GDBDEBUGEXECUTIONCONTEXTHOOK_H
