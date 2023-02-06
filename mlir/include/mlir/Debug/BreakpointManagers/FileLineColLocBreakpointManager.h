//===- FileLineColLocBreakpointManager.h - TODO: add message ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO: Write a proper description for the service
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRACING_BREAKPOINTMANAGERS_FILELINECOLLOCBREAKPOINTMANAGER_H
#define MLIR_TRACING_BREAKPOINTMANAGERS_FILELINECOLLOCBREAKPOINTMANAGER_H

#include "mlir/Debug/BreakpointManager.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/IR/Action.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace mlir {
namespace tracing {

class FileLineColLocBreakpoint : public Breakpoint {
public:
  FileLineColLocBreakpoint()
      : Breakpoint(TypeID::get<FileLineColLocBreakpoint>()) {}

  FileLineColLocBreakpoint(StringRef file, int64_t line, int64_t col)
      : Breakpoint(TypeID::get<FileLineColLocBreakpoint>()), file(file),
        line(line), col(col) {}

  /// Provide classof to allow casting between breakpoint types.
  static bool classof(const Breakpoint *breakpoint) {
    return breakpoint->getTypeID() == TypeID::get<FileLineColLocBreakpoint>();
  }

  void print(raw_ostream &os) const override {
    os << "Location: " << file << ':' << line << ':' << col;
  }

  /// Parse a string representation in the form of "<file>:<line>:<col>". Return
  /// a tuple with these three elements, the first one is a StringRef pointing
  /// into the original string.
  static FailureOr<std::tuple<StringRef, int64_t, int64_t>> parseFromString(
      StringRef str,
      llvm::function_ref<void(StringRef)> diag = [](StringRef) {});

private:
  /// A filename on which to break.
  StringRef file;

  /// A particular line on which to break, or -1 to break on any line.
  int64_t line;

  /// A particular column on which to break, or -1 to break on any column
  int64_t col;

  friend class FileLineColLocBreakpointManager;
};

class FileLineColLocBreakpointManager : public BreakpointManager {
public:
  FileLineColLocBreakpointManager()
      : BreakpointManager(TypeID::get<FileLineColLocBreakpointManager>()) {}

  /// Provide classof to allow casting between breakpoint manager types.
  static bool classof(const BreakpointManager *breakpointManager) {
    return breakpointManager->getTypeID() ==
           TypeID::get<FileLineColLocBreakpointManager>();
  }

  Breakpoint *match(const Action &action) const override {
    for (const IRUnit &unit : action.getContextIRUnits()) {
      if (auto *op = unit.dyn_cast<Operation *>()) {
        if (auto match = matchFromLocation(op->getLoc())) {
          return *match;
        }
        continue;
      }
      if (auto *block = unit.dyn_cast<Block *>()) {
        for (auto &op : block->getOperations()) {
          if (auto match = matchFromLocation(op.getLoc())) {
            return *match;
          }
        }
        continue;
      }
      if (Region *region = unit.dyn_cast<Region *>()) {
        if (auto match = matchFromLocation(region->getLoc())) {
          return *match;
        }
        continue;
      }
    }
    return {};
  }

  FileLineColLocBreakpoint *addBreakpoint(StringRef file, int64_t line,
                                          int64_t col = -1) {
    auto &breakpoint = breakpoints[file][line][col];
    if (!breakpoint)
      breakpoint = std::make_unique<FileLineColLocBreakpoint>(file, line, col);
    return breakpoint.get();
  }

  /// A map from a filename -> line -> column -> breakpoint.
  DenseMap<
      StringRef,
      DenseMap<int64_t,
               DenseMap<int64_t, std::unique_ptr<FileLineColLocBreakpoint>>>>
      breakpoints;

private:
  llvm::Optional<Breakpoint *> matchFromLocation(Location loc) const {
    auto fileLoc = loc.dyn_cast<FileLineColLoc>();
    if (!fileLoc)
      return {};
    auto fileLookup = breakpoints.find(fileLoc.getFilename());
    if (fileLookup == breakpoints.end())
      return {};

    auto lineLookup = fileLookup->second.find(fileLoc.getLine());
    // If not found, check with the -1 key if we have a breakpoint for any line.
    if (lineLookup == fileLookup->second.end())
      lineLookup = fileLookup->second.find(-1);
    if (lineLookup == fileLookup->second.end())
      return {};

    auto colLookup = lineLookup->second.find(fileLoc.getColumn());
    // If not found, check with the -1 key if we have a breakpoint for any col.
    if (colLookup == lineLookup->second.end())
      colLookup = lineLookup->second.find(-1);
    if (colLookup == lineLookup->second.end())
      return {};
    if (colLookup->second.get()->isEnabled())
      return colLookup->second.get();
    return {};
  }
};

} // namespace tracing
} // namespace mlir

#endif // MLIR_TRACING_BREAKPOINTMANAGERS_FILELINECOLLOCBREAKPOINTMANAGER_H
