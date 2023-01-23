//===- TagBreakpointManager.h - Simple breakpoint Support -------*- C++ -*-===//
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

#ifndef MLIR_TRACING_BREAKPOINTMANAGERS_TAGBREAKPOINTMANAGER_H
#define MLIR_TRACING_BREAKPOINTMANAGERS_TAGBREAKPOINTMANAGER_H

#include "mlir/Debug/BreakpointManager.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/IR/Action.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
namespace tracing {

/// Simple breakpoint matching an action "tag".
class TagBreakpoint : public Breakpoint {
public:
  TagBreakpoint() : Breakpoint(TypeID::get<TagBreakpoint>()) {}

  TagBreakpoint(const std::string &_tag)
      : Breakpoint(TypeID::get<TagBreakpoint>()), tag(_tag) {}

  /// Provide classof to allow casting between breakpoint types.
  static bool classof(const Breakpoint *breakpoint) {
    return breakpoint->getTypeID() == TypeID::get<TagBreakpoint>();
  }

  void print(raw_ostream &os) const override { os << "Tag: `" << tag << '`'; }

private:
  /// A tag to associate the TagBreakpoint with.
  std::string tag;

  /// Allow access to `tag`.
  friend class TagBreakpointManager;
};

class TagBreakpointManager
    : public BreakpointManagerImpl<TagBreakpointManager> {
public:
  Breakpoint *match(const Action &action) const override {
    auto it = breakpoints.find(action.getTag());
    if (it != breakpoints.end() && it->second->isEnabled()) {
      return it->second.get();
    }
    return {};
  }

  /// Add a breakpoint to the manager for the given tag and return it.
  /// If a breakpoint already exists for the given tag, return the existing
  /// instance.
  TagBreakpoint *addBreakpoint(StringRef tag) {
    auto result = breakpoints.insert({tag, nullptr});
    auto &it = result.first;
    if (result.second)
      it->second = std::make_unique<TagBreakpoint>(tag.str());
    return it->second.get();
  }

private:
  llvm::StringMap<std::unique_ptr<TagBreakpoint>> breakpoints;
};

} // namespace tracing
} // namespace mlir

#endif // MLIR_TRACING_BREAKPOINTMANAGERS_TAGBREAKPOINTMANAGER_H
