//===- RegionKindInterface.td - Region kind interfaces -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces to query the properties of regions
// in an operation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_REGIONKINDINTERFACE
#define MLIR_IR_REGIONKINDINTERFACE

include "mlir/IR/OpBase.td"

// OpInterface to query the properties of regions in an operation
def RegionKindInterface : OpInterface<"RegionKindInterface"> {
  let description = [{
    Interface for operations to describe the abstract semantics of
    their regions. Currently, two kinds of regions are
    supported. RegionKind::Graph represents a graph region without
    control flow semantics. RegionKind::SSACFG represents an
    [SSA-style control flow](../LangRef.md/#modeling-control-flow) region
    with basic blocks, sequential semantics, and reachability.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/[{
        Return the kind of the region with the given index inside this operation.
      }],
      /*retTy=*/"::mlir::RegionKind",
      /*methodName=*/"getRegionKind",
      /*args=*/(ins "unsigned":$index)
    >,
    StaticInterfaceMethod<
      /*desc=*/"Return true if the kind of the given region requires the "
               "SSA-Dominance property",
      /*retTy=*/"bool",
      /*methodName=*/"hasSSADominance",
      /*args=*/(ins "unsigned":$index),
      /*methodBody=*/[{
        return getRegionKind(index) == ::mlir::RegionKind::SSACFG;
      }]
    >,
  ];
}

def HasOnlyGraphRegion : NativeOpTrait<"HasOnlyGraphRegion">;

//===----------------------------------------------------------------------===//
// Breaking Control Flow Trait and Interface
//===----------------------------------------------------------------------===//

/// Trait for ops that transparently pass breaking control flow upward without
/// consuming it. An intermediate op in a break chain must have this trait.
def PropagateControlFlowBreak : NativeOpTrait<"PropagateControlFlowBreak">;

/// Interface for terminator ops that can break out of more than one nested
/// region level at once (e.g., `scf.break 2` exits through an intermediate
/// `scf.if` to reach the enclosing `scf.loop`). The default implementation
/// returns 1, meaning the terminator exits only the immediately enclosing
/// region (normal behaviour). Ops in the IR layer can query this interface
/// without depending on ControlFlowInterfaces.
def BreakingTerminatorOpInterface
    : OpInterface<"BreakingTerminatorOpInterface"> {
  let description = [{
    Implemented by terminator operations that may exit more than one nested
    region level.  A value of 1 (the default) exits only the immediately
    enclosing region.  A value of K exits K levels: the K-1 intermediate
    enclosing ops must carry `PropagateControlFlowBreak`, and the op at
    level K must implement `HasBreakingControlFlowOpInterface`.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the number of nested region levels this terminator breaks out
        of.  1 means exit the immediately enclosing region; K means exit K
        levels.
      }],
      "unsigned", "getNumBreakingRegions", (ins), "",
      /*defaultImplementation=*/ "return 1;"
    >,
  ];
}

/// Interface for ops that can be targeted by breaking control flow terminators
/// (e.g., scf.break, scf.continue). Ops implementing this interface can
/// receive break/continue operations that break out of nested regions.
def HasBreakingControlFlowOpInterface
    : OpInterface<"HasBreakingControlFlowOpInterface"> {
  let description = [{
    This interface is implemented by operations that can be targeted by
    breaking control flow (e.g., loop operations that accept `scf.break`
    and `scf.continue` terminators jumping out of nested regions).

    An operation implementing this interface can appear as the final target
    in a break chain: intermediate ops (those at levels 1..K-1 for a K-level
    break) must carry `PropagateControlFlowBreak`, while the target op at
    level K must implement `HasBreakingControlFlowOpInterface`.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns true if the given terminator operation can target this op to
        perform a break or continue. By default, all terminators are accepted.
      }],
      "bool", "acceptsTerminator",
      (ins "::mlir::Operation *":$op), "",
      /*defaultImplementation=*/ "return true;"
    >,
    InterfaceMethod<[{
        Returns true if this op has nested terminator operations that may
        break to or continue at this op. Used for analysis purposes.
      }],
      "bool", "hasNestedPredecessors", (ins), "",
      /*defaultImplementation=*/ "return false;"
    >,
  ];
}

// Op's regions that don't need a terminator: requires some other traits
// so it defines a list that must be concatenated.
def GraphRegionNoTerminator : TraitList<[
    NoTerminator,
    SingleBlock,
    RegionKindInterface,
    HasOnlyGraphRegion
  ]>;

#endif // MLIR_IR_REGIONKINDINTERFACE
