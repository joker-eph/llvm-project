//===- Unit.h -  IR Unit definition--------------------*- C++ -*-=============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_UNIT_H
#define MLIR_IR_UNIT_H

#include "llvm/ADT/PointerUnion.h"

namespace llvm {
class raw_ostream;
}
namespace mlir {
class Operation;
class Region;
class Block;
class Value;

// IRUnit is a union of the different types of IR objects that can be
// manipulated. This is used to associate some context to an action.
using IRUnit = llvm::PointerUnion<Operation *, Region *, Block *, Value>;

// Print the IRUnit to the given stream.
void printIRUnit(const IRUnit &unit, llvm::raw_ostream &os);

} // end namespace mlir

#endif // MLIR_IR_UNIT_H