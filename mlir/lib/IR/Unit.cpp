//===- Unit.cpp - Support for manipulating IR Unit ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Unit.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

using namespace mlir;

static void printOp(llvm::raw_ostream &os, Operation *op) {
  if (!op) {
    os << "<Operation:nullptr>";
    return;
  }
  op->print(os, OpPrintingFlags().skipRegions().useLocalScope());
}

static void printRegion(llvm::raw_ostream &os, Region *region) {
  if (!region) {
    os << "<Region:nullptr>";
    return;
  }
  os << "Region #" << region->getRegionNumber() << " for op ";
  printOp(os, region->getParentOp());
}

static void printBlock(llvm::raw_ostream &os, Block *block) {
  int blockId = 0;
  Block *cur = block;
  while ((cur = cur->getPrevNode()))
    blockId++;
  os << "Block #" << blockId << " for ";
  printRegion(os, block->getParent());
}

void mlir::printIRUnit(const IRUnit &unit, llvm::raw_ostream &os) {
  if (auto *op = unit.dyn_cast<Operation *>()) {
    printOp(os, op);
    return;
  }
  if (auto *region = unit.dyn_cast<Region *>()) {
    printRegion(os, region);
    return;
  }
  if (auto *block = unit.dyn_cast<Block *>()) {
    printBlock(os, block);
    return;
  }
  llvm_unreachable("unknown IRUnit");
}
