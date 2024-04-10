//===-- TestBenchInterface.td - Interfaces Bench --*- tablegen -*-============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces that can be used to define information
// about control flow operations, e.g. branches.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_TESTBENCHINTERFACES
#define MLIR_INTERFACES_TESTBENCHINTERFACES

include "mlir/IR/OpBase.td"


def TestBenchInterface : OpInterface<"TestBenchInterface"> {
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      "Some method.",
      "int64_t", "getCost",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return reinterpret_cast<int64_t>(this); }]
    >
  ];
}

#endif // MLIR_INTERFACES_TESTBENCHINTERFACES