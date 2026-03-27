// Test that --mlir-print-ir-* options are respected by passes applied via
// transform.apply_registered_pass. See https://github.com/llvm/llvm-project/issues/171063

// RUN: mlir-opt %s --transform-interpreter --mlir-print-ir-after-all -o /dev/null 2>&1 | FileCheck %s --check-prefix=AFTER_ALL
// RUN: mlir-opt %s --transform-interpreter --mlir-print-ir-before-all -o /dev/null 2>&1 | FileCheck %s --check-prefix=BEFORE_ALL

func.func @test_func() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %sum = arith.addi %c1, %c2 : i32
  return %sum : i32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg
      : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "canonicalize" to %0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// Verify that the inner pass (CanonicalizerPass) emits an IR dump when
// --mlir-print-ir-after-all is passed.
// AFTER_ALL: IR Dump After CanonicalizerPass
// AFTER_ALL: func @test_func

// Verify that the inner pass (CanonicalizerPass) emits an IR dump when
// --mlir-print-ir-before-all is passed.
// BEFORE_ALL: IR Dump Before CanonicalizerPass
// BEFORE_ALL: func @test_func
