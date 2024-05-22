// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{conservative=true}))' | FileCheck %s  --check-prefixes=CHECK-CONSERVATIVE

// CHECK-LABEL: func @remove_op_with_inner_ops_pattern
func.func @remove_op_with_inner_ops_pattern() {
  // CHECK-NEXT: return
  // CHECK-CONSERVATIVE: test.op_with_region_pattern
  "test.op_with_region_pattern"() ({
    "test.op_with_region_terminator"() : () -> ()
  }) : () -> ()
  return
}
