// Run the canonicalize on each function, use the --log-mlir-actions-filter= option
// to filter which action should be logged.

func.func @a() {
    return
}

func.func @b() {
    return
}

func.func @c() {
    return
}

////////////////////////////////////
/// 1. All actions should be logged.

// RUN: mlir-opt %s --log-actions-to=- -pass-pipeline="builtin.module(func.func(canonicalize))" -o %t --mlir-disable-threading | FileCheck %s
// Specify the current file as filter, expect to see all actions.
// RUN: mlir-opt %s --log-mlir-actions-filter=%s --log-actions-to=- -pass-pipeline="builtin.module(func.func(canonicalize))" -o %t --mlir-disable-threading | FileCheck %s

// CHECK: [thread 0] begins (no breakpoint) Action `pass-execution-action`  running "Canonicalizer" on Operation "func.func" (func.func @a() {/*skip region4*/})
// CHECK-NEXT: [thread 0] completed `pass-execution-action`
// CHECK-NEXT: [thread 0] begins (no breakpoint) Action `pass-execution-action`  running "Canonicalizer" on Operation "func.func" (func.func @b() {/*skip region4*/})
// CHECK-NEXT: [thread 0] completed `pass-execution-action`
// CHECK-NEXT: [thread 0] begins (no breakpoint) Action `pass-execution-action`  running "Canonicalizer" on Operation "func.func" (func.func @c() {/*skip region4*/})
// CHECK-NEXT: [thread 0] completed `pass-execution-action`

////////////////////////////////////
/// 2. No match

// Specify a non-existing file as filter, expect to see no actions.
// RUN: mlir-opt %s --log-mlir-actions-filter=foo.mlir --log-actions-to=- -pass-pipeline="builtin.module(func.func(canonicalize))" -o %t --mlir-disable-threading | FileCheck %s --check-prefix=CHECK-NONE --allow-empty
// Filter on a non-matching line, expect to see no actions.
// RUN: mlir-opt %s --log-mlir-actions-filter=%s:1 --log-actions-to=- -pass-pipeline="builtin.module(func.func(canonicalize))" -o %t --mlir-disable-threading | FileCheck %s --check-prefix=CHECK-NONE --allow-empty

// Invalid Filter
// CHECK-NONE-NOT: Canonicalizer

////////////////////////////////////
/// 3. Matching filters

// Filter the second function only
// RUN: mlir-opt %s --log-mlir-actions-filter=%s:8 --log-actions-to=- -pass-pipeline="builtin.module(func.func(canonicalize))" -o %t --mlir-disable-threading | FileCheck %s --check-prefix=CHECK-SECOND

// CHECK-SECOND-NOT: @a
// CHECK-SECOND-NOT: @c
// CHECK-SECOND: [thread 0] begins (no breakpoint) Action `pass-execution-action`  running "Canonicalizer" on Operation "func.func" (func.func @b() {/*skip region4*/})
// CHECK-SECOND-NEXT: [thread 0] completed `pass-execution-action`

// Filter the first and third functions
// RUN: mlir-opt %s --log-mlir-actions-filter=%s:4,%s:12 --log-actions-to=- -pass-pipeline="builtin.module(func.func(canonicalize))" -o %t --mlir-disable-threading | FileCheck %s  --check-prefix=CHECK-FIRST-THIRD

// CHECK-FIRST-THIRD-NOT: Canonicalizer
// CHECK-FIRST-THIRD: [thread 0] begins (no breakpoint) Action `pass-execution-action`  running "Canonicalizer" on Operation "func.func" (func.func @a() {/*skip region4*/})
// CHECK-FIRST-THIRD-NEXT: [thread 0] completed `pass-execution-action`
// CHECK-FIRST-THIRD-NEXT: [thread 0] begins (no breakpoint) Action `pass-execution-action`  running "Canonicalizer" on Operation "func.func" (func.func @c() {/*skip region4*/})
// CHECK-FIRST-THIRD-NEXT: [thread 0] completed `pass-execution-action`
// CHECK-FIRST-THIRD-NOT: Canonicalizer
