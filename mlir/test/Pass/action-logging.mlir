// RUN: mlir-opt %s --log-actions-to=- -canonicalize -test-module-pass | FileCheck %s

// CHECK: [thread 0] begins (no breakpoint) Action `pass-execution-action` running "Canonicalizer" on Operation "builtin.module"
// CHECK: [thread 0] completed `pass-execution-action`
// CHECK: [thread 0] begins (no breakpoint) Action `pass-execution-action` running "(anonymous namespace)::TestModulePass" on Operation "builtin.module"
// CHECK: [thread 0] completed `pass-execution-action`
