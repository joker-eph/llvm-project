// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @simple_loop
func.func @simple_loop() {
  // CHECK: scf.loop {
  scf.loop {
    // CHECK: scf.break 1
    scf.break 1
  }
  return
}

// CHECK-LABEL: func @loop_with_implicit_yield
func.func @loop_with_implicit_yield() {
  // CHECK: scf.loop {
  // CHECK-NOT: scf.yield
  scf.loop {
    scf.yield
  }
  return
}

// CHECK-LABEL: func @continue_in_loop
func.func @continue_in_loop() {
  scf.loop {
    // CHECK: scf.continue 1
    scf.continue 1
  }
  return
}

// CHECK-LABEL: func @break_through_if
func.func @break_through_if(%cond: i1) {
  scf.loop {
    scf.if %cond {
      // CHECK: scf.break 2
      scf.break 2
    }
    scf.yield
  }
  return
}

// CHECK-LABEL: func @continue_through_if
func.func @continue_through_if(%cond: i1) {
  scf.loop {
    scf.if %cond {
      // CHECK: scf.continue 2
      scf.continue 2
    }
    scf.yield
  }
  return
}

// CHECK-LABEL: func @nested_loops
func.func @nested_loops(%cond: i1) {
  scf.loop {
    scf.loop {
      scf.if %cond {
        // CHECK: scf.break 2
        scf.break 2
      }
      scf.yield
    }
    scf.yield
  }
  return
}
