// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
// Break count must be >= 1.
func.func @break_zero() {
  scf.loop {
    // expected-error@+1 {{'scf.break' op breaking region count must be at least 1}}
    scf.break 0
  }
  return
}

// -----
// Continue count must be >= 1.
func.func @continue_zero() {
  scf.loop {
    // expected-error@+1 {{'scf.continue' op breaking region count must be at least 1}}
    scf.continue 0
  }
  return
}

// -----
// scf.break must have a parent IfOp or LoopOp (ParentOneOf check).
func.func @break_invalid_parent() {
  // expected-error@+1 {{'scf.break' op expects parent op to be one of 'scf.if, scf.loop'}}
  scf.break 1
  return
}

// -----
// scf.break 2 inside an scf.if directly in func.func: scf.if has
// PropagateControlFlowBreak, but func.func does not implement
// HasBreakingControlFlowOpInterface, so the chain is broken.
// expected-error@+1 {{operation has a nested predecessor but does not have the HasBreakingControlFlowOpInterface trait.}}
func.func @break_chain_no_loop(%cond: i1) {
  scf.if %cond {
    // expected-note@+1 {{ for this predecessor operation (scf.break)}}
    scf.break 2
  }
  return
}
