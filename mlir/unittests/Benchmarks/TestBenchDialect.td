include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def TestBench_Dialect : Dialect {
  let name = "testbench";
  let cppNamespace = "mlir";
}

class TESTBench_Op<string mnemonic, list<Trait> traits = []> :
    Op<TestBench_Dialect, mnemonic, traits>;

def EmptyOp : TESTBench_Op<"empty"> {
}
