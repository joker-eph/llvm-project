include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/RegionKindInterface.td"

def TestBench_Dialect : Dialect {
  let name = "testbench";
  let cppNamespace = "mlir";
}

class TESTBench_Op<string mnemonic, list<Trait> traits = []> :
    Op<TestBench_Dialect, mnemonic, traits>;

def EmptyOp : TESTBench_Op<"empty"> {
}

def OpWithRegion : TESTBench_Op<"with_region",
  [SingleBlock, RegionKindInterface, HasOnlyGraphRegion]> {
  let regions = (region SizedRegion<1>:$body);
}

// Interfaces with be externally registered
def OpWithManyInterfaces : TESTBench_Op<"op_with_interfaces"> {
}
