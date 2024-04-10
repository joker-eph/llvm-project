include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/RegionKindInterface.td"
include "TestBenchInterface.td"

def TestBench_Dialect : Dialect {
  let name = "testbench";
  let cppNamespace = "mlir";
}

class TESTBench_Op<string mnemonic, list<Trait> traits = []> :
    Op<TestBench_Dialect, mnemonic, traits>;

def EmptyOp : TESTBench_Op<"empty"> {
}

def TerminatorOp : TESTBench_Op<"terminator", [Terminator]> {
}

def OpWithRegion : TESTBench_Op<"with_region",
  [SingleBlock, RegionKindInterface, HasOnlyGraphRegion]> {
  let regions = (region SizedRegion<1>:$body);
}

// Interfaces with be externally registered
def OpWithInterface : TESTBench_Op<"op_with_interfaces",
  [TestBenchInterface]> {
}

def PassthroughOp : TESTBench_Op<"passthrough"> {
  let arguments = (ins AnyType:$in);
  let results = (outs AnyType:$out);
}

def OpWithAttr : TESTBench_Op<"with_attr"> {
  let arguments = (ins OptionalAttr<StrAttr>:$attr);
}

def SymbolOp : TESTBench_Op<"symbol"> {
  let arguments = (ins StrAttr:$sym_name);
}
