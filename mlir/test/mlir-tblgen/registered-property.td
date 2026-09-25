// RUN: mlir-tblgen -gen-registered-property-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-registered-property-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "mlir/IR/Properties.td"

def PlainProperty : Property<"int64_t">;

def CountKind : PropDef<"int64_t", "count", "test", "CountKind"> {
  let verifier = [{
    return ::mlir::success(value >= 0);
  }];
  let parser = [{ return parser.parseInteger(value); }];
  let printer = [{ printer.printInteger(value); }];
}
def CountProperty : RegisteredProp<CountKind>;

def LabelKind : PropDef<"std::string", "label", "test", "LabelKind"> {
  let parser = [{ return parser.parseString(&value); }];
  let printer = [{ printer.printString(value); }];
}
def LabelProperty : RegisteredProp<LabelKind>;

def ArrayKind : PropDef<"std::vector<int64_t>", "array", "test", "ArrayKind"> {
  let hasher = [{
    return ::llvm::hash_combine_range(value.begin(), value.end());
  }];
  let parser = [{
    if (parser.parseLSquare()) return ::mlir::failure();
    if (parser.parseOptionalRSquare()) {
      do {
        int64_t element;
        if (parser.parseInteger(element)) return ::mlir::failure();
        value.push_back(element);
      } while (::mlir::succeeded(parser.parseOptionalComma()));
      return parser.parseRSquare();
    }
    return ::mlir::success();
  }];
  let printer = [{
    printer << '[';
    ::llvm::interleaveComma(value, printer.getStream());
    printer << ']';
  }];
}
def ArrayProperty : RegisteredProp<ArrayKind>;

// DECL-NOT: PlainProperty
// DECL: namespace test {
// DECL: struct ArrayKind {
// DECL: using StorageType = std::vector<int64_t>;
// DECL: static ::llvm::hash_code hash(const StorageType &value);
// DECL: static ::mlir::ParseResult parse(::mlir::AsmParser &parser, StorageType &value);
// DECL: static void print(::mlir::AsmPrinter &printer, const StorageType &value);
// DECL: struct CountKind {
// DECL: using StorageType = int64_t;
// DECL: static constexpr llvm::StringLiteral name = "count";
// DECL: static ::mlir::LogicalResult verify(const StorageType &value);
// DECL: struct LabelKind {
// DECL: using StorageType = std::string;
// DECL: static constexpr llvm::StringLiteral name = "label";

// DEF-NOT: PlainProperty
// DEF: ::mlir::LogicalResult test::ArrayKind::verify(const StorageType &value) {
// DEF: return ::mlir::success();
// DEF: ::llvm::hash_code test::ArrayKind::hash(const StorageType &value) {
// DEF: return ::llvm::hash_combine_range(value.begin(), value.end());
// DEF: ::mlir::ParseResult test::ArrayKind::parse(::mlir::AsmParser &parser, StorageType &value) {
// DEF: void test::ArrayKind::print(::mlir::AsmPrinter &printer, const StorageType &value) {
// DEF: ::mlir::LogicalResult test::CountKind::verify(const StorageType &value) {
// DEF: return ::mlir::success(value >= 0);
// DEF: ::mlir::LogicalResult test::LabelKind::verify(const StorageType &value) {
// DEF: (void)value;
// DEF: return ::mlir::success();
