//===- TestOpProperties.cpp - Test all properties-related APIs ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"
#include <optional>

using namespace mlir;

namespace {
/// Simple structure definining a struct to define "properties" for a given
/// operation. Default values are honored when creating an operation.
struct TestProperties {
  int a = -1;
  float b = -1.;
  std::vector<int64_t> array = {-33};
  /// A shared_ptr to a const object is safe: it is equivalent to a value-based
  /// member. Here the label will be deallocated when the last operation
  /// referring to it is destroyed. However there is no pool-allocation: this is
  /// offloaded to the client.
  std::shared_ptr<const std::string> label;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestProperties)
};

struct TestI32Kind {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestI32Kind)
  using StorageType = int;
  static constexpr llvm::StringLiteral name = "i32";
  static LogicalResult verify(int) { return success(); }
  static ParseResult parse(AsmParser &parser, int &value) {
    return parser.parseInteger(value);
  }
  static void print(AsmPrinter &printer, int value) {
    printer.printInteger(value);
  }
};

struct TestArrayKind {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestArrayKind)
  using StorageType = std::vector<int64_t>;
  static constexpr llvm::StringLiteral name = "array";
  static LogicalResult verify(const StorageType &) { return success(); }
  static llvm::hash_code hash(const StorageType &value) {
    return llvm::hash_combine_range(value.begin(), value.end());
  }
  static ParseResult parse(AsmParser &parser, StorageType &value) {
    if (parser.parseLSquare())
      return failure();
    if (succeeded(parser.parseOptionalRSquare()))
      return success();
    do {
      int64_t element;
      if (parser.parseInteger(element))
        return failure();
      value.push_back(element);
    } while (succeeded(parser.parseOptionalComma()));
    return parser.parseRSquare();
  }
  static void print(AsmPrinter &printer, const StorageType &value) {
    printer << '[';
    llvm::interleaveComma(value, printer.getStream());
    printer << ']';
  }
};

bool operator==(const TestProperties &lhs, TestProperties &rhs) {
  return lhs.a == rhs.a && lhs.b == rhs.b && lhs.array == rhs.array &&
         lhs.label == rhs.label;
}

/// Convert a DictionaryAttr to a TestProperties struct, optionally emit errors
/// through the provided diagnostic if any. This is used for example during
/// parsing with the generic format.
static LogicalResult
setPropertiesFromAttribute(TestProperties &prop, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  DictionaryAttr dict = dyn_cast<DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set TestProperties";
    return failure();
  }
  auto aAttr = dict.getAs<IntegerAttr>("a");
  if (!aAttr) {
    emitError() << "expected IntegerAttr for key `a`";
    return failure();
  }
  auto bAttr = dict.getAs<FloatAttr>("b");
  if (!bAttr ||
      &bAttr.getValue().getSemantics() != &llvm::APFloatBase::IEEEsingle()) {
    emitError() << "expected FloatAttr for key `b`";
    return failure();
  }

  auto arrayAttr = dict.getAs<DenseI64ArrayAttr>("array");
  if (!arrayAttr) {
    emitError() << "expected DenseI64ArrayAttr for key `array`";
    return failure();
  }

  auto label = dict.getAs<mlir::StringAttr>("label");
  if (!label) {
    emitError() << "expected StringAttr for key `label`";
    return failure();
  }

  prop.a = aAttr.getValue().getSExtValue();
  prop.b = bAttr.getValue().convertToFloat();
  prop.array.assign(arrayAttr.asArrayRef().begin(),
                    arrayAttr.asArrayRef().end());
  prop.label = std::make_shared<std::string>(label.getValue());
  return success();
}

/// Convert a TestProperties struct to a DictionaryAttr, this is used for
/// example during printing with the generic format.
static Attribute getPropertiesAsAttribute(MLIRContext *ctx,
                                          const TestProperties &prop) {
  SmallVector<NamedAttribute> attrs;
  Builder b{ctx};
  attrs.push_back(b.getNamedAttr("a", b.getI32IntegerAttr(prop.a)));
  attrs.push_back(b.getNamedAttr("b", b.getF32FloatAttr(prop.b)));
  attrs.push_back(b.getNamedAttr("array", b.getDenseI64ArrayAttr(prop.array)));
  attrs.push_back(b.getNamedAttr(
      "label", b.getStringAttr(prop.label ? *prop.label : "<nullptr>")));
  return b.getDictionaryAttr(attrs);
}

inline llvm::hash_code computeHash(const TestProperties &prop) {
  // We hash `b` which is a float using its underlying array of char:
  unsigned char const *p = reinterpret_cast<unsigned char const *>(&prop.b);
  ArrayRef<unsigned char> bBytes{p, sizeof(prop.b)};
  return llvm::hash_combine(prop.a, llvm::hash_combine_range(bBytes),
                            llvm::hash_combine_range(prop.array),
                            StringRef(*prop.label));
}

/// A custom operation for the purpose of showcasing how to use "properties".
class OpWithProperties : public Op<OpWithProperties> {
public:
  // Begin boilerplate
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpWithProperties)
  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  static StringRef getOperationName() {
    return "test_op_properties.op_with_properties";
  }
  // End boilerplate

  // This alias is the only definition needed for enabling "properties" for this
  // operation.
  using Properties = TestProperties;
  static ArrayRef<PropertyFieldDescriptor> getPropertyFieldDescriptors() {
    static const PropertyFieldDescriptor fields[] = {
        {"a", TypeID::get<TestI32Kind>(),
         [](Operation *op) {
           const AbstractProperty &kind = *AbstractProperty::lookup(
               TypeID::get<TestI32Kind>(), op->getContext());
           return Property(kind, &cast<OpWithProperties>(op).getProperties().a);
         },
         [](Operation *op, Property value) {
           cast<OpWithProperties>(op).getProperties().a =
               *value.get<TestI32Kind>();
         },
         nullptr,
         [](Operation *op) {
           cast<OpWithProperties>(op).getProperties().a = -1;
           return success();
         },
         [](OperationState &state, Property value) {
           state.getOrAddProperties<TestProperties>().a =
               *value.get<TestI32Kind>();
         },
         nullptr},
        {"array", TypeID::get<TestArrayKind>(),
         [](Operation *op) {
           const AbstractProperty &kind = *AbstractProperty::lookup(
               TypeID::get<TestArrayKind>(), op->getContext());
           return Property(kind,
                           &cast<OpWithProperties>(op).getProperties().array);
         },
         [](Operation *op, Property value) {
           cast<OpWithProperties>(op).getProperties().array =
               *value.get<TestArrayKind>();
         },
         [](Operation *, Property value) {
           return success(!value.get<TestArrayKind>()->empty());
         },
         nullptr,
         [](OperationState &state, Property value) {
           state.getOrAddProperties<TestProperties>().array =
               *value.get<TestArrayKind>();
         },
         [](Property value) {
           return success(!value.get<TestArrayKind>()->empty());
         }}};
    return fields;
  }
  static std::optional<mlir::Attribute> getInherentAttr(MLIRContext *context,
                                                        const Properties &prop,
                                                        StringRef name) {
    return std::nullopt;
  }
  static void setInherentAttr(Properties &prop, StringRef name,
                              mlir::Attribute value) {}
  static void walkInherentAttrs(MLIRContext *context, Properties &prop,
                                OperationName::InherentAttrVisitor visitor) {}
  static LogicalResult
  verifyInherentAttrs(OperationName opName, NamedAttrList &attrs,
                      function_ref<InFlightDiagnostic()> emitError) {
    return success();
  }
};

/// A custom operation for the purpose of showcasing how discardable attributes
/// are handled in absence of properties.
class OpWithoutProperties : public Op<OpWithoutProperties> {
public:
  // Begin boilerplate.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpWithoutProperties)
  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() {
    static StringRef attributeNames[] = {StringRef("inherent_attr")};
    return ArrayRef(attributeNames);
  };
  static StringRef getOperationName() {
    return "test_op_properties.op_without_properties";
  }
  // End boilerplate.
};

struct NativeOnlyProperties {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeOnlyProperties)
  int count = 0;
  bool operator==(const NativeOnlyProperties &rhs) const {
    return count == rhs.count;
  }
};

static llvm::hash_code computeHash(const NativeOnlyProperties &value) {
  return llvm::hash_value(value.count);
}

/// An operation whose native property deliberately has no attribute encoding.
class NativeOnlyOp : public Op<NativeOnlyOp, BytecodeOpInterface::Trait> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeOnlyOp)
  using Op::Op;
  using Properties = NativeOnlyProperties;
  static StringRef getOperationName() {
    return "test_op_properties.native_only";
  }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  static LogicalResult readProperties(DialectBytecodeReader &reader,
                                      OperationState &state) {
    uint64_t count;
    if (failed(reader.readVarInt(count)) || count > INT_MAX)
      return failure();
    state.getOrAddProperties<Properties>().count = count;
    return success();
  }
  void writeProperties(DialectBytecodeWriter &writer) {
    writer.writeVarInt(getProperties().count);
  }
  static ArrayRef<PropertyFieldDescriptor> getPropertyFieldDescriptors() {
    static const PropertyFieldDescriptor fields[] = {
        {"count", TypeID::get<TestI32Kind>(),
         [](Operation *op) {
           const AbstractProperty &kind = *AbstractProperty::lookup(
               TypeID::get<TestI32Kind>(), op->getContext());
           return Property(kind, &cast<NativeOnlyOp>(op).getProperties().count);
         },
         [](Operation *op, Property value) {
           cast<NativeOnlyOp>(op).getProperties().count =
               *value.get<TestI32Kind>();
         },
         nullptr, nullptr,
         [](OperationState &state, Property value) {
           state.getOrAddProperties<NativeOnlyProperties>().count =
               *value.get<TestI32Kind>();
         },
         nullptr, true}};
    return fields;
  }
  static LogicalResult
  setPropertiesFromAttr(Properties &, Attribute,
                        function_ref<InFlightDiagnostic()> emitError) {
    emitError() << "native_only has no legacy attribute conversion";
    return failure();
  }
  static Attribute getPropertiesAsAttr(MLIRContext *, const Properties &) {
    return {};
  }
  static std::optional<Attribute>
  getInherentAttr(MLIRContext *, const Properties &, StringRef) {
    return std::nullopt;
  }
  static void setInherentAttr(Properties &, StringRef, Attribute) {}
  static void walkInherentAttrs(MLIRContext *, Properties &,
                                OperationName::InherentAttrVisitor) {}
  static LogicalResult verifyInherentAttrs(OperationName, NamedAttrList &,
                                           function_ref<InFlightDiagnostic()>) {
    return success();
  }
};

// A trivial supporting dialect to register the above operation.
class TestOpPropertiesDialect : public Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOpPropertiesDialect)
  static constexpr StringLiteral getDialectNamespace() {
    return StringLiteral("test_op_properties");
  }
  explicit TestOpPropertiesDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<TestOpPropertiesDialect>()) {
    addProperties<TestI32Kind, TestArrayKind>();
    addOperations<OpWithProperties, OpWithoutProperties, NativeOnlyOp>();
  }
};

constexpr StringLiteral mlirSrc = R"mlir(
    "test_op_properties.op_with_properties"()
      <{a = -42 : i32,
        b = -4.200000e+01 : f32,
        array = array<i64: 40, 41>,
        label = "bar foo"}> : () -> ()
)mlir";

TEST(OpPropertiesTest, Properties) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  ParserConfig config(&context);
  // Parse the operation with some properties.
  OwningOpRef<Operation *> op = parseSourceString(mlirSrc, config);
  ASSERT_TRUE(op.get() != nullptr);
  auto opWithProp = dyn_cast<OpWithProperties>(op.get());
  ASSERT_TRUE(opWithProp);
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    opWithProp.print(os);
    ASSERT_STREQ("\"test_op_properties.op_with_properties\"() "
                 "<{a = -42 : i32, "
                 "array = array<i64: 40, 41>, "
                 "b = -4.200000e+01 : f32, "
                 "label = \"bar foo\"}> : () -> ()\n",
                 output.c_str());
  }
  // Get a mutable reference to the properties for this operation and modify it
  // in place one member at a time.
  TestProperties &prop = opWithProp.getProperties();
  prop.a = 42;
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    opWithProp.print(os);
    StringRef view(output);
    EXPECT_TRUE(view.contains("a = 42"));
    EXPECT_TRUE(view.contains("b = -4.200000e+01"));
    EXPECT_TRUE(view.contains("array = array<i64: 40, 41>"));
    EXPECT_TRUE(view.contains("label = \"bar foo\""));
  }
  prop.b = 42.;
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    opWithProp.print(os);
    StringRef view(output);
    EXPECT_TRUE(view.contains("a = 42"));
    EXPECT_TRUE(view.contains("b = 4.200000e+01"));
    EXPECT_TRUE(view.contains("array = array<i64: 40, 41>"));
    EXPECT_TRUE(view.contains("label = \"bar foo\""));
  }
  prop.array.push_back(42);
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    opWithProp.print(os);
    StringRef view(output);
    EXPECT_TRUE(view.contains("a = 42"));
    EXPECT_TRUE(view.contains("b = 4.200000e+01"));
    EXPECT_TRUE(view.contains("array = array<i64: 40, 41, 42>"));
    EXPECT_TRUE(view.contains("label = \"bar foo\""));
  }
  prop.label = std::make_shared<std::string>("foo bar");
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    opWithProp.print(os);
    StringRef view(output);
    EXPECT_TRUE(view.contains("a = 42"));
    EXPECT_TRUE(view.contains("b = 4.200000e+01"));
    EXPECT_TRUE(view.contains("array = array<i64: 40, 41, 42>"));
    EXPECT_TRUE(view.contains("label = \"foo bar\""));
  }
}

TEST(OpPropertiesTest, RegisteredFieldReferences) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  ParserConfig config(&context);
  OwningOpRef<Operation *> op = parseSourceString(mlirSrc, config);
  ASSERT_TRUE(op);
  EXPECT_EQ(op->getPropertyFieldDescriptors().size(), 2u);
  EXPECT_FALSE(op->getPropertyField("missing"));

  auto a = op->getPropertyField("a");
  auto array = op->getPropertyField("array");
  ASSERT_TRUE(a);
  ASSERT_TRUE(array);
  EXPECT_EQ(*a->read().get<TestI32Kind>(), -42);
  OwningProperty snapshot = a->copy();

  const AbstractProperty &i32Kind =
      *AbstractProperty::lookup(TypeID::get<TestI32Kind>(), &context);
  auto next = OwningProperty::create<TestI32Kind>(i32Kind, 12);
  EXPECT_TRUE(succeeded(a->assign(next.get())));
  EXPECT_EQ(*a->read().get<TestI32Kind>(), 12);
  EXPECT_EQ(*snapshot.get().get<TestI32Kind>(), -42);
  EXPECT_TRUE(succeeded(a->reset()));
  EXPECT_EQ(*a->read().get<TestI32Kind>(), -1);

  const AbstractProperty &arrayKind =
      *AbstractProperty::lookup(TypeID::get<TestArrayKind>(), &context);
  auto empty = OwningProperty::create<TestArrayKind>(arrayKind, {});
  EXPECT_TRUE(failed(array->assign(empty.get())));
  EXPECT_EQ(array->read().get<TestArrayKind>()->size(), 2u);
  EXPECT_TRUE(failed(array->reset()));
  EXPECT_TRUE(failed(a->assign(empty.get())));
  EXPECT_EQ(*a->read().get<TestI32Kind>(), -1);

  MLIRContext otherContext;
  otherContext.getOrLoadDialect<TestOpPropertiesDialect>();
  const AbstractProperty &otherKind =
      *AbstractProperty::lookup(TypeID::get<TestI32Kind>(), &otherContext);
  auto foreign = OwningProperty::create<TestI32Kind>(otherKind, 99);
  EXPECT_TRUE(failed(a->assign(foreign.get())));
  EXPECT_EQ(*a->read().get<TestI32Kind>(), -1);
}

TEST(OpPropertiesTest, ConstructWithRegisteredFields) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  OperationState state(UnknownLoc::get(&context),
                       OpWithProperties::getOperationName());
  const AbstractProperty &i32Kind =
      *AbstractProperty::lookup(TypeID::get<TestI32Kind>(), &context);
  const AbstractProperty &arrayKind =
      *AbstractProperty::lookup(TypeID::get<TestArrayKind>(), &context);
  auto count = OwningProperty::create<TestI32Kind>(i32Kind, 42);
  auto array = OwningProperty::create<TestArrayKind>(arrayKind, {1, 2});
  EXPECT_TRUE(succeeded(state.setNamedProperty("a", count.get())));
  EXPECT_TRUE(succeeded(state.setNamedProperty("array", array.get())));
  EXPECT_EQ(state.getRawProperties().as<TestProperties *>()->a, 42);
  EXPECT_TRUE(failed(state.setNamedProperty("missing", count.get())));
  auto empty = OwningProperty::create<TestArrayKind>(arrayKind, {});
  EXPECT_TRUE(failed(state.setNamedProperty("array", empty.get())));
  EXPECT_EQ(state.getRawProperties().as<TestProperties *>()->array.size(), 2u);

  OwningOpRef<Operation *> op(Operation::create(state));
  ASSERT_TRUE(op);
  EXPECT_EQ(*op->getPropertyField("a")->read().get<TestI32Kind>(), 42);
  EXPECT_EQ(op->getPropertyField("array")->read().get<TestArrayKind>()->size(),
            2u);
}

TEST(OpPropertiesTest, NativeOnlyGenericAssembly) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  ParserConfig config(&context);
  constexpr StringLiteral assembly =
      R"mlir("test_op_properties.native_only"() <{count = &test_op_properties.i32<42>}> : () -> ())mlir";
  OwningOpRef<Operation *> op = parseSourceString(assembly, config);
  ASSERT_TRUE(op);
  ASSERT_TRUE(isa<NativeOnlyOp>(*op));
  EXPECT_EQ(*op->getPropertyField("count")->read().get<TestI32Kind>(), 42);

  std::string printed;
  llvm::raw_string_ostream os(printed);
  op->print(os, OpPrintingFlags().printGenericOpForm());
  EXPECT_NE(printed.find("count = &test_op_properties.i32<42>"),
            std::string::npos);
  OwningOpRef<Operation *> reparsed = parseSourceString(printed, config);
  ASSERT_TRUE(reparsed);
  EXPECT_EQ(*reparsed->getPropertyField("count")->read().get<TestI32Kind>(),
            42);

  constexpr StringLiteral legacy =
      R"mlir("test_op_properties.native_only"() <{count = 42 : i32}> : () -> ())mlir";
  EXPECT_FALSE(parseSourceString(legacy, config));

  std::string bytecode;
  llvm::raw_string_ostream bytecodeStream(bytecode);
  ASSERT_TRUE(succeeded(writeBytecodeToFile(op.get(), bytecodeStream)));
  Block block;
  ASSERT_TRUE(succeeded(readBytecodeFile(
      llvm::MemoryBufferRef(bytecode, "native-only"), &block, config)));
  ASSERT_FALSE(block.empty());
  EXPECT_EQ(*block.front().getPropertyField("count")->read().get<TestI32Kind>(),
            42);
}

// Test diagnostic emission when using invalid dictionary.
TEST(OpPropertiesTest, FailedProperties) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  std::string diagnosticStr;
  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    diagnosticStr += diag.str();
    return success();
  });

  // Parse the operation with some properties.
  ParserConfig config(&context);

  // Parse an operation with invalid (incomplete) properties.
  OwningOpRef<Operation *> owningOp =
      parseSourceString("\"test_op_properties.op_with_properties\"() "
                        "<{a = -42 : i32}> : () -> ()\n",
                        config);
  ASSERT_EQ(owningOp.get(), nullptr);
  EXPECT_STREQ(
      "invalid properties {a = -42 : i32} for op "
      "test_op_properties.op_with_properties: expected FloatAttr for key `b`",
      diagnosticStr.c_str());
  diagnosticStr.clear();

  owningOp = parseSourceString(mlirSrc, config);
  Operation *op = owningOp.get();
  ASSERT_TRUE(op != nullptr);
  Location loc = op->getLoc();
  auto opWithProp = dyn_cast<OpWithProperties>(op);
  ASSERT_TRUE(opWithProp);

  OperationState state(loc, op->getName());
  Builder b{&context};
  NamedAttrList attrs;
  attrs.push_back(b.getNamedAttr("a", b.getStringAttr("foo")));
  state.propertiesAttr = attrs.getDictionary(&context);
  {
    auto emitError = [&]() {
      return op->emitError("setting properties failed: ");
    };
    auto result = state.setProperties(op, emitError);
    EXPECT_TRUE(result.failed());
  }
  EXPECT_STREQ("setting properties failed: expected IntegerAttr for key `a`",
               diagnosticStr.c_str());
}

TEST(OpPropertiesTest, DefaultValues) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  OperationState state(UnknownLoc::get(&context),
                       "test_op_properties.op_with_properties");
  Operation *op = Operation::create(state);
  ASSERT_TRUE(op != nullptr);
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    op->print(os);
    StringRef view(output);
    EXPECT_TRUE(view.contains("a = -1"));
    EXPECT_TRUE(view.contains("b = -1"));
    EXPECT_TRUE(view.contains("array = array<i64: -33>"));
  }
  op->erase();
}

TEST(OpPropertiesTest, Cloning) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  ParserConfig config(&context);
  // Parse the operation with some properties.
  OwningOpRef<Operation *> op = parseSourceString(mlirSrc, config);
  ASSERT_TRUE(op.get() != nullptr);
  auto opWithProp = dyn_cast<OpWithProperties>(op.get());
  ASSERT_TRUE(opWithProp);
  Operation *clone = opWithProp->clone();

  // Check that op and its clone prints equally
  std::string opStr;
  std::string cloneStr;
  {
    llvm::raw_string_ostream os(opStr);
    op.get()->print(os);
  }
  {
    llvm::raw_string_ostream os(cloneStr);
    clone->print(os);
  }
  clone->erase();
  EXPECT_STREQ(opStr.c_str(), cloneStr.c_str());
}

TEST(OpPropertiesTest, Equivalence) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  ParserConfig config(&context);
  // Parse the operation with some properties.
  OwningOpRef<Operation *> op = parseSourceString(mlirSrc, config);
  ASSERT_TRUE(op.get() != nullptr);
  auto opWithProp = dyn_cast<OpWithProperties>(op.get());
  ASSERT_TRUE(opWithProp);
  llvm::hash_code reference = OperationEquivalence::computeHash(opWithProp);
  TestProperties &prop = opWithProp.getProperties();
  prop.a = 42;
  EXPECT_NE(reference, OperationEquivalence::computeHash(opWithProp));
  prop.a = -42;
  EXPECT_EQ(reference, OperationEquivalence::computeHash(opWithProp));
  prop.b = 42.;
  EXPECT_NE(reference, OperationEquivalence::computeHash(opWithProp));
  prop.b = -42.;
  EXPECT_EQ(reference, OperationEquivalence::computeHash(opWithProp));
  prop.array.push_back(42);
  EXPECT_NE(reference, OperationEquivalence::computeHash(opWithProp));
  prop.array.pop_back();
  EXPECT_EQ(reference, OperationEquivalence::computeHash(opWithProp));
}

TEST(OpPropertiesTest, getOrAddProperties) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  OperationState state(UnknownLoc::get(&context),
                       "test_op_properties.op_with_properties");
  // Test `getOrAddProperties` API on OperationState.
  TestProperties &prop = state.getOrAddProperties<TestProperties>();
  prop.a = 1;
  prop.b = 2;
  prop.array = {3, 4, 5};
  Operation *op = Operation::create(state);
  ASSERT_TRUE(op != nullptr);
  {
    std::string output;
    llvm::raw_string_ostream os(output);
    op->print(os);
    StringRef view(output);
    EXPECT_TRUE(view.contains("a = 1"));
    EXPECT_TRUE(view.contains("b = 2"));
    EXPECT_TRUE(view.contains("array = array<i64: 3, 4, 5>"));
  }
  op->erase();
}

constexpr StringLiteral withoutPropertiesAttrsSrc = R"mlir(
    "test_op_properties.op_without_properties"()
      {inherent_attr = 42, other_attr = 56} : () -> ()
)mlir";

TEST(OpPropertiesTest, withoutPropertiesDiscardableAttrs) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  ParserConfig config(&context);
  OwningOpRef<Operation *> op =
      parseSourceString(withoutPropertiesAttrsSrc, config);
  ASSERT_EQ(llvm::range_size(op->getDiscardableAttrDictionary().getValue()),
            1u);
  EXPECT_EQ(op->getDiscardableAttrDictionary()
                .getValue()
                .begin()
                ->getName()
                .getValue(),
            "other_attr");

  EXPECT_EQ(op->getInherentAttr("inherent_attr"), std::nullopt);
  EXPECT_NE(op->getDiscardableAttr("inherent_attr"), Attribute());
  EXPECT_NE(op->getDiscardableAttr("other_attr"), Attribute());

  std::string output;
  llvm::raw_string_ostream os(output);
  op->print(os);
  StringRef view(output);
  EXPECT_TRUE(view.contains("inherent_attr = 42"));
  EXPECT_TRUE(view.contains("other_attr = 56"));

  OwningOpRef<Operation *> reparsed = parseSourceString(os.str(), config);
  auto trivialHash = [](Value v) { return hash_value(v); };
  auto hash = [&](Operation *operation) {
    return OperationEquivalence::computeHash(
        operation, trivialHash, trivialHash,
        OperationEquivalence::Flags::IgnoreLocations);
  };
  EXPECT_TRUE(hash(op.get()) == hash(reparsed.get()));
}

} // namespace
