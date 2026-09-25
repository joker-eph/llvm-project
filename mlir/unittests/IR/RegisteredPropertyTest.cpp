//===- RegisteredPropertyTest.cpp - Registered property tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinProperties.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Properties.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace {
struct BoolProp {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BoolProp)
  using StorageType = bool;
  static constexpr llvm::StringLiteral name = "bool";
  static LogicalResult verify(bool) { return success(); }
  static ParseResult parse(AsmParser &parser, bool &value) {
    int64_t number;
    if (parser.parseInteger(number) || (number != 0 && number != 1))
      return failure();
    value = number;
    return success();
  }
  static void print(AsmPrinter &printer, bool value) {
    printer.printInteger(value);
  }
};

struct UnitProp {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnitProp)
  using StorageType = bool;
  static constexpr llvm::StringLiteral name = "unit";
  static LogicalResult verify(bool value) { return success(value); }
  static ParseResult parse(AsmParser &parser, bool &value) {
    int64_t number;
    if (parser.parseInteger(number) || (number != 0 && number != 1))
      return failure();
    value = number;
    return success();
  }
  static void print(AsmPrinter &printer, bool value) {
    printer.printInteger(value);
  }
};

struct alignas(128) TrackedStorage {
  static int live;
  std::string value;
  TrackedStorage() { ++live; }
  explicit TrackedStorage(std::string value) : value(std::move(value)) {
    ++live;
  }
  TrackedStorage(const TrackedStorage &other) : value(other.value) { ++live; }
  ~TrackedStorage() { --live; }
  bool operator==(const TrackedStorage &other) const {
    return value == other.value;
  }
  friend llvm::hash_code hash_value(const TrackedStorage &value) {
    return llvm::hash_value(StringRef(value.value));
  }
};
int TrackedStorage::live = 0;

struct TrackedProp {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TrackedProp)
  using StorageType = TrackedStorage;
  static constexpr llvm::StringLiteral name = "tracked";
  static LogicalResult verify(const TrackedStorage &value) {
    return success(!value.value.empty());
  }
  static ParseResult parse(AsmParser &parser, TrackedStorage &value) {
    return parser.parseString(&value.value);
  }
  static void print(AsmPrinter &printer, const TrackedStorage &value) {
    printer.printString(value.value);
  }
};

enum class TestMode { Fast, Slow };
struct ModeProp {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModeProp)
  using StorageType = TestMode;
  static constexpr llvm::StringLiteral name = "mode";
  static LogicalResult verify(TestMode) { return success(); }
  static llvm::hash_code hash(TestMode mode) {
    return llvm::hash_value(static_cast<int>(mode));
  }
  static ParseResult parse(AsmParser &parser, TestMode &mode) {
    if (succeeded(parser.parseOptionalKeyword("fast"))) {
      mode = TestMode::Fast;
      return success();
    }
    if (parser.parseKeyword("slow"))
      return failure();
    mode = TestMode::Slow;
    return success();
  }
  static void print(AsmPrinter &printer, TestMode mode) {
    printer << (mode == TestMode::Fast ? "fast" : "slow");
  }
};

struct ArrayProp {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ArrayProp)
  using StorageType = std::vector<int64_t>;
  static constexpr llvm::StringLiteral name = "array";
  static LogicalResult verify(const StorageType &) { return success(); }
  static llvm::hash_code hash(const StorageType &values) {
    return llvm::hash_combine_range(values.begin(), values.end());
  }
  static ParseResult parse(AsmParser &parser, StorageType &values) {
    if (parser.parseLSquare())
      return failure();
    if (succeeded(parser.parseOptionalRSquare()))
      return success();
    do {
      int64_t value;
      if (parser.parseInteger(value))
        return failure();
      values.push_back(value);
    } while (succeeded(parser.parseOptionalComma()));
    return parser.parseRSquare();
  }
  static void print(AsmPrinter &printer, const StorageType &values) {
    printer << '[';
    llvm::interleaveComma(values, printer.getStream());
    printer << ']';
  }
};

struct OptionalProp {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptionalProp)
  using StorageType = std::optional<int64_t>;
  static constexpr llvm::StringLiteral name = "optional";
  static LogicalResult verify(const StorageType &) { return success(); }
  static llvm::hash_code hash(const StorageType &value) {
    return value ? llvm::hash_combine(true, *value) : llvm::hash_value(false);
  }
  static ParseResult parse(AsmParser &parser, StorageType &value) {
    if (succeeded(parser.parseOptionalKeyword("none"))) {
      value.reset();
      return success();
    }
    int64_t number;
    if (parser.parseInteger(number))
      return failure();
    value = number;
    return success();
  }
  static void print(AsmPrinter &printer, const StorageType &value) {
    if (value)
      printer.printInteger(*value);
    else
      printer << "none";
  }
};

struct PropertyDialect : public Dialect {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PropertyDialect)
  static StringRef getDialectNamespace() { return "registered_property_test"; }
  PropertyDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<PropertyDialect>()) {
    addProperties<BoolProp, UnitProp, TrackedProp, ModeProp, ArrayProp,
                  OptionalProp, ArrayPropertyKind<BoolProp>,
                  ArrayPropertyKind<UnitProp>,
                  OptionalPropertyKind<ModeProp>>();
    addProperties<BoolProp>();
  }
};

struct NumberInterfaceTraits {
  struct Concept {
    virtual int64_t getNumber(Property) const = 0;
  };
  template <typename T>
  struct Model : Concept {};
  template <typename T>
  struct FallbackModel : Concept {};
  template <typename T, typename U>
  struct ExternalModel : Concept {};
};

class NumberInterface
    : public PropertyInterface<NumberInterface, NumberInterfaceTraits> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NumberInterface)
  using Base::Base;
  int64_t getNumber() const { return getImpl()->getNumber(*this); }
};

struct NativeNumberModel : NumberInterfaceTraits::Concept {
  using Interface = NumberInterface;
  int64_t getNumber(Property property) const override {
    return *property.get<I64Property>();
  }
};

struct AttributeNumberModel : NumberInterfaceTraits::Concept {
  using Interface = NumberInterface;
  int64_t getNumber(Property property) const override {
    return cast<IntegerAttr>(property.getAttribute()).getInt();
  }
};

TEST(RegisteredProperty, KindIdentityOwnershipAndAttributes) {
  MLIRContext context;
  auto *dialect = context.getOrLoadDialect<PropertyDialect>();
  const AbstractProperty *boolKind =
      dialect->lookupProperty(TypeID::get<BoolProp>());
  const AbstractProperty *unitKind =
      dialect->lookupProperty(TypeID::get<UnitProp>());
  ASSERT_NE(boolKind, nullptr);
  ASSERT_NE(unitKind, nullptr);
  EXPECT_NE(boolKind, unitKind);
  EXPECT_EQ(dialect->lookupProperty("bool"), boolKind);
  EXPECT_EQ(dialect->lookupProperty("unit"), unitKind);
  EXPECT_EQ(AbstractProperty::lookup(TypeID::get<BoolProp>(), &context),
            boolKind);
  EXPECT_EQ(AbstractProperty::lookup("registered_property_test.bool", &context),
            boolKind);
  EXPECT_EQ(
      AbstractProperty::lookup("registered_property_test.unknown", &context),
      nullptr);

  auto boolValue = OwningProperty::create<BoolProp>(*boolKind, true);
  auto unitValue = OwningProperty::create<UnitProp>(*unitKind, true);
  EXPECT_TRUE(*boolValue.get().get<BoolProp>());
  EXPECT_EQ(boolValue.get().get<UnitProp>(), nullptr);
  EXPECT_NE(boolValue.get(), unitValue.get());
  EXPECT_TRUE(succeeded(boolValue.get().verify()));
  EXPECT_TRUE(succeeded(unitValue.get().verify()));

  Builder builder(&context);
  Attribute attr = builder.getBoolAttr(true);
  OwningProperty wrapped(attr);
  EXPECT_TRUE(wrapped.get().isAttribute());
  EXPECT_EQ(wrapped.clone().get(), wrapped.get());
  EXPECT_NE(wrapped.get(), boolValue.get());
  EXPECT_EQ(wrapped.get().getContext(), &context);
}

TEST(RegisteredProperty, AlignedStorageAndClone) {
  MLIRContext context;
  const AbstractProperty &kind =
      *context.getOrLoadDialect<PropertyDialect>()->lookupProperty(
          TypeID::get<TrackedProp>());
  EXPECT_EQ(TrackedStorage::live, 0);
  {
    TrackedStorage source("payload");
    auto value = OwningProperty::create<TrackedProp>(kind, source);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(value.get().getStorage()) % 128, 0u);
    auto copy = value.clone();
    EXPECT_EQ(value.get(), copy.get());
    EXPECT_EQ(TrackedStorage::live, 3);
    auto empty = OwningProperty::create(kind);
    EXPECT_TRUE(failed(empty.get().verify()));
  }
  EXPECT_EQ(TrackedStorage::live, 0);
}

TEST(RegisteredProperty, BuiltinKinds) {
  MLIRContext context;
  const AbstractProperty *i64 =
      AbstractProperty::lookup("builtin.i64", &context);
  ASSERT_NE(i64, nullptr);
  EXPECT_EQ(i64->getTypeID(), TypeID::get<I64Property>());
  auto value = OwningProperty::create<I64Property>(*i64, 42);
  ASSERT_NE(value.get().get<I64Property>(), nullptr);
  EXPECT_EQ(*value.get().get<I64Property>(), 42);

  const AbstractProperty *string =
      AbstractProperty::lookup("builtin.string", &context);
  ASSERT_NE(string, nullptr);
  StringPropertyStorage source{"text"};
  auto text = OwningProperty::create<StringProperty>(*string, source);
  EXPECT_EQ(text.get().get<StringProperty>()->value, "text");
}

TEST(RegisteredProperty, SharedInterfaceAndExternalModels) {
  MLIRContext context;
  auto *builtin = context.getOrLoadDialect<BuiltinDialect>();
  builtin->attachPropertyInterface<I64Property, NativeNumberModel>();
  builtin
      ->attachAttributePropertyInterface<IntegerAttr, AttributeNumberModel>();

  const AbstractProperty &kind =
      *builtin->lookupProperty(TypeID::get<I64Property>());
  auto native = OwningProperty::create<I64Property>(kind, 42);
  Builder builder(&context);
  Property attribute(builder.getI64IntegerAttr(17));
  EXPECT_EQ(NumberInterface(native.get()).getNumber(), 42);
  EXPECT_EQ(NumberInterface(attribute).getNumber(), 17);
  EXPECT_EQ(native.get().getInterface(NumberInterface::getInterfaceID()),
            kind.getInterface(NumberInterface::getInterfaceID()));
}

TEST(RegisteredProperty, NamedComposedKinds) {
  MLIRContext context;
  auto *dialect = context.getOrLoadDialect<PropertyDialect>();
  const AbstractProperty &mode =
      *dialect->lookupProperty(TypeID::get<ModeProp>());
  const AbstractProperty &array =
      *dialect->lookupProperty(TypeID::get<ArrayProp>());
  const AbstractProperty &optional =
      *dialect->lookupProperty(TypeID::get<OptionalProp>());
  auto modeValue = OwningProperty::create<ModeProp>(mode, TestMode::Fast);
  auto arrayValue = OwningProperty::create<ArrayProp>(array, {1, 2, 3});
  auto optionalValue = OwningProperty::create<OptionalProp>(optional, 4);
  EXPECT_EQ(*modeValue.get().get<ModeProp>(), TestMode::Fast);
  EXPECT_EQ(arrayValue.get().get<ArrayProp>()->size(), 3u);
  EXPECT_EQ(optionalValue.get().get<OptionalProp>()->value(), 4);
  EXPECT_EQ(arrayValue.get(), arrayValue.clone().get());
  EXPECT_NE(arrayValue.get(), optionalValue.get());
}

TEST(RegisteredProperty, TemplateCombinatorsPreserveElementIdentity) {
  MLIRContext context;
  auto *dialect = context.getOrLoadDialect<PropertyDialect>();
  using BoolArray = ArrayPropertyKind<BoolProp>;
  using UnitArray = ArrayPropertyKind<UnitProp>;
  const AbstractProperty *boolArray =
      dialect->lookupProperty(TypeID::get<BoolArray>());
  const AbstractProperty *unitArray =
      dialect->lookupProperty(TypeID::get<UnitArray>());
  ASSERT_NE(boolArray, nullptr);
  ASSERT_NE(unitArray, nullptr);
  EXPECT_NE(boolArray->getTypeID(), unitArray->getTypeID());
  EXPECT_EQ(boolArray->getName(), "array.registered_property_test.bool");
  EXPECT_EQ(unitArray->getName(), "array.registered_property_test.unit");
  EXPECT_EQ(boolArray, dialect->lookupProperty(boolArray->getName()));
  auto parsed = parseProperty(
      "&registered_property_test.array.registered_property_test.bool<[1, "
      "0]>",
      &context);
  ASSERT_TRUE(succeeded(parsed));
  EXPECT_EQ(parsed->get().get<BoolArray>()->size(), 2u);
  EXPECT_EQ(parsed->get().get<UnitArray>(), nullptr);

  using ModeOptional = OptionalPropertyKind<ModeProp>;
  auto optional = parseProperty(
      "&registered_property_test.optional.registered_property_test.mode<slow>",
      &context);
  ASSERT_TRUE(succeeded(optional));
  EXPECT_EQ(optional->get().get<ModeOptional>()->value(), TestMode::Slow);
  auto absent = parseProperty(
      "none", *dialect->lookupProperty(TypeID::get<ModeOptional>()));
  ASSERT_TRUE(succeeded(absent));
  EXPECT_FALSE(absent->get().get<ModeOptional>()->has_value());
}

TEST(RegisteredProperty, StandaloneParsingAndPrinting) {
  MLIRContext context;
  context.getOrLoadDialect<PropertyDialect>();
  auto print = [](Property property) {
    std::string result;
    llvm::raw_string_ostream stream(result);
    property.print(stream);
    return result;
  };
  auto value = parseProperty("&builtin.i64<42>", &context);
  ASSERT_TRUE(succeeded(value));
  ASSERT_NE(value->get().get<I64Property>(), nullptr);
  EXPECT_EQ(*value->get().get<I64Property>(), 42);
  EXPECT_EQ(print(value->get()), "&builtin.i64<42>");

  const AbstractProperty &kind =
      *AbstractProperty::lookup("builtin.i64", &context);
  auto payload = parseProperty("-17", kind);
  ASSERT_TRUE(succeeded(payload));
  EXPECT_EQ(*payload->get().get<I64Property>(), -17);

  auto attribute = parseProperty("42 : i64", &context);
  ASSERT_TRUE(succeeded(attribute));
  EXPECT_TRUE(attribute->get().isAttribute());
  EXPECT_EQ(attribute->get().getAttribute(),
            parseAttribute("42 : i64", &context));
  EXPECT_EQ(print(attribute->get()), "42 : i64");

  auto string = parseProperty("&builtin.string<\"hello\">", &context);
  ASSERT_TRUE(succeeded(string));
  EXPECT_EQ(string->get().get<StringProperty>()->value, "hello");
  EXPECT_EQ(print(string->get()), "&builtin.string<\"hello\">");

  auto mode = parseProperty("&registered_property_test.mode<slow>", &context);
  ASSERT_TRUE(succeeded(mode));
  EXPECT_EQ(*mode->get().get<ModeProp>(), TestMode::Slow);
  EXPECT_EQ(print(mode->get()), "&registered_property_test.mode<slow>");

  auto array =
      parseProperty("&registered_property_test.array<[1, 2, 3]>", &context);
  ASSERT_TRUE(succeeded(array));
  EXPECT_EQ(array->get().get<ArrayProp>()->size(), 3u);
  EXPECT_EQ(print(array->get()), "&registered_property_test.array<[1, 2, 3]>");
  auto optional =
      parseProperty("&registered_property_test.optional<none>", &context);
  ASSERT_TRUE(succeeded(optional));
  EXPECT_FALSE(optional->get().get<OptionalProp>()->has_value());
}

TEST(RegisteredProperty, ParsingErrorsDestroyTemporaryStorage) {
  MLIRContext context;
  context.getOrLoadDialect<PropertyDialect>();
  std::vector<std::string> errors;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
    errors.push_back(diag.str());
    return success();
  });
  EXPECT_EQ(TrackedStorage::live, 0);
  EXPECT_TRUE(failed(
      parseProperty("&registered_property_test.tracked<\"\">", &context)));
  ASSERT_FALSE(errors.empty());
  EXPECT_NE(errors.back().find("invalid value for property"),
            std::string::npos);
  EXPECT_EQ(TrackedStorage::live, 0);
  EXPECT_TRUE(failed(
      parseProperty("&registered_property_test.tracked<\"a\"", &context)));
  EXPECT_EQ(TrackedStorage::live, 0);
  EXPECT_TRUE(failed(parseProperty("&builtin.i64<42> trailing", &context)));
  EXPECT_TRUE(failed(parseProperty(
      "42 trailing", *AbstractProperty::lookup("builtin.i64", &context))));
  EXPECT_TRUE(failed(parseProperty("&builtin.unknown<42>", &context)));
  EXPECT_NE(errors.back().find("unknown property kind"), std::string::npos);
  EXPECT_TRUE(failed(parseProperty("&unavailable.i64<42>", &context)));
  EXPECT_NE(errors.back().find("is not available"), std::string::npos);
}
} // namespace
