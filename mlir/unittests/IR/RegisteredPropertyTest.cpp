//===- RegisteredPropertyTest.cpp - Registered property tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinProperties.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Properties.h"
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
};

struct UnitProp {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnitProp)
  using StorageType = bool;
  static constexpr llvm::StringLiteral name = "unit";
  static LogicalResult verify(bool value) { return success(value); }
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
};

struct ArrayProp {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ArrayProp)
  using StorageType = std::vector<int64_t>;
  static constexpr llvm::StringLiteral name = "array";
  static LogicalResult verify(const StorageType &) { return success(); }
  static llvm::hash_code hash(const StorageType &values) {
    return llvm::hash_combine_range(values.begin(), values.end());
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
};

struct PropertyDialect : public Dialect {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PropertyDialect)
  static StringRef getDialectNamespace() { return "registered_property_test"; }
  PropertyDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<PropertyDialect>()) {
    addProperties<BoolProp, UnitProp, TrackedProp, ModeProp, ArrayProp,
                  OptionalProp>();
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
} // namespace
