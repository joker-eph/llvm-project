//===- RegisteredPropertyTest.cpp - Registered property tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinProperties.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Properties.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <string>
#include <utility>

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

struct PropertyDialect : public Dialect {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PropertyDialect)
  static StringRef getDialectNamespace() { return "registered_property_test"; }
  PropertyDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<PropertyDialect>()) {
    addProperties<BoolProp, UnitProp, TrackedProp>();
    addProperties<BoolProp>();
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
} // namespace
