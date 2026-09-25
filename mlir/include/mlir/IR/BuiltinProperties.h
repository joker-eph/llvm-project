//===- BuiltinProperties.h - Builtin native property kinds ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINPROPERTIES_H
#define MLIR_IR_BUILTINPROPERTIES_H

#include "mlir/IR/Properties.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {

struct BoolProperty {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BoolProperty)
  using StorageType = bool;
  static constexpr llvm::StringLiteral name = "bool";
  static LogicalResult verify(bool) { return success(); }
};

struct I64Property {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(I64Property)
  using StorageType = int64_t;
  static constexpr llvm::StringLiteral name = "i64";
  static LogicalResult verify(int64_t) { return success(); }
};

struct StringPropertyStorage {
  std::string value;
  bool operator==(const StringPropertyStorage &other) const {
    return value == other.value;
  }
  friend llvm::hash_code hash_value(const StringPropertyStorage &storage) {
    return llvm::hash_value(StringRef(storage.value));
  }
};

struct StringProperty {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StringProperty)
  using StorageType = StringPropertyStorage;
  static constexpr llvm::StringLiteral name = "string";
  static LogicalResult verify(const StorageType &) { return success(); }
};

/// A distinct registered kind for each element kind. The element's semantic
/// kind, rather than its C++ storage type, is part of this kind's TypeID.
template <typename ElementKind>
struct ArrayPropertyKind {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ArrayPropertyKind)
  using ElementStorage = typename ElementKind::StorageType;
  using StorageType = llvm::SmallVector<ElementStorage>;
  static std::string getName(Dialect &dialect) {
    return getComposedPropertyName(dialect, TypeID::get<ElementKind>(),
                                   "array");
  }

  static LogicalResult verify(const StorageType &values) {
    for (const ElementStorage &value : values)
      if (failed(ElementKind::verify(value)))
        return failure();
    return success();
  }
  static llvm::hash_code hash(const StorageType &values) {
    llvm::hash_code result = llvm::hash_value(values.size());
    for (const ElementStorage &value : values)
      result = llvm::hash_combine(
          result,
          detail::RegisteredPropertyHash<ElementKind, ElementStorage>::get(
              value));
    return result;
  }
  static ParseResult parse(AsmParser &parser, StorageType &values) {
    if (parser.parseLSquare())
      return failure();
    if (succeeded(parser.parseOptionalRSquare()))
      return success();
    do {
      ElementStorage value{};
      if (failed(ElementKind::parse(parser, value)))
        return failure();
      values.push_back(std::move(value));
    } while (succeeded(parser.parseOptionalComma()));
    return parser.parseRSquare();
  }
  static void print(AsmPrinter &printer, const StorageType &values) {
    printer << '[';
    bool first = true;
    for (const ElementStorage &value : values) {
      if (!first)
        printer << ", ";
      first = false;
      ElementKind::print(printer, value);
    }
    printer << ']';
  }
};

template <typename ElementKind>
struct OptionalPropertyKind {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptionalPropertyKind)
  using ElementStorage = typename ElementKind::StorageType;
  using StorageType = std::optional<ElementStorage>;
  static std::string getName(Dialect &dialect) {
    return getComposedPropertyName(dialect, TypeID::get<ElementKind>(),
                                   "optional");
  }

  static LogicalResult verify(const StorageType &value) {
    return value ? ElementKind::verify(*value) : success();
  }
  static llvm::hash_code hash(const StorageType &value) {
    return value ? llvm::hash_combine(
                       true, detail::RegisteredPropertyHash<
                                 ElementKind, ElementStorage>::get(*value))
                 : llvm::hash_value(false);
  }
  static ParseResult parse(AsmParser &parser, StorageType &value) {
    if (succeeded(parser.parseOptionalKeyword("none"))) {
      value.reset();
      return success();
    }
    ElementStorage parsed{};
    if (failed(ElementKind::parse(parser, parsed)))
      return failure();
    value = std::move(parsed);
    return success();
  }
  static void print(AsmPrinter &printer, const StorageType &value) {
    if (value)
      ElementKind::print(printer, *value);
    else
      printer << "none";
  }
};

} // namespace mlir

#endif // MLIR_IR_BUILTINPROPERTIES_H
