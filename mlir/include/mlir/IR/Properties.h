//===- Properties.h - Registered property values ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_PROPERTIES_H
#define MLIR_IR_PROPERTIES_H

#include "mlir/IR/Attributes.h"
#include "mlir/Support/InterfaceSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

namespace mlir {
class AsmParser;
class AsmPrinter;
class Dialect;
class MLIRContext;

/// Form a stable local mnemonic from a registered element kind. The element
/// kind must already be loaded before its composition is registered.
std::string getComposedPropertyName(Dialect &dialect, TypeID elementID,
                                    StringRef combinator);

namespace detail {
template <typename T, typename Storage, typename = void>
struct RegisteredPropertyHash {
  static llvm::hash_code get(const Storage &value) {
    using llvm::hash_value;
    return hash_value(value);
  }
};

template <typename T, typename Storage>
struct RegisteredPropertyHash<
    T, Storage,
    std::void_t<decltype(T::hash(std::declval<const Storage &>()))>> {
  static llvm::hash_code get(const Storage &value) { return T::hash(value); }
};

template <typename T, typename = void>
struct RegisteredPropertyName {
  static std::string get(Dialect &) { return StringRef(T::name).str(); }
};

template <typename T>
struct RegisteredPropertyName<
    T, std::void_t<decltype(T::getName(std::declval<Dialect &>()))>> {
  static std::string get(Dialect &dialect) { return T::getName(dialect); }
};
} // namespace detail

/// The semantic identity and value operations of a native property kind.
/// Instances are owned by a dialect, and thus by its MLIRContext. The TypeID
/// identifies the kind, not the C++ type used to store its values.
/// A kind class supplies StorageType, name, and verify(StorageType). Its
/// storage type must be default and copy constructible, comparable, and
/// hashable with hash_value, or the kind must supply hash(StorageType).
class AbstractProperty {
public:
  static const AbstractProperty *lookup(TypeID typeID, MLIRContext *context);
  /// Look up a fully qualified name such as "builtin.i64". This may load the
  /// dialect if it is available in the context's registry.
  static const AbstractProperty *lookup(StringRef name, MLIRContext *context);
  using ConstructFn = void (*)(void *);
  using CopyFn = void (*)(void *, const void *);
  using DestroyFn = void (*)(void *);
  using EqualFn = bool (*)(const void *, const void *);
  using HashFn = llvm::hash_code (*)(const void *);
  using VerifyFn = LogicalResult (*)(const void *);
  using ParseFn = ParseResult (*)(AsmParser &, void *);
  using PrintFn = void (*)(AsmPrinter &, const void *);

  template <typename T>
  static AbstractProperty get(Dialect &dialect) {
    using Storage = typename T::StorageType;
    return AbstractProperty(
        dialect, TypeID::get<T>(),
        detail::RegisteredPropertyName<T>::get(dialect), sizeof(Storage),
        alignof(Storage), [](void *ptr) { new (ptr) Storage(); },
        [](void *ptr, const void *source) {
          new (ptr) Storage(*static_cast<const Storage *>(source));
        },
        [](void *ptr) { static_cast<Storage *>(ptr)->~Storage(); },
        [](const void *lhs, const void *rhs) {
          return *static_cast<const Storage *>(lhs) ==
                 *static_cast<const Storage *>(rhs);
        },
        [](const void *ptr) {
          return detail::RegisteredPropertyHash<T, Storage>::get(
              *static_cast<const Storage *>(ptr));
        },
        [](const void *ptr) {
          return T::verify(*static_cast<const Storage *>(ptr));
        },
        [](AsmParser &parser, void *ptr) {
          return T::parse(parser, *static_cast<Storage *>(ptr));
        },
        [](AsmPrinter &printer, const void *ptr) {
          T::print(printer, *static_cast<const Storage *>(ptr));
        });
  }

  Dialect &getDialect() const { return dialect; }
  MLIRContext *getContext() const;
  TypeID getTypeID() const { return typeID; }
  StringRef getName() const { return name; }
  size_t getSize() const { return size; }
  size_t getAlignment() const { return alignment; }
  void construct(void *ptr) const { constructFn(ptr); }
  void copy(void *ptr, const void *source) const { copyFn(ptr, source); }
  void destroy(void *ptr) const { destroyFn(ptr); }
  bool equals(const void *lhs, const void *rhs) const {
    return equalFn(lhs, rhs);
  }
  llvm::hash_code hash(const void *ptr) const { return hashFn(ptr); }
  LogicalResult verify(const void *ptr) const { return verifyFn(ptr); }
  ParseResult parse(AsmParser &parser, void *ptr) const {
    return parseFn(parser, ptr);
  }
  void print(AsmPrinter &printer, const void *ptr) const {
    printFn(printer, ptr);
  }
  void *getInterface(TypeID interfaceID) const {
    return interfaceMap.lookupInterface(interfaceID);
  }
  template <typename ModelT>
  void attachInterfaceModel() {
    interfaceMap.insertModels<ModelT>();
  }

private:
  AbstractProperty(Dialect &dialect, TypeID typeID, std::string name,
                   size_t size, size_t alignment, ConstructFn constructFn,
                   CopyFn copyFn, DestroyFn destroyFn, EqualFn equalFn,
                   HashFn hashFn, VerifyFn verifyFn, ParseFn parseFn,
                   PrintFn printFn)
      : dialect(dialect), typeID(typeID), name(std::move(name)), size(size),
        alignment(alignment), constructFn(constructFn), copyFn(copyFn),
        destroyFn(destroyFn), equalFn(equalFn), hashFn(hashFn),
        verifyFn(verifyFn), parseFn(parseFn), printFn(printFn) {}

  Dialect &dialect;
  TypeID typeID;
  std::string name;
  size_t size;
  size_t alignment;
  ConstructFn constructFn;
  CopyFn copyFn;
  DestroyFn destroyFn;
  EqualFn equalFn;
  HashFn hashFn;
  VerifyFn verifyFn;
  ParseFn parseFn;
  PrintFn printFn;
  detail::InterfaceMap interfaceMap;
};

/// Borrowed, immutable view of a native value or a context-owned attribute.
/// Native storage and the context must outlive this view.
class Property {
public:
  Property() = default;
  Property(const AbstractProperty &kind, const void *value)
      : kind(&kind), value(value) {
    assert(value && "native property view requires storage");
  }
  Property(Attribute attr) : attr(attr) {}

  explicit operator bool() const { return value || attr; }
  bool isAttribute() const { return bool(attr); }
  Attribute getAttribute() const { return attr; }
  const AbstractProperty *getKind() const { return kind; }
  TypeID getTypeID() const {
    Attribute attribute = attr;
    return kind        ? kind->getTypeID()
           : attribute ? attribute.getTypeID()
                       : TypeID::get<void>();
  }
  MLIRContext *getContext() const;

  template <typename T>
  const typename T::StorageType *get() const {
    return kind && kind->getTypeID() == TypeID::get<T>()
               ? static_cast<const typename T::StorageType *>(value)
               : nullptr;
  }
  const void *getStorage() const { return value; }
  void *getInterface(TypeID interfaceID) const;
  LogicalResult verify() const {
    return kind ? kind->verify(value) : success();
  }
  llvm::hash_code hash() const;
  void print(raw_ostream &os) const;
  friend bool operator==(Property lhs, Property rhs);
  friend bool operator!=(Property lhs, Property rhs) { return !(lhs == rhs); }

private:
  const AbstractProperty *kind = nullptr;
  const void *value = nullptr;
  Attribute attr;
};

bool operator==(Property lhs, Property rhs);

namespace PropertyTrait {
template <typename ConcreteType, template <typename> class TraitType>
struct TraitBase {};
} // namespace PropertyTrait

/// Base for interfaces implemented by registered native property kinds and
/// attribute-backed values. Models receive a Property view in both cases.
template <typename ConcreteType, typename Traits>
class PropertyInterface
    : public detail::Interface<ConcreteType, Property, Traits, Property,
                               PropertyTrait::TraitBase> {
public:
  using Base = PropertyInterface<ConcreteType, Traits>;
  using InterfaceBase = detail::Interface<ConcreteType, Property, Traits,
                                          Property, PropertyTrait::TraitBase>;
  using InterfaceBase::InterfaceBase;

protected:
  static typename InterfaceBase::Concept *getInterfaceFor(Property property) {
    return static_cast<typename InterfaceBase::Concept *>(
        property.getInterface(ConcreteType::getInterfaceID()));
  }
  friend InterfaceBase;
};

/// Move-only standalone native storage, or a retained context-owned attribute.
/// Cloning is explicit; destroying native storage runs its C++ destructor.
class OwningProperty {
public:
  OwningProperty() = default;
  OwningProperty(const OwningProperty &) = delete;
  OwningProperty &operator=(const OwningProperty &) = delete;
  OwningProperty(OwningProperty &&other) noexcept;
  OwningProperty &operator=(OwningProperty &&other) noexcept;
  ~OwningProperty();

  explicit OwningProperty(Attribute attr) : attr(attr) {}
  static OwningProperty create(const AbstractProperty &kind);
  static OwningProperty copy(Property property);
  template <typename T>
  static OwningProperty create(const AbstractProperty &kind,
                               const typename T::StorageType &value) {
    assert(kind.getTypeID() == TypeID::get<T>() && "property kind mismatch");
    return copy(Property(kind, &value));
  }

  Property get() const {
    return storage ? Property(*kind, storage) : Property(attr);
  }
  void *getMutableStorage() { return storage; }
  OwningProperty clone() const { return copy(get()); }
  explicit operator bool() const { return bool(get()); }

private:
  OwningProperty(const AbstractProperty &kind, void *storage)
      : kind(&kind), storage(storage) {}
  void reset();
  const AbstractProperty *kind = nullptr;
  void *storage = nullptr;
  Attribute attr;
};

/// Parse either a native `&dialect.mnemonic<payload>` property or an existing
/// attribute assembly form. Both reject trailing input.
FailureOr<OwningProperty> parseProperty(StringRef text, MLIRContext *context);

/// Parse only the payload of a known native kind, without its wrapper.
FailureOr<OwningProperty> parseProperty(StringRef payload,
                                        const AbstractProperty &kind);
} // namespace mlir

#endif // MLIR_IR_PROPERTIES_H
