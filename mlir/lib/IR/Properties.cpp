//===- Properties.cpp - Registered property values -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Properties.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/ErrorHandling.h"
#include <new>
#include <utility>

using namespace mlir;

namespace {
void *allocateStorage(const AbstractProperty &kind) {
  if (kind.getAlignment() > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    return ::operator new(kind.getSize(),
                          std::align_val_t(kind.getAlignment()));
  return ::operator new(kind.getSize());
}

void deallocateStorage(const AbstractProperty &kind, void *storage) {
  if (kind.getAlignment() > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    ::operator delete(storage, std::align_val_t(kind.getAlignment()));
  else
    ::operator delete(storage);
}
} // namespace

const AbstractProperty *AbstractProperty::lookup(TypeID typeID,
                                                 MLIRContext *context) {
  for (Dialect *dialect : context->getLoadedDialects())
    if (const AbstractProperty *kind = dialect->lookupProperty(typeID))
      return kind;
  return nullptr;
}

std::string mlir::getComposedPropertyName(Dialect &dialect, TypeID elementID,
                                          StringRef combinator) {
  const AbstractProperty *element = dialect.lookupProperty(elementID);
  if (!element)
    element = AbstractProperty::lookup(elementID, dialect.getContext());
  if (!element)
    llvm::report_fatal_error(
        "register the element kind before its property composition");
  std::string name = combinator.str();
  name += '.';
  name += element->getDialect().getNamespace().str();
  name += '.';
  name += element->getName().str();
  return name;
}

const AbstractProperty *AbstractProperty::lookup(StringRef name,
                                                 MLIRContext *context) {
  auto split = name.split('.');
  if (split.second.empty())
    return nullptr;
  Dialect *dialect = context->getOrLoadDialect(split.first);
  return dialect ? dialect->lookupProperty(split.second) : nullptr;
}

MLIRContext *AbstractProperty::getContext() const {
  return dialect.getContext();
}

MLIRContext *Property::getContext() const {
  return kind ? kind->getContext() : attr ? attr.getContext() : nullptr;
}

void *Property::getInterface(TypeID interfaceID) const {
  if (kind)
    return kind->getInterface(interfaceID);
  if (!attr)
    return nullptr;
  Attribute attribute = attr;
  return attr.getDialect().lookupAttributePropertyInterface(
      attribute.getTypeID(), interfaceID);
}

bool mlir::operator==(Property lhs, Property rhs) {
  if (lhs.kind || rhs.kind)
    return lhs.kind && rhs.kind && lhs.kind == rhs.kind &&
           lhs.kind->equals(lhs.value, rhs.value);
  return lhs.attr == rhs.attr;
}

llvm::hash_code Property::hash() const {
  if (kind)
    return llvm::hash_combine(kind->getTypeID(), kind->hash(value));
  return hash_value(attr);
}

OwningProperty OwningProperty::create(const AbstractProperty &kind) {
  void *storage = allocateStorage(kind);
  kind.construct(storage);
  return OwningProperty(kind, storage);
}

OwningProperty OwningProperty::copy(Property property) {
  if (property.isAttribute())
    return OwningProperty(property.getAttribute());
  const AbstractProperty *kind = property.getKind();
  if (!kind)
    return {};
  void *storage = allocateStorage(*kind);
  kind->copy(storage, property.getStorage());
  return OwningProperty(*kind, storage);
}

OwningProperty::OwningProperty(OwningProperty &&other) noexcept
    : kind(std::exchange(other.kind, nullptr)),
      storage(std::exchange(other.storage, nullptr)),
      attr(std::exchange(other.attr, Attribute())) {}

OwningProperty &OwningProperty::operator=(OwningProperty &&other) noexcept {
  if (this != &other) {
    reset();
    kind = std::exchange(other.kind, nullptr);
    storage = std::exchange(other.storage, nullptr);
    attr = std::exchange(other.attr, Attribute());
  }
  return *this;
}

OwningProperty::~OwningProperty() { reset(); }

void OwningProperty::reset() {
  if (storage) {
    kind->destroy(storage);
    deallocateStorage(*kind, storage);
  }
  kind = nullptr;
  storage = nullptr;
  attr = {};
}
