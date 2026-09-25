//===- PythonTestCAPI.cpp - C API for the PythonTest dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PythonTestCAPI.h"
#include "PythonTestDialect.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinProperties.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PythonTest, python_test,
                                      python_test::PythonTestDialect)

namespace {
struct NumberInterfaceTraits {
  struct Concept {
    virtual int64_t getNumber(mlir::Property) const = 0;
  };
  template <typename T>
  struct Model : Concept {};
  template <typename T>
  struct FallbackModel : Concept {};
  template <typename T, typename U>
  struct ExternalModel : Concept {};
};

class NumberInterface
    : public mlir::PropertyInterface<NumberInterface, NumberInterfaceTraits> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NumberInterface)
  using Base::Base;
  int64_t getNumber() const { return getImpl()->getNumber(*this); }
};

struct NativeNumberModel : NumberInterfaceTraits::Concept {
  using Interface = NumberInterface;
  int64_t getNumber(mlir::Property property) const override {
    return *property.get<mlir::I64Property>();
  }
};

struct AttributeNumberModel : NumberInterfaceTraits::Concept {
  using Interface = NumberInterface;
  int64_t getNumber(mlir::Property property) const override {
    return llvm::cast<mlir::IntegerAttr>(property.getAttribute()).getInt();
  }
};
} // namespace

bool mlirPythonTestPropertyGetNumber(MlirProperty property, int64_t *number) {
  if (!property.ptr || !number)
    return false;
  mlir::Property value = unwrap(property)->get();
  mlir::MLIRContext *context = value.getContext();
  auto *builtin = context->getOrLoadDialect<mlir::BuiltinDialect>();
  const mlir::AbstractProperty *kind =
      builtin->lookupProperty(mlir::TypeID::get<mlir::I64Property>());
  if (!kind->getInterface(NumberInterface::getInterfaceID()))
    builtin->attachPropertyInterface<mlir::I64Property, NativeNumberModel>();
  if (!builtin->lookupAttributePropertyInterface(
          mlir::TypeID::get<mlir::IntegerAttr>(),
          NumberInterface::getInterfaceID()))
    builtin->attachAttributePropertyInterface<mlir::IntegerAttr,
                                              AttributeNumberModel>();
  if (!value.getInterface(NumberInterface::getInterfaceID()))
    return false;
  *number = NumberInterface(value).getNumber();
  return true;
}

bool mlirAttributeIsAPythonTestTestAttribute(MlirAttribute attr) {
  return llvm::isa<python_test::TestAttrAttr>(unwrap(attr));
}

MlirAttribute mlirPythonTestTestAttributeGet(MlirContext context) {
  return wrap(python_test::TestAttrAttr::get(unwrap(context)));
}

MlirTypeID mlirPythonTestTestAttributeGetTypeID(void) {
  return wrap(python_test::TestAttrAttr::getTypeID());
}

bool mlirTypeIsAPythonTestTestType(MlirType type) {
  return llvm::isa<python_test::TestTypeType>(unwrap(type));
}

MlirType mlirPythonTestTestTypeGet(MlirContext context) {
  return wrap(python_test::TestTypeType::get(unwrap(context)));
}

MlirTypeID mlirPythonTestTestTypeGetTypeID(void) {
  return wrap(python_test::TestTypeType::getTypeID());
}

bool mlirTypeIsAPythonTestTestTensorValue(MlirValue value) {
  return mlirTypeIsATensor(wrap(unwrap(value).getType()));
}

void mlirPythonTestEmitDiagnosticWithNote(MlirContext ctx) {
  auto diag =
      mlir::emitError(unwrap(mlirLocationUnknownGet(ctx)), "created error");
  diag.attachNote() << "attached note";
}
