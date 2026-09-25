//===- registered_properties.c - C API operation property test ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-registered-property-test | FileCheck %s

#include "PythonTestCAPI.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
  MlirContext context = mlirContextCreate();
  MlirDialectHandle dialect = mlirGetDialectHandle__python_test__();
  mlirDialectHandleRegisterDialect(dialect, context);
  mlirDialectHandleLoadDialect(dialect, context);

  MlirProperty count = mlirI64PropertyGet(context, 42);
  int64_t interfaceNumber = 0;
  assert(mlirPythonTestPropertyGetNumber(count, &interfaceNumber) &&
         interfaceNumber == 42);
  MlirProperty attributeValue = mlirPropertyFromAttribute(
      mlirIntegerAttrGet(mlirIntegerTypeGet(context, 64), 17));
  assert(mlirPythonTestPropertyGetNumber(attributeValue, &interfaceNumber) &&
         interfaceNumber == 17);
  MlirNamedProperty named = {mlirStringRefCreateFromCString("count"), count};
  MlirOperationState state = mlirOperationStateGet(
      mlirStringRefCreateFromCString("python_test.properties"),
      mlirLocationUnknownGet(context));
  mlirOperationStateEnableResultTypeInference(&state);
  MlirOperation op = mlirOperationCreateWithProperties(&state, 1, &named);
  assert(!mlirOperationIsNull(op));
  assert(mlirOperationGetNumResults(op) == 1);
  assert(mlirIntegerTypeGetWidth(
             mlirValueGetType(mlirOperationGetResult(op, 0))) == 42);
  MlirOperationState rejected = mlirOperationStateGet(
      mlirStringRefCreateFromCString("python_test.properties"),
      mlirLocationUnknownGet(context));
  MlirNamedAttribute inherent = mlirNamedAttributeGet(
      mlirIdentifierGet(context, mlirStringRefCreateFromCString("count")),
      mlirIntegerAttrGet(mlirIntegerTypeGet(context, 64), 1));
  mlirOperationStateAddAttributes(&rejected, 1, &inherent);
  assert(mlirOperationIsNull(
      mlirOperationCreateWithProperties(&rejected, 1, &named)));
  assert(mlirOperationGetNumPropertyFields(op) == 1);
  MlirOperationPropertyRef field = mlirOperationGetPropertyFieldByName(
      op, mlirStringRefCreateFromCString("count"));
  assert(!mlirOperationPropertyRefIsNull(field));
  MlirProperty snapshot = mlirOperationPropertyRefCopy(field);
  int64_t value = 0;
  assert(mlirPropertyGetI64(snapshot, &value) && value == 42);
  MlirProperty replacement = mlirI64PropertyGet(context, 7);
  assert(mlirLogicalResultIsSuccess(
      mlirOperationPropertyRefAssign(field, replacement)));
  assert(mlirLogicalResultIsFailure(mlirOperationPropertyRefReset(field)));

  MlirContext foreignContext = mlirContextCreate();
  MlirProperty foreign = mlirI64PropertyGet(foreignContext, 99);
  assert(mlirLogicalResultIsFailure(
      mlirOperationPropertyRefAssign(field, foreign)));
  MlirProperty current = mlirOperationPropertyRefCopy(field);
  assert(mlirPropertyGetI64(current, &value) && value == 7);
  mlirOperationDestroy(op);
  assert(mlirPropertyGetI64(snapshot, &value) && value == 42);

  MlirNamedProperty positive = {mlirStringRefCreateFromCString("positive"),
                                count};
  assert(mlirContextIsRegisteredOperation(
      context,
      mlirStringRefCreateFromCString("python_test.checked_properties")));
  MlirOperationState checkedState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("python_test.checked_properties"),
      mlirLocationUnknownGet(context));
  MlirOperation checked =
      mlirOperationCreateWithProperties(&checkedState, 1, &positive);
  assert(!mlirOperationIsNull(checked));
  MlirOperationPropertyRef positiveField = mlirOperationGetPropertyFieldByName(
      checked, mlirStringRefCreateFromCString("positive"));
  MlirOperationPropertyRef fallbackField = mlirOperationGetPropertyFieldByName(
      checked, mlirStringRefCreateFromCString("fallback"));
  MlirProperty fallback = mlirOperationPropertyRefCopy(fallbackField);
  assert(mlirPropertyGetI64(fallback, &value) && value == 7);
  MlirProperty zero = mlirI64PropertyGet(context, 0);
  assert(mlirLogicalResultIsFailure(
      mlirOperationPropertyRefAssign(positiveField, zero)));
  MlirProperty unchanged = mlirOperationPropertyRefCopy(positiveField);
  assert(mlirPropertyGetI64(unchanged, &value) && value == 42);
  assert(mlirLogicalResultIsSuccess(
      mlirOperationPropertyRefAssign(fallbackField, replacement)));
  assert(
      mlirLogicalResultIsSuccess(mlirOperationPropertyRefReset(fallbackField)));
  MlirProperty resetValue = mlirOperationPropertyRefCopy(fallbackField);
  assert(mlirPropertyGetI64(resetValue, &value) && value == 7);
  assert(
      mlirLogicalResultIsFailure(mlirOperationPropertyRefReset(positiveField)));
  mlirOperationDestroy(checked);
  mlirPropertyDestroy(resetValue);
  mlirPropertyDestroy(unchanged);
  mlirPropertyDestroy(zero);
  mlirPropertyDestroy(fallback);

  MlirProperty array = mlirPropertyParse(
      context,
      mlirStringRefCreateFromCString("&builtin.array.builtin.i64<[1, 2]>"));
  MlirProperty optional = mlirPropertyParseWithKind(
      context, mlirStringRefCreateFromCString("builtin.optional.builtin.i64"),
      mlirStringRefCreateFromCString("3"));
  assert(!mlirPropertyIsNull(array) && !mlirPropertyIsNull(optional));
  MlirNamedProperty composedFields[] = {
      {mlirStringRefCreateFromCString("values"), array},
      {mlirStringRefCreateFromCString("maybe"), optional}};
  MlirOperationState composedState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("python_test.composed_properties"),
      mlirLocationUnknownGet(context));
  MlirOperation composed =
      mlirOperationCreateWithProperties(&composedState, 2, composedFields);
  assert(!mlirOperationIsNull(composed));
  assert(mlirOperationGetNumPropertyFields(composed) == 2);
  MlirOperationPropertyRef valuesField = mlirOperationGetPropertyFieldByName(
      composed, mlirStringRefCreateFromCString("values"));
  MlirProperty valuesCopy = mlirOperationPropertyRefCopy(valuesField);
  assert(mlirPropertyEqual(valuesCopy, array));
  mlirOperationDestroy(composed);
  mlirPropertyDestroy(valuesCopy);
  mlirPropertyDestroy(optional);
  mlirPropertyDestroy(array);

  mlirPropertyDestroy(current);
  mlirPropertyDestroy(foreign);
  mlirContextDestroy(foreignContext);
  mlirPropertyDestroy(replacement);
  mlirPropertyDestroy(snapshot);
  mlirPropertyDestroy(count);
  mlirPropertyDestroy(attributeValue);
  mlirContextDestroy(context);
  // CHECK: registered C properties: ok
  puts("registered C properties: ok");
  return 0;
}
