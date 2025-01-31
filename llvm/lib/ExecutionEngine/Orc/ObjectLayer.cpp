//===-------------------- Layer.cpp - Layer interfaces --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Layer.h"

#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ObjectFileInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

char ObjectLayer::ID;

ObjectLayer::ObjectLayer(ExecutionSession &ES) : ES(ES) {}

ObjectLayer::~ObjectLayer() = default;

Error ObjectLayer::add(ResourceTrackerSP RT, std::unique_ptr<MemoryBuffer> O,
                       MaterializationUnit::Interface I) {
  assert(RT && "RT can not be null");
  auto &JD = RT->getJITDylib();
  return JD.define(std::make_unique<BasicObjectLayerMaterializationUnit>(
                       *this, std::move(O), std::move(I)),
                   std::move(RT));
}

Error ObjectLayer::add(ResourceTrackerSP RT, std::unique_ptr<MemoryBuffer> O) {
  auto I = getObjectFileInterface(getExecutionSession(), O->getMemBufferRef());
  if (!I)
    return I.takeError();
  return add(std::move(RT), std::move(O), std::move(*I));
}

Error ObjectLayer::add(JITDylib &JD, std::unique_ptr<MemoryBuffer> O) {
  auto I = getObjectFileInterface(getExecutionSession(), O->getMemBufferRef());
  if (!I)
    return I.takeError();
  return add(JD, std::move(O), std::move(*I));
}

Expected<std::unique_ptr<BasicObjectLayerMaterializationUnit>>
BasicObjectLayerMaterializationUnit::Create(ObjectLayer &L,
                                            std::unique_ptr<MemoryBuffer> O) {

  auto ObjInterface =
      getObjectFileInterface(L.getExecutionSession(), O->getMemBufferRef());

  if (!ObjInterface)
    return ObjInterface.takeError();

  return std::make_unique<BasicObjectLayerMaterializationUnit>(
      L, std::move(O), std::move(*ObjInterface));
}

BasicObjectLayerMaterializationUnit::BasicObjectLayerMaterializationUnit(
    ObjectLayer &L, std::unique_ptr<MemoryBuffer> O, Interface I)
    : MaterializationUnit(std::move(I)), L(L), O(std::move(O)) {}

StringRef BasicObjectLayerMaterializationUnit::getName() const {
  if (O)
    return O->getBufferIdentifier();
  return "<null object>";
}

void BasicObjectLayerMaterializationUnit::materialize(
    std::unique_ptr<MaterializationResponsibility> R) {
  L.emit(std::move(R), std::move(O));
}

void BasicObjectLayerMaterializationUnit::discard(const JITDylib &JD,
                                                  const SymbolStringPtr &Name) {
  // This is a no-op for object files: Having removed 'Name' from SymbolFlags
  // the symbol will be dead-stripped by the JIT linker.
}

} // End namespace orc.
} // End namespace llvm.
