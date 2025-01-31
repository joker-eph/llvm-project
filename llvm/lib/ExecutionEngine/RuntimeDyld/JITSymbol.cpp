//===----------- JITSymbol.cpp - JITSymbol class implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// JITSymbol class implementation plus helper functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/Object/ObjectFile.h"

using namespace llvm;

Expected<JITSymbolFlags>
llvm::JITSymbolFlags::fromObjectSymbol(const object::SymbolRef &Symbol) {
  Expected<uint32_t> SymbolFlagsOrErr = Symbol.getFlags();
  if (!SymbolFlagsOrErr)
    // TODO: Test this error.
    return SymbolFlagsOrErr.takeError();

  JITSymbolFlags Flags = JITSymbolFlags::None;
  if (*SymbolFlagsOrErr & object::BasicSymbolRef::SF_Weak)
    Flags |= JITSymbolFlags::Weak;
  if (*SymbolFlagsOrErr & object::BasicSymbolRef::SF_Common)
    Flags |= JITSymbolFlags::Common;
  if (*SymbolFlagsOrErr & object::BasicSymbolRef::SF_Exported)
    Flags |= JITSymbolFlags::Exported;

  auto SymbolType = Symbol.getType();
  if (!SymbolType)
    return SymbolType.takeError();

  if (*SymbolType == object::SymbolRef::ST_Function)
    Flags |= JITSymbolFlags::Callable;

  return Flags;
}

ARMJITSymbolFlags
llvm::ARMJITSymbolFlags::fromObjectSymbol(const object::SymbolRef &Symbol) {
  Expected<uint32_t> SymbolFlagsOrErr = Symbol.getFlags();
  if (!SymbolFlagsOrErr)
    // TODO: Actually report errors helpfully.
    report_fatal_error(SymbolFlagsOrErr.takeError());
  ARMJITSymbolFlags Flags;
  if (*SymbolFlagsOrErr & object::BasicSymbolRef::SF_Thumb)
    Flags |= ARMJITSymbolFlags::Thumb;
  return Flags;
}

/// Performs lookup by, for each symbol, first calling
///        findSymbolInLogicalDylib and if that fails calling
///        findSymbol.
void LegacyJITSymbolResolver::lookup(const LookupSet &Symbols,
                                     OnResolvedFunction OnResolved) {
  JITSymbolResolver::LookupResult Result;
  for (auto &Symbol : Symbols) {
    std::string SymName = Symbol.str();
    if (auto Sym = findSymbolInLogicalDylib(SymName)) {
      if (auto AddrOrErr = Sym.getAddress())
        Result[Symbol] = JITEvaluatedSymbol(*AddrOrErr, Sym.getFlags());
      else {
        OnResolved(AddrOrErr.takeError());
        return;
      }
    } else if (auto Err = Sym.takeError()) {
      OnResolved(std::move(Err));
      return;
    } else {
      // findSymbolInLogicalDylib failed. Lets try findSymbol.
      if (auto Sym = findSymbol(SymName)) {
        if (auto AddrOrErr = Sym.getAddress())
          Result[Symbol] = JITEvaluatedSymbol(*AddrOrErr, Sym.getFlags());
        else {
          OnResolved(AddrOrErr.takeError());
          return;
        }
      } else if (auto Err = Sym.takeError()) {
        OnResolved(std::move(Err));
        return;
      } else {
        OnResolved(make_error<StringError>("Symbol not found: " + Symbol,
                                           inconvertibleErrorCode()));
        return;
      }
    }
  }

  OnResolved(std::move(Result));
}

/// Performs flags lookup by calling findSymbolInLogicalDylib and
///        returning the flags value for that symbol.
Expected<JITSymbolResolver::LookupSet>
LegacyJITSymbolResolver::getResponsibilitySet(const LookupSet &Symbols) {
  JITSymbolResolver::LookupSet Result;

  for (auto &Symbol : Symbols) {
    std::string SymName = Symbol.str();
    if (auto Sym = findSymbolInLogicalDylib(SymName)) {
      // If there's an existing def but it is not strong, then the caller is
      // responsible for it.
      if (!Sym.getFlags().isStrong())
        Result.insert(Symbol);
    } else if (auto Err = Sym.takeError())
      return std::move(Err);
    else {
      // If there is no existing definition then the caller is responsible for
      // it.
      Result.insert(Symbol);
    }
  }

  return std::move(Result);
}
