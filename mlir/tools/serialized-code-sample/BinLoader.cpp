
#include "llvm/Support/ErrorHandling.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>
#include <memory>
#include <sstream>

using namespace llvm;

ExitOnError ExitOnErr;

class LoadedObjectFile {
public:
  LoadedObjectFile(StringRef fileName);
  ~LoadedObjectFile() { cantFail(ES->endSession()); }

  template <typename T>
  auto lookup(StringRef symbolName) const {
    orc::ExecutorAddr func =
        cantFail(ES->lookup(ArrayRef{&*dylib}, symbolName)).getAddress();
    return func.toPtr<T>();
  }

private:
  std::unique_ptr<orc::ExecutionSession> ES;
  std::unique_ptr<orc::ObjectLinkingLayer> ObjLinkingLayer;
  std::unique_ptr<orc::ObjectTransformLayer> ObjTransformLayer;
  orc::JITDylib *dylib;
};

LoadedObjectFile::LoadedObjectFile(StringRef fileName) {
  if (auto EPC = orc::SelfExecutorProcessControl::Create()) {
    ES = std::make_unique<orc::ExecutionSession>(std::move(*EPC));
  } else {
    llvm::report_fatal_error("failed to create ExecutionSession\n");
  }
  auto maybeDylib = ES->createJITDylib(fileName.str());
  if (!maybeDylib) {
    llvm::report_fatal_error("failed to create dylib\n");
    exit(1);
  }
  dylib = &*maybeDylib;

  ObjLinkingLayer = std::make_unique<orc::ObjectLinkingLayer>(*ES);

  ObjTransformLayer =
      std::make_unique<orc::ObjectTransformLayer>(*ES, *ObjLinkingLayer);
  errs() << "======= Initial dylib =====\n";
  dylib->dump(outs());

  errs() << "======= Add Object =====\n";
  orc::SymbolMap symMap;
  symMap[ES->intern("puts")] = {orc::ExecutorAddr::fromPtr(puts),
                                JITSymbolFlags()};
  ;
  cantFail(dylib->define(orc::absoluteSymbols(symMap)));
  auto obj = ExitOnErr(errorOrToExpected(MemoryBuffer::getFile(fileName)));
  cantFail(ObjTransformLayer->add(*dylib, std::move(obj)));

  dylib->dump(outs());
}

int main(int argc, char **argv) {
  if (argc < 2)
    report_fatal_error("requires the file to load as first command line arg");
  LoadedObjectFile file(argv[1]);

  file.lookup<void (*)()>("_Z3foov")();

  return 0;
}
