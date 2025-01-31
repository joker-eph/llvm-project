if (NOT HAVE_CXX_ATOMICS64_WITHOUT_LIB)
  set (atomic_lib atomic)
endif()

if( CMAKE_HOST_UNIX AND HAVE_LIBRT )
  set(rt_lib rt)
endif()

add_llvm_component_library(LLVMOrcJITLight
  AbsoluteSymbols.cpp
  Core.cpp
  ExecutorProcessControl.cpp
  ObjectLayer.cpp
  LinkGraphLayer.cpp
  LinkGraphLinkingLayer.cpp
  LoadLinkableFile.cpp
  ObjectFileInterface.cpp
  ObjectLinkingLayer.cpp
  ObjectTransformLayer.cpp
  RTDyldObjectLinkingLayer.cpp
  TaskDispatch.cpp
  ThreadSafeModule.cpp
  DebugUtils.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  intrinsics_gen

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}
  ${rt_lib}
  ${atomic_lib}

  LINK_COMPONENTS
  BinaryFormat
  ExecutionEngine
  JITLink
  Object
  OrcShared
  OrcTargetProcess
  RuntimeDyld
  Support
)

add_llvm_component_library(LLVMOrcJIT
  COFFVCRuntimeSupport.cpp
  COFFPlatform.cpp
  CompileOnDemandLayer.cpp
  CompileUtils.cpp
  Core.cpp
  DebugObjectManagerPlugin.cpp
  EHFrameRegistrationPlugin.cpp
  EPCDynamicLibrarySearchGenerator.cpp
  EPCDebugObjectRegistrar.cpp
  EPCEHFrameRegistrar.cpp
  EPCGenericDylibManager.cpp
  EPCGenericJITLinkMemoryManager.cpp
  EPCGenericRTDyldMemoryManager.cpp
  EPCIndirectionUtils.cpp
  ExecutionUtils.cpp
  IndirectionUtils.cpp
  IRCompileLayer.cpp
  IRTransformLayer.cpp
  IRPartitionLayer.cpp
  JITTargetMachineBuilder.cpp
  JITLinkReentryTrampolines.cpp
  Layer.cpp
  LazyObjectLinkingLayer.cpp
  LazyReexports.cpp
  LookupAndRecordAddrs.cpp
  LLJIT.cpp
  MachO.cpp
  MachOPlatform.cpp
  MapperJITLinkMemoryManager.cpp
  MemoryMapper.cpp
  ELFNixPlatform.cpp
  Mangling.cpp
  OrcABISupport.cpp
  OrcV2CBindings.cpp
  SectCreate.cpp
  SimpleRemoteEPC.cpp
  Speculation.cpp
  SpeculateAnalyses.cpp
  RedirectionManager.cpp
  JITLinkRedirectableSymbolManager.cpp
  ReOptimizeLayer.cpp
  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/Orc


  PARTIAL_SOURCES_INTENDED
  DEPENDS
  intrinsics_gen

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}
  ${rt_lib}
  ${atomic_lib}

  LINK_COMPONENTS
  BinaryFormat
  Core
  ExecutionEngine
  JITLink
  Object
  OrcJITLight
  OrcShared
  OrcTargetProcess
  WindowsDriver
  MC
  Passes
  RuntimeDyld
  Support
  Target
  TargetParser
  TransformUtils
  )

add_subdirectory(Debugging)
add_subdirectory(Shared)
add_subdirectory(TargetProcess)

target_link_libraries(LLVMOrcJIT
  PRIVATE
  LLVMAnalysis
  LLVMBitReader
  LLVMBitWriter
  LLVMPasses
  )
