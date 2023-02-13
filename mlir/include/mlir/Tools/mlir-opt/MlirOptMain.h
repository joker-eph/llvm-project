//===- MlirOptMain.h - MLIR Optimizer Driver main ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIROPT_MLIROPTMAIN_H
#define MLIR_TOOLS_MLIROPT_MLIROPTMAIN_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <memory>

namespace llvm {
class raw_ostream;
class MemoryBuffer;
} // namespace llvm

namespace mlir {
class DialectRegistry;
class PassPipelineCLParser;
class PassManager;

/// Configuration options for the mlir-opt tool.
class MlirOptMainConfig {
public:
  /// Set whether to split the input file based on the `// -----` marker into
  /// pieces and process each chunk independently.
  MlirOptMainConfig &setSplitInputFile(bool split) {
    splitInputFile = split;
    return *this;
  }
  bool shouldSplitInputFile() const { return splitInputFile; }

  /// Set whether to check that emitted diagnostics match `expected-*` lines on
  /// the corresponding line
  MlirOptMainConfig &setVerifyDiagnostics(bool verify) {
    verifyDiagnostics = verify;
    return *this;
  }
  bool shouldVerifyDiagnostics() const { return verifyDiagnostics; }

  /// Set whether to run the verifier after each transformation pass.
  MlirOptMainConfig &setVerifyPasses(bool verify) {
    verifyPasses = verify;
    return *this;
  }
  bool shouldVerifyPasses() const { return verifyPasses; }

  /// Set whether to run the verifier after each transformation pass.
  MlirOptMainConfig &setAllowUnregisteredDialects(bool allow) {
    allowUnregisteredDialects = allow;
    return *this;
  }
  bool shouldAllowUnregisteredDialects() const {
    return allowUnregisteredDialects;
  }

  /// Set whether to run the verifier after each transformation pass.
  MlirOptMainConfig &setShowDialects(bool show) {
    showDialects = show;
    return *this;
  }
  bool shouldShowDialects() const { return showDialects; }

  /// Set whether to run the verifier after each transformation pass.
  MlirOptMainConfig &setEmitBytecode(bool emit) {
    emitBytecode = emit;
    return *this;
  }
  bool shouldEmitBytecode() const { return emitBytecode; }

  /// Set whether to run the verifier after each transformation pass.
  MlirOptMainConfig &setUseImplicitModule(bool useImplicitModule) {
    this->useImplicitModule = useImplicitModule;
    return *this;
  }
  bool shouldUseImplicitModule() const { return useImplicitModule; }

  /// Set whether to run the verifier after each transformation pass.
  MlirOptMainConfig &setDumpPassPipeline(bool dump) {
    dumpPassPipeline = dump;
    return *this;
  }
  bool shouldDumpPassPipeline() const { return dumpPassPipeline; }

  /// Set the callback to populate the pass manager.
  MlirOptMainConfig &
  setPassPipelineSetupFn(std::function<LogicalResult(PassManager &)> callback) {
    passPipelineCallback = std::move(callback);
    return *this;
  }

  MlirOptMainConfig &setPassPipelineParser(const PassPipelineCLParser &parser);

  /// Populate the passmanager, if any callback was set.
  LogicalResult setupPassPipeline(PassManager &pm) const {
    if (passPipelineCallback)
      return passPipelineCallback(pm);
    return success();
  }

  /// Set the filename to use for logging actions, use "-" for stdout.
  MlirOptMainConfig &setLogActionsTo(StringRef filename) {
    logActionsTo = filename;
    return *this;
  }

  /// Get the filename to use for logging actions.
  StringRef getLogActionsTo() const { return logActionsTo; }

  /// Deprecated.
  MlirOptMainConfig &setPreloadDialectsInContext(bool preload) {
    preloadDialectsInContext = preload;
    return *this;
  }

  /// Deprecated.
  bool shouldPreloadDialectsInContext() const {
    return preloadDialectsInContext;
  }

private:
  /// Input .mlir or .mlirbc filename for the mlir-opt tool.
  std::string inputFilename = "-";

  /// Output .mlir or .mlirbc filename for the mlir-opt tool.
  std::string outputFilename = "-";

  /// Split the input file based on the `// -----` marker into pieces and
  /// process each chunk independently.
  bool splitInputFile = false;

  /// Check that emitted diagnostics match `expected-*` lines on the
  /// corresponding line
  bool verifyDiagnostics = false;

  /// Run the verifier after each transformation pass.
  bool verifyPasses = true;

  /// Allow operation with no registered dialects.
  bool allowUnregisteredDialects = false;

  /// Print the list of registered dialects.
  bool showDialects = false;

  /// Emit bytecode instead of textual assembly when generating output.
  bool emitBytecode = false;

  /// Use an implicit top-level module op during parsing.
  bool useImplicitModule = false;

  /// Deprecated.
  bool preloadDialectsInContext = false;

  /// Print the pipeline that will be run.
  bool dumpPassPipeline = false;

  /// Log action execution to the given file (or "-" for stdout)
  std::string logActionsTo;

  /// The callback to populate the pass manager.
  std::function<LogicalResult(PassManager &)> passPipelineCallback;
};

/// This defines the function type used to setup the pass manager. This can be
/// used to pass in a callback to setup a default pass pipeline to be applied on
/// the loaded IR.
using PassPipelineFn = llvm::function_ref<LogicalResult(PassManager &pm)>;

/// Perform the core processing behind `mlir-opt`.
/// - outputStream is the stream where the resulting IR is printed.
/// - buffer is the in-memory file to parser and process.
/// - registry should contain all the dialects that can be parsed in the source.
/// - config contains the configuration options for the tool.
LogicalResult MlirOptMain(llvm::raw_ostream &outputStream,
                          std::unique_ptr<llvm::MemoryBuffer> buffer,
                          DialectRegistry &registry,
                          const MlirOptMainConfig &config);

/// Perform the core processing behind `mlir-opt`.
/// This API is deprecated, use the MlirOptMainConfig version above instead.
LogicalResult
MlirOptMain(llvm::raw_ostream &outputStream,
            std::unique_ptr<llvm::MemoryBuffer> buffer,
            const PassPipelineCLParser &passPipeline, DialectRegistry &registry,
            bool splitInputFile, bool verifyDiagnostics, bool verifyPasses,
            bool allowUnregisteredDialects,
            bool preloadDialectsInContext = false, bool emitBytecode = false,
            bool implicitModule = false, bool dumpPassPipeline = false);

/// Perform the core processing behind `mlir-opt`.
/// This API is deprecated, use the MlirOptMainConfig version above instead.
LogicalResult MlirOptMain(
    llvm::raw_ostream &outputStream, std::unique_ptr<llvm::MemoryBuffer> buffer,
    PassPipelineFn passManagerSetupFn, DialectRegistry &registry,
    bool splitInputFile, bool verifyDiagnostics, bool verifyPasses,
    bool allowUnregisteredDialects, bool preloadDialectsInContext = false,
    bool emitBytecode = false, bool implicitModule = false);

/// Implementation for tools like `mlir-opt`.
/// - toolName is used for the header displayed by `--help`.
/// - registry should contain all the dialects that can be parsed in the source.
/// - preloadDialectsInContext will trigger the upfront loading of all
///   dialects from the global registry in the MLIRContext. This option is
///   deprecated and will be removed soon.
LogicalResult MlirOptMain(int argc, char **argv, llvm::StringRef toolName,
                          DialectRegistry &registry,
                          bool preloadDialectsInContext = false);

/// Helper wrapper to return the result of MlirOptMain directly from main.
///
/// Example:
///
///     int main(int argc, char **argv) {
///       // ...
///       return mlir::asMainReturnCode(mlir::MlirOptMain(
///           argc, argv, /* ... */);
///     }
///
inline int asMainReturnCode(LogicalResult r) {
  return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace mlir

#endif // MLIR_TOOLS_MLIROPT_MLIROPTMAIN_H
