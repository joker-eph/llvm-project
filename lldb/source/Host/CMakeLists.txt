macro(add_host_subdirectory group)
  list(APPEND HOST_SOURCES ${ARGN})
  source_group(${group} FILES ${ARGN})
endmacro()

add_host_subdirectory(common
  common/File.cpp
  common/FileCache.cpp
  common/FileSpec.cpp
  common/FileSystem.cpp
  common/GetOptInc.cpp
  common/Host.cpp
  common/HostInfoBase.cpp
  common/HostNativeThreadBase.cpp
  common/HostProcess.cpp
  common/HostThread.cpp
  common/IOObject.cpp
  common/LockFileBase.cpp
  common/MonitoringProcessLauncher.cpp
  common/NativeBreakpoint.cpp
  common/NativeBreakpointList.cpp
  common/NativeWatchpointList.cpp
  common/NativeProcessProtocol.cpp
  common/NativeRegisterContext.cpp
  common/NativeRegisterContextRegisterInfo.cpp
  common/NativeThreadProtocol.cpp
  common/OptionParser.cpp
  common/PipeBase.cpp
  common/ProcessRunLock.cpp
  common/Socket.cpp
  common/SocketAddress.cpp
  common/SoftwareBreakpoint.cpp
  common/StringConvert.cpp
  common/Symbols.cpp
  common/TCPSocket.cpp
  common/Terminal.cpp
  common/ThisThread.cpp
  common/ThreadLauncher.cpp
  common/TimeValue.cpp
  common/XML.cpp
  common/UDPSocket.cpp
  )

# Keep track of whether we want to provide a define for the
# Python's architecture-specific lib path (i.e. where a
# Python lldb module would go).
set (get_python_libdir 0)

if (NOT LLDB_DISABLE_LIBEDIT)
  add_host_subdirectory(common
    common/Editline.cpp
    )
endif()

add_host_subdirectory(posix
  posix/ConnectionFileDescriptorPosix.cpp
  )

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  add_host_subdirectory(windows
    windows/ConnectionGenericFileWindows.cpp
    windows/EditLineWin.cpp
    windows/FileSystem.cpp
    windows/Host.cpp
    windows/HostInfoWindows.cpp
    windows/HostProcessWindows.cpp
    windows/HostThreadWindows.cpp
    windows/LockFileWindows.cpp
    windows/PipeWindows.cpp
    windows/ProcessLauncherWindows.cpp
    windows/ProcessRunLock.cpp
    windows/ThisThread.cpp
    windows/Windows.cpp
    )
else()
  if (NOT LLDB_DISABLE_PYTHON)
    # We'll grab the arch-specific python libdir on POSIX systems.
    set (get_python_libdir 1)
  endif()

  add_host_subdirectory(posix
    posix/DomainSocket.cpp
    posix/FileSystem.cpp
    posix/HostInfoPosix.cpp
    posix/HostProcessPosix.cpp
    posix/HostThreadPosix.cpp
    posix/LockFilePosix.cpp
    posix/MainLoopPosix.cpp
    posix/PipePosix.cpp
    )

  if (NOT __ANDROID_NDK__)
    add_host_subdirectory(posix
      posix/ProcessLauncherPosix.cpp
      )
  endif()

  if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    include_directories(SYSTEM ${LIBXML2_INCLUDE_DIR})
    add_host_subdirectory(macosx
      macosx/Host.mm
      macosx/HostInfoMacOSX.mm
      macosx/HostThreadMacOSX.mm
      macosx/Symbols.cpp
      macosx/ThisThread.cpp
      macosx/cfcpp/CFCBundle.cpp
      macosx/cfcpp/CFCData.cpp
      macosx/cfcpp/CFCMutableArray.cpp
      macosx/cfcpp/CFCMutableDictionary.cpp
      macosx/cfcpp/CFCMutableSet.cpp
      macosx/cfcpp/CFCString.cpp
      )

  elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    if (__ANDROID_NDK__)
      add_host_subdirectory(android
        android/HostInfoAndroid.cpp
        android/LibcGlue.cpp
        )
    endif()
    add_host_subdirectory(linux
      linux/AbstractSocket.cpp
      linux/Host.cpp
      linux/HostInfoLinux.cpp
      linux/HostThreadLinux.cpp
      linux/LibcGlue.cpp
      linux/ProcessLauncherLinux.cpp
      linux/ThisThread.cpp
      )

  elseif (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_host_subdirectory(freebsd
      freebsd/Host.cpp
      freebsd/HostInfoFreeBSD.cpp
      freebsd/HostThreadFreeBSD.cpp
      freebsd/ThisThread.cpp
      )

  elseif (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
    add_host_subdirectory(netbsd
      netbsd/Host.cpp
      netbsd/HostInfoNetBSD.cpp
      netbsd/HostThreadNetBSD.cpp
      netbsd/ThisThread.cpp
      )
  endif()
endif()

if (${get_python_libdir})
  # Call a python script to gather the arch-specific libdir for
  # modules like the lldb module.
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/get_relative_lib_dir.py
    RESULT_VARIABLE get_libdir_status
    OUTPUT_VARIABLE relative_libdir
    )
  if (get_libdir_status EQUAL 0)
    add_definitions(-DLLDB_PYTHON_RELATIVE_LIBDIR="${relative_libdir}")
  endif()
endif()

add_lldb_library(lldbHost ${HOST_SOURCES})

if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
target_link_libraries(lldbHost kvm)
endif ()
