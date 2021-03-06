//===-- ConnectionFileDescriptorPosix.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_posix_ConnectionFileDescriptorPosix_h_
#define liblldb_Host_posix_ConnectionFileDescriptorPosix_h_

// C++ Includes
#include <atomic>
#include <memory>
#include <mutex>

#include "lldb/lldb-forward.h"

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Connection.h"
#include "lldb/Host/IOObject.h"
#include "lldb/Host/Pipe.h"
#include "lldb/Host/Predicate.h"

namespace lldb_private {

class Error;
class Socket;
class SocketAddress;

class ConnectionFileDescriptor : public Connection {
public:
  static const char *LISTEN_SCHEME;
  static const char *ACCEPT_SCHEME;
  static const char *UNIX_ACCEPT_SCHEME;
  static const char *CONNECT_SCHEME;
  static const char *TCP_CONNECT_SCHEME;
  static const char *UDP_SCHEME;
  static const char *UNIX_CONNECT_SCHEME;
  static const char *UNIX_ABSTRACT_CONNECT_SCHEME;
  static const char *FD_SCHEME;
  static const char *FILE_SCHEME;

  ConnectionFileDescriptor(bool child_processes_inherit = false);

  ConnectionFileDescriptor(int fd, bool owns_fd);

  ConnectionFileDescriptor(Socket *socket);

  ~ConnectionFileDescriptor() override;

  bool IsConnected() const override;

  lldb::ConnectionStatus Connect(const char *s, Error *error_ptr) override;

  lldb::ConnectionStatus Disconnect(Error *error_ptr) override;

  size_t Read(void *dst, size_t dst_len, uint32_t timeout_usec,
              lldb::ConnectionStatus &status, Error *error_ptr) override;

  size_t Write(const void *src, size_t src_len, lldb::ConnectionStatus &status,
               Error *error_ptr) override;

  std::string GetURI() override;

  lldb::ConnectionStatus BytesAvailable(uint32_t timeout_usec,
                                        Error *error_ptr);

  bool InterruptRead() override;

  lldb::IOObjectSP GetReadObject() override { return m_read_sp; }

  uint16_t GetListeningPort(uint32_t timeout_sec);

  bool GetChildProcessesInherit() const;
  void SetChildProcessesInherit(bool child_processes_inherit);

protected:
  void OpenCommandPipe();

  void CloseCommandPipe();

  lldb::ConnectionStatus SocketListenAndAccept(const char *host_and_port,
                                               Error *error_ptr);

  lldb::ConnectionStatus ConnectTCP(const char *host_and_port,
                                    Error *error_ptr);

  lldb::ConnectionStatus ConnectUDP(const char *args, Error *error_ptr);

  lldb::ConnectionStatus NamedSocketConnect(const char *socket_name,
                                            Error *error_ptr);

  lldb::ConnectionStatus NamedSocketAccept(const char *socket_name,
                                           Error *error_ptr);

  lldb::ConnectionStatus UnixAbstractSocketConnect(const char *socket_name,
                                                   Error *error_ptr);

  lldb::IOObjectSP m_read_sp;
  lldb::IOObjectSP m_write_sp;

  Predicate<uint16_t>
      m_port_predicate; // Used when binding to port zero to wait for the thread
                        // that creates the socket, binds and listens to resolve
                        // the port number.

  Pipe m_pipe;
  std::recursive_mutex m_mutex;
  std::atomic<bool> m_shutting_down; // This marks that we are shutting down so
                                     // if we get woken up from
  // BytesAvailable to disconnect, we won't try to read again.
  bool m_waiting_for_accept;
  bool m_child_processes_inherit;

  std::string m_uri;

private:
  void InitializeSocket(Socket *socket);

  DISALLOW_COPY_AND_ASSIGN(ConnectionFileDescriptor);
};

} // namespace lldb_private

#endif // liblldb_ConnectionFileDescriptor_h_
