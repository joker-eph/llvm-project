# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe
# RUN: llvm-readobj -file-headers -sections -program-headers -symbols %t.exe \
# RUN:   | FileCheck %s

# REQUIRES: mips

# Exits with return code 1 on Linux.
        .globl  __start
__start:
        li      $a0,1
        li      $v0,4001
        syscall

# CHECK:      ElfHeader {
# CHECK-NEXT:   Ident {
# CHECK-NEXT:     Magic: (7F 45 4C 46)
# CHECK-NEXT:     Class: 32-bit (0x1)
# CHECK-NEXT:     DataEncoding: LittleEndian (0x1)
# CHECK-NEXT:     FileVersion: 1
# CHECK-NEXT:     OS/ABI: SystemV (0x0)
# CHECK-NEXT:     ABIVersion: 0
# CHECK-NEXT:     Unused: (00 00 00 00 00 00 00)
# CHECK-NEXT:   }
# CHECK-NEXT:   Type: Executable (0x2)
# CHECK-NEXT:   Machine: EM_MIPS (0x8)
# CHECK-NEXT:   Version: 1
# CHECK-NEXT:   Entry: 0x20000
# CHECK-NEXT:   ProgramHeaderOffset: 0x34
# CHECK-NEXT:   SectionHeaderOffset: 0x30088
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     EF_MIPS_ABI_O32
# CHECK-NEXT:     EF_MIPS_ARCH_32
# CHECK-NEXT:     EF_MIPS_CPIC
# CHECK-NEXT:   ]
# CHECK-NEXT:   HeaderSize: 52
# CHECK-NEXT:   ProgramHeaderEntrySize: 32
# CHECK-NEXT:   ProgramHeaderCount: 6
# CHECK-NEXT:   SectionHeaderEntrySize: 40
# CHECK-NEXT:   SectionHeaderCount: 10
# CHECK-NEXT:   StringTableSectionIndex: 8
# CHECK-NEXT: }
# CHECK-NEXT: Sections [
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 0
# CHECK-NEXT:     Name:  (0)
# CHECK-NEXT:     Type: SHT_NULL (0x0)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 0
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 1
# CHECK-NEXT:     Name: .reginfo
# CHECK-NEXT:     Type: SHT_MIPS_REGINFO (0x70000006)
# CHECK-NEXT:     Flags [ (0x2)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x100F4
# CHECK-NEXT:     Offset: 0xF4
# CHECK-NEXT:     Size: 24
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 4
# CHECK-NEXT:     EntrySize: 24
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 2
# CHECK-NEXT:     Name: .MIPS.abiflags
# CHECK-NEXT:     Type: SHT_MIPS_ABIFLAGS (0x7000002A)
# CHECK-NEXT:     Flags [ (0x2)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x10110
# CHECK-NEXT:     Offset: 0x110
# CHECK-NEXT:     Size: 24
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 8
# CHECK-NEXT:     EntrySize: 24
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 3
# CHECK-NEXT:     Name: .text
# CHECK-NEXT:     Type: SHT_PROGBITS (0x1)
# CHECK-NEXT:     Flags [ (0x6)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_EXECINSTR (0x4)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x20000
# CHECK-NEXT:     Offset: 0x10000
# CHECK-NEXT:     Size: 12
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 16
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 4
# CHECK-NEXT:     Name: .got
# CHECK-NEXT:     Type: SHT_PROGBITS (0x1)
# CHECK-NEXT:     Flags [ (0x10000003)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_MIPS_GPREL (0x10000000)
# CHECK-NEXT:       SHF_WRITE (0x1)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x30000
# CHECK-NEXT:     Offset: 0x20000
# CHECK-NEXT:     Size: 8
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 4
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 5
# CHECK-NEXT:     Name: .data
# CHECK-NEXT:     Type: SHT_PROGBITS (0x1)
# CHECK-NEXT:     Flags [ (0x3)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_WRITE (0x1)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x40000
# CHECK-NEXT:     Offset: 0x30000
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 16
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 6
# CHECK-NEXT:     Name: .bss
# CHECK-NEXT:     Type: SHT_NOBITS (0x8)
# CHECK-NEXT:     Flags [ (0x3)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_WRITE (0x1)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x40000
# CHECK-NEXT:     Offset: 0x30000
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 16
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 7
# CHECK-NEXT:     Name: .symtab
# CHECK-NEXT:     Type: SHT_SYMTAB (0x2)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x30000
# CHECK-NEXT:     Size: 48
# CHECK-NEXT:     Link: 9
# CHECK-NEXT:     Info: 1
# CHECK-NEXT:     AddressAlignment: 4
# CHECK-NEXT:     EntrySize: 16
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 8
# CHECK-NEXT:     Name: .shstrtab
# CHECK-NEXT:     Type: SHT_STRTAB (0x3)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x30030
# CHECK-NEXT:     Size: 73
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 9
# CHECK-NEXT:     Name: .strtab
# CHECK-NEXT:     Type: SHT_STRTAB (0x3)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x30079
# CHECK-NEXT:     Size: 13
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT: ]
# CHECK-NEXT: Symbols [
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name:  (0)
# CHECK-NEXT:     Value: 0x0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Local (0x0)
# CHECK-NEXT:     Type: None (0x0)
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: Undefined (0x0)
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: _gp
# CHECK-NEXT:     Value: 0x37FF0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Local
# CHECK-NEXT:     Type: None (0x0)
# CHECK-NEXT:     Other [ (0x2)
# CHECK-NEXT:       STV_HIDDEN (0x2)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Section: .got
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: __start
# CHECK-NEXT:     Value: 0x20000
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Global (0x1)
# CHECK-NEXT:     Type: None (0x0)
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: .text
# CHECK-NEXT:   }
# CHECK-NEXT: ]
# CHECK-NEXT: ProgramHeaders [
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_PHDR (0x6)
# CHECK-NEXT:     Offset: 0x34
# CHECK-NEXT:     VirtualAddress: 0x10034
# CHECK-NEXT:     PhysicalAddress: 0x10034
# CHECK-NEXT:     FileSize: 192
# CHECK-NEXT:     MemSize: 192
# CHECK-NEXT:     Flags [ (0x4)
# CHECK-NEXT:       PF_R (0x4)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment: 4
# CHECK-NEXT:   }
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_LOAD (0x1)
# CHECK-NEXT:     Offset: 0x0
# CHECK-NEXT:     VirtualAddress: 0x10000
# CHECK-NEXT:     PhysicalAddress: 0x10000
# CHECK-NEXT:     FileSize: 296
# CHECK-NEXT:     MemSize: 296
# CHECK-NEXT:     Flags [ (0x4)
# CHECK-NEXT:       PF_R (0x4)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment: 65536
# CHECK-NEXT:   }
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_LOAD (0x1)
# CHECK-NEXT:     Offset: 0x10000
# CHECK-NEXT:     VirtualAddress: 0x20000
# CHECK-NEXT:     PhysicalAddress: 0x20000
# CHECK-NEXT:     FileSize: 12
# CHECK-NEXT:     MemSize: 12
# CHECK-NEXT:     Flags [ (0x5)
# CHECK-NEXT:       PF_R (0x4)
# CHECK-NEXT:       PF_X (0x1)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment: 65536
# CHECK-NEXT:   }
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_LOAD (0x1)
# CHECK-NEXT:     Offset: 0x20000
# CHECK-NEXT:     VirtualAddress: 0x30000
# CHECK-NEXT:     PhysicalAddress: 0x30000
# CHECK-NEXT:     FileSize: 65536
# CHECK-NEXT:     MemSize: 65536
# CHECK-NEXT:     Flags [
# CHECK-NEXT:       PF_R
# CHECK-NEXT:       PF_W
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment: 65536
# CHECK-NEXT:   }
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_GNU_RELRO (0x6474E552)
# CHECK-NEXT:     Offset: 0x20000
# CHECK-NEXT:     VirtualAddress: 0x30000
# CHECK-NEXT:     PhysicalAddress: 0x30000
# CHECK-NEXT:     FileSize: 8
# CHECK-NEXT:     MemSize: 8
# CHECK-NEXT:     Flags [ (0x4)
# CHECK-NEXT:       PF_R (0x4)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment: 1
# CHECK-NEXT:   }
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:    Type: PT_GNU_STACK
# CHECK-NEXT:    Offset: 0x0
# CHECK-NEXT:    VirtualAddress: 0x0
# CHECK-NEXT:    PhysicalAddress: 0x0
# CHECK-NEXT:    FileSize: 0
# CHECK-NEXT:    MemSize: 0
# CHECK-NEXT:    Flags [
# CHECK-NEXT:      PF_R
# CHECK-NEXT:      PF_W
# CHECK-NEXT:    ]
# CHECK-NEXT:    Alignment: 0
# CHECK-NEXT:  }
# CHECK-NEXT:]
