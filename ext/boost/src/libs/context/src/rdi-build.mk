include $(RDI_MAKEROOT)/top.mk

include $(RDI_MAKEROOT)/platform.mk

RDI_BOOST_SOURCE := yes
include $(SSROOT)/Boost/rdi-boost.mk

CPP_SUFFIX := .cpp
OBJS_EXCLUDE_SRCS := unsupported.cpp untested.cpp

ifeq ($(RDI_OS),windows)
 MYCFLAGS := -DBOOST_CONTEXT_DYN_LINK -DBOOST_CONTEXT_SOURCE
endif 
include $(RDI_MAKEROOT)/objs.mk

################################################################
# Most of the "assembler" logic belongs in rdi/makefiles but
# since assembly files are not used elsewhere and since boost
# is pre-compiled, it may make sense to leave this stuff here...
################################################################
ifneq ($(filter lnx32 win32,$(RDI_PLATFORM)),)
 CONTEXT_MIDDLE := i386
 PLATFORM_LNX_BITS := 32
 PLATFORM_WIN_BITS :=
else
 CONTEXT_MIDDLE := x86_64
 PLATFORM_LNX_BITS := 64
 PLATFORM_WIN_BITS := 64
endif

ifeq ($(RDI_OS),windows)
 CONTEXT_END := _ms_pe_masm
 ASSEMBLYEXTN := .asm
 ASSEMBLYCMD := ml$(PLATFORM_WIN_BITS) /DBOOST_CONTEXT_EXPORT=EXPORT /c /Fo 
endif
ifeq ($(RDI_OS)/$(RDI_OS_FLAVOR),gnu/linux/x86)
 CONTEXT_END := _sysv_elf_gas
 ASSEMBLYEXTN := .S
 ASSEMBLYCMD := as --$(PLATFORM_LNX_BITS) -o
endif
ifeq ($(RDI_OS_FLAVOR),arm)
 $(error todo)
endif

define ASSEMBLY_template
 OBJNAME_$1 := $(OBJS_TARGET_DIR)/$1$(CONTEXT_MIDDLE)$(CONTEXT_END)$(OBJEXTN)

 $$(OBJNAME_$1): $(_OBJS_SRC_DIR)/asm/$1$(CONTEXT_MIDDLE)$(CONTEXT_END)$(ASSEMBLYEXTN)
	$(ASSEMBLYCMD) $$@ $$<

 OBJS_$(SSDIR)_$(RDI_PLATFORM) += $$(OBJNAME_$1)

 objs: $$(OBJNAME_$1)
endef

$(eval $(call ASSEMBLY_template,jump_))
$(eval $(call ASSEMBLY_template,make_))

LIBRARY_NAME := boost_context
include $(RDI_MAKEROOT)/shlib.mk
include $(RDI_MAKEROOT)/bottom.mk
