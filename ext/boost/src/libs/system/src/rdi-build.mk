include $(RDI_MAKEROOT)/top.mk

RDI_BOOST_SOURCE := yes
include $(SSROOT)/Boost/rdi-boost.mk

TOOLSET_HOST := lnx64
TOOLSET_TARGETS := aarch64 arm64 ppc64le
include $(RDI_MAKEROOT)/toolset.mk

MYINCLUDES := $(BOOST_INC_DIR)
MYCFLAGS := $(RDI_BOOST_CFLAGS)

include $(RDI_MAKEROOT)/platform.mk
ifeq ($(RDI_OS),windows)
 ifeq ($(RDI_BOOST_STATIC_BUILD),yes)
  MYCFLAGS += -DRDI_BOOST_STATIC_BUILD -DBOOST_SYSTEM_STATIC_LINK
 else
  MYCFLAGS += -DBOOST_SYSTEM_DYN_LINK
 endif
endif

CPP_SUFFIX := .cpp
include $(RDI_MAKEROOT)/objs.mk

ifneq ($(RDI_BOOST_STATIC_BUILD),yes)
 LIBRARY_NAME := boost_system
 LIBRARIES :=
 include $(RDI_MAKEROOT)/shlib.mk
else
 LIBRARY_NAME := boost_system_static
 include $(RDI_MAKEROOT)/stlib.mk
endif

include $(RDI_MAKEROOT)/bottom.mk
