include $(RDI_MAKEROOT)/top.mk

RDI_BOOST_SOURCE := yes
include $(SSROOT)/Boost/rdi-boost.mk
MYINCLUDES := $(BOOST_INC_DIR)
MYCFLAGS := $(RDI_BOOST_CFLAGS)

include $(RDI_MAKEROOT)/platform.mk
ifeq ($(RDI_OS),windows)
 ifeq ($(RDI_BOOST_STATIC_BUILD),yes)
  MYCFLAGS += -DRDI_BOOST_STATIC_BUILD -DBOOST_PROGRAM_OPTIONS_STATIC_LINK
 else
  MYCFLAGS += -DBOOST_PROGRAM_OPTIONS_DYN_LINK
 endif
endif

CPP_SUFFIX := .cpp
include $(RDI_MAKEROOT)/objs.mk

ifneq ($(RDI_BOOST_STATIC_BUILD),yes)
 LIBRARY_NAME := boost_program_options
 LIBRARIES :=
 include $(RDI_MAKEROOT)/shlib.mk
else
 LIBRARY_NAME := boost_program_options_static
 include $(RDI_MAKEROOT)/stlib.mk
endif

include $(RDI_MAKEROOT)/bottom.mk

