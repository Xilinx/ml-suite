include $(RDI_MAKEROOT)/top.mk
SUBDIR_MAKEFILE_NAME := rdi-build.mk
include $(RDI_MAKEROOT)/subdir.mk

RDI_BOOST_SOURCE := yes
include $(SSROOT)/Boost/rdi-boost.mk
MYINCLUDES := $(BOOST_INC_DIR)
MYCFLAGS := $(RDI_BOOST_CFLAGS)

# We enable zlib and also leave in gzip.cpp which is new in 1.43 (compared to 1.38)
MYCFLAGS += -DBOOST_USE_SEGMENETED_STACKS -DBOOST_COROUTINES_SOURCE -DBOOST_COROUTINES_DYN_LINK
ifeq ($(RDI_OS),gnu/linux)
 MYCFLAGS += -fsplit-stack
 MYFINALLINK := -static-libgcc
endif

CPP_SUFFIX := .cpp
include $(RDI_MAKEROOT)/objs.mk

LIBRARY_NAME := boost_coroutine
SHLIBRARIES := ext/boost_context ext/boost_system ext/boost_thread
include $(RDI_MAKEROOT)/shlib.mk

include $(RDI_MAKEROOT)/bottom.mk

