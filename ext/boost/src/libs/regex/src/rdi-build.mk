include $(RDI_MAKEROOT)/top.mk

RDI_BOOST_SOURCE := yes
include $(SSROOT)/Boost/rdi-boost.mk
MYINCLUDES := $(BOOST_INC_DIR)
MYCFLAGS := $(RDI_BOOST_CFLAGS)

include $(RDI_MAKEROOT)/platform.mk
ifeq ($(RDI_OS),windows)
 MYCFLAGS += -DBOOST_REGEX_DYN_LINK
endif

CPP_SUFFIX := .cpp
include $(RDI_MAKEROOT)/objs.mk

LIBRARY_NAME := boost_regex
LIBRARIES :=
include $(RDI_MAKEROOT)/libs.mk

include $(RDI_MAKEROOT)/bottom.mk

