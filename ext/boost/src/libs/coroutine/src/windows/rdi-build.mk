include $(RDI_MAKEROOT)/top.mk
include $(RDI_MAKEROOT)/platform.mk

ifeq ($(RDI_OS),windows)
 RDI_BOOST_SOURCE := yes
 include $(SSROOT)/Boost/rdi-boost.mk
 MYINCLUDES := $(BOOST_INC_DIR)
 MYCFLAGS := $(RDI_BOOST_CFLAGS)

 CPP_SUFFIX := .cpp
 MYCFLAGS += -DBOOST_USE_SEGMENETED_STACKS -DBOOST_COROUTINES_SOURCE -DBOOST_COROUTINES_DYN_LINK
 include $(RDI_MAKEROOT)/objs.mk
endif

include $(RDI_MAKEROOT)/bottom.mk


