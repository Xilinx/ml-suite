include $(RDI_MAKEROOT)/top.mk

RDI_BOOST_SOURCE := yes
include $(SSROOT)/Boost/rdi-boost.mk
MYINCLUDES := $(BOOST_INC_DIR)
MYCFLAGS := $(RDI_BOOST_CFLAGS)

CPP_SUFFIX := .cpp
MYCFLAGS += -DBOOST_USE_SEGMENETED_STACKS -DBOOST_COROUTINES_SOURCE -DBOOST_COROUTINES_DYN_LINK
ifeq ($(RDI_OS),gnu/linux)
 MYCFLAGS += -fsplit-stack
endif

include $(RDI_MAKEROOT)/objs.mk

include $(RDI_MAKEROOT)/bottom.mk


