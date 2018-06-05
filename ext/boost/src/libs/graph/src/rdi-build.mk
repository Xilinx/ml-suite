include $(RDI_MAKEROOT)/top.mk

boost-source := yes
include $(SSROOT)/Boost/rdi-boost.mk
MYINCLUDES := $(BOOST_INC_DIR)
MYCFLAGS := $(RDI_BOOST_CFLAGS)

CPP_SUFFIX := .cpp
ifeq ($(RDI_OS),windows)
 MYCFLAGS += -DBOOST_GRAPH_DYN_LINK
endif
include $(RDI_MAKEROOT)/objs.mk

LIBRARY_NAME := boost_graph
LIBRARIES := ext/boost_regex
include $(RDI_MAKEROOT)/shlib.mk

include $(RDI_MAKEROOT)/bottom.mk
