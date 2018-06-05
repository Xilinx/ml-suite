include $(RDI_MAKEROOT)/top.mk

boost-source := yes
include $(SSROOT)/Boost/rdi-boost.mk
MYINCLUDES := $(BOOST_INC_DIR)
MYCFLAGS := $(RDI_BOOST_CFLAGS)

CPP_SUFFIX := .cpp
ifeq ($(RDI_OS),windows)
 MYCFLAGS += -DBOOST_CHRONO_DYN_LINK
endif
include $(RDI_MAKEROOT)/objs.mk

LIBRARY_NAME := boost_chrono
LIBRARIES := ext/boost_system
ifeq ($(RDI_OS),gnu/linux)
 MYLDLIBS := -lrt -lpthread
endif
include $(RDI_MAKEROOT)/shlib.mk

include $(RDI_MAKEROOT)/bottom.mk
