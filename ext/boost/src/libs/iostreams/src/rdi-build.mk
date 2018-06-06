include $(RDI_MAKEROOT)/top.mk

RDI_BOOST_SOURCE := yes
include $(SSROOT)/Boost/rdi-boost.mk
MYINCLUDES := $(BOOST_INC_DIR)
MYCFLAGS := $(RDI_BOOST_CFLAGS)

CPP_SUFFIX := .cpp

# We enable zlib and also leave in gzip.cpp which is new in 1.43 (compared to 1.38)
MYCFLAGS += -DNO_BZIP2 -DNO_COMPRESSION
ifeq ($(RDI_OS),windows)
 MYCFLAGS += -DBOOST_IOSTREAMS_DYN_LINK
endif
OBJS_EXCLUDE_SRCS := bzip2.cpp

include $(RDI_MAKEROOT)/objs.mk

LIBRARY_NAME := boost_iostreams
SHLIBRARIES := ext/rdizlib
include $(RDI_MAKEROOT)/libs.mk

include $(RDI_MAKEROOT)/bottom.mk

