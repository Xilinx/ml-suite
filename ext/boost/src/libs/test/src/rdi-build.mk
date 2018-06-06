include $(RDI_MAKEROOT)/top.mk

RDI_BOOST_SOURCE := yes
include $(SSROOT)/Boost/rdi-boost.mk

TOOLSET_HOST := lnx64
TOOLSET_TARGETS := aarch64 arm64 ppc64le
include $(RDI_MAKEROOT)/toolset.mk

MYINCLUDES := $(BOOST_INC_DIR)
MYCFLAGS := $(RDI_BOOST_CFLAGS)

MYCFLAGS += -DBOOST_TEST_NO_AUTO_LINK -DBOOST_TEST_DYN_LINK
CPP_SUFFIX := .cpp
OBJS_EXCLUDE_SRCS := cpp_main.cpp test_main.cpp
include $(RDI_MAKEROOT)/objs.mk

ifneq ($(RDI_BOOST_STATIC_BUILD),yes)
 LIBRARY_NAME := boost_unit_test_framework
 SHLIBRARIES :=
 include $(RDI_MAKEROOT)/shlib.mk
else
 LIBRARY_NAME := boost_unit_test_framework_static
 include $(RDI_MAKEROOT)/stlib.mk
endif


include $(RDI_MAKEROOT)/bottom.mk

