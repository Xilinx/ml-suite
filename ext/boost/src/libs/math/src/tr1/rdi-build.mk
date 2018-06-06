include $(RDI_MAKEROOT)/top.mk

include $(SSROOT)/Boost/rdi-boost.mk
MYINCLUDES := $(BOOST_INC_DIR) $(SSCURDIR)

CPP_SUFFIX := .cpp

ifeq ($(RDI_OS),windows)
 MYCFLAGS += -DBOOST_MATH_TR1_DYN_LINK
endif

include $(RDI_MAKEROOT)/objs.mk

include $(RDI_MAKEROOT)/bottom.mk

