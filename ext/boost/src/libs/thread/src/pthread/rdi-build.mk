include $(RDI_MAKEROOT)/top.mk

include $(RDI_MAKEROOT)/platform.mk
ifeq ($(RDI_OS),gnu/linux)
 boost-source := yes
 include $(SSROOT)/Boost/rdi-boost.mk
 MYINCLUDES := $(BOOST_INC_DIR)
 MYCFLAGS := $(RDI_BOOST_CFLAGS)

 CPP_SUFFIX := .cpp
 MYCFLAGS += -DBOOST_THREAD_POSIX -DBOOST_SYSTEM_NO_DEPRECATED -DBOOST_THREAD_USES_CHRONO -Wno-long-long
 OBJS_EXCLUDE_SRCS := once_atomic.cpp
 include $(RDI_MAKEROOT)/objs.mk

 LIBRARY_NAME := boost_thread
 LIBRARIES := ext/boost_chrono ext/boost_system 
 ifeq ($(RDI_OS),gnu/linux)
  MYLDLIBS := -lrt
 endif
 include $(RDI_MAKEROOT)/shlib.mk
endif

include $(RDI_MAKEROOT)/bottom.mk
