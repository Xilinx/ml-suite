include $(RDI_MAKEROOT)/top.mk

include $(RDI_MAKEROOT)/platform.mk
ifeq ($(RDI_OS),windows)
 boost-source := yes
 include $(SSROOT)/Boost/rdi-boost.mk
 MYINCLUDES := $(BOOST_INC_DIR)
 MYCFLAGS := $(RDI_BOOST_CFLAGS)

 CPP_SUFFIX := .cpp
 MYCFLAGS += -DBOOST_THREAD_BUILD_DLL -DBOOST_SYSTEM_NO_DEPRECATED -DBOOST_THREAD_USES_CHRONO
 include $(RDI_MAKEROOT)/objs.mk

 LIBRARY_NAME := boost_thread
 LIBRARIES := ext/boost_chrono ext/boost_system ext/boost_date_time
 include $(RDI_MAKEROOT)/libs.mk
endif

include $(RDI_MAKEROOT)/bottom.mk


