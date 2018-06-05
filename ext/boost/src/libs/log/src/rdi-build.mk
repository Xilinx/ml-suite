include $(RDI_MAKEROOT)/top.mk

RDI_BOOST_SOURCE := yes
include $(SSROOT)/Boost/rdi-boost.mk
MYINCLUDES := $(BOOST_INC_DIR)
MYCFLAGS := $(RDI_BOOST_CFLAGS)

BOOST_LOG_COMMON_SRC := $(addprefix $(SSROOT)/$(SSDIR)/,\
 attribute_name.cpp \
 attribute_set.cpp \
 attribute_value_set.cpp \
 code_conversion.cpp \
 core.cpp \
 record_ostream.cpp \
 severity_level.cpp \
 global_logger_storage.cpp \
 named_scope.cpp \
 process_name.cpp \
 process_id.cpp \
 thread_id.cpp \
 timer.cpp \
 exceptions.cpp \
 default_attribute_names.cpp \
 default_sink.cpp \
 text_ostream_backend.cpp \
 text_file_backend.cpp \
 text_multifile_backend.cpp \
 syslog_backend.cpp \
 thread_specific.cpp \
 once_block.cpp \
 timestamp.cpp \
 threadsafe_queue.cpp \
 event.cpp \
 trivial.cpp \
 spirit_encoding.cpp \
 format_parser.cpp \
 date_time_format_parser.cpp \
 named_scope_format_parser.cpp \
 unhandled_exception_count.cpp \
 dump.cpp)

OBJS_EXCLUDE_SRCS := $(filter-out $(BOOST_LOG_COMMON_SRC),$(wildcard $(SSCURDIR)/*.cpp))

# Configuration that affects only compilation of the library
MYCFLAGS += -DBOOST_LOG_WITHOUT_EVENT_LOG
MYCFLAGS += -DBOOST_LOG_WITHOUT_SYSLOG
MYCFLAGS += -DBOOST_LOG_WITHOUT_DEBUG_OUTPUT

# Configuration for shared object, implied by BOOST_ALL_DYN_LINK in user.hpp
MYCFLAGS += -DBOOST_LOG_DYN_LINK -DBOOST_LOG_DLL
MYCFLAGS += -DBOOST_LOG_BUILDING_THE_LIB 
MYCFLAGS += -DBOOST_LOG_SETUP_BUILDING_THE_LIB 

OBJS_EXCLUDE_SRCS := dump_avx2.cpp dump_ssse3.cpp

# Platform specific code
include $(RDI_MAKEROOT)/platform.mk
ifeq ($(RDI_OS),windows)
 OBJS_EXCLUDE_SRCS += 
endif
ifeq ($(RDI_OS),gnu/linux)
 OBJS_EXCLUDE_SRCS += debug_output_backend.cpp light_rw_mutex.cpp
endif

CPP_SUFFIX := .cpp
include $(RDI_MAKEROOT)/objs.mk

LIBRARY_NAME := boost_log
SHLIBRARIES := ext/boost_system ext/boost_filesystem ext/boost_thread ext/boost_regex ext/boost_date_time

ifeq ($(RDI_OS),gnu/linux)
 MYFINALLINK += -lrt
endif

include $(RDI_MAKEROOT)/shlib.mk

include $(RDI_MAKEROOT)/bottom.mk

