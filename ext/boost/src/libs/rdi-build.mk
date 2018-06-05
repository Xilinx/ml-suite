include $(RDI_MAKEROOT)/top.mk

RDI_BOOST_SOURCE := yes
include $(SSROOT)/Boost/rdi-boost.mk

SUBDIR_EXCLUDE := context coroutine

ifeq ($(RDI_BOOST_STATIC_BUILD),yes)
 SUBDIR_EXCLUDE := \
  chrono \
  context \
  coroutine \
  date_time \
  graph \
  iostreams \
  log \
  math \
  regex \
  serialization \
  signals \
  thread
endif

SUBDIR_MAKEFILE_NAME := rdi-build.mk
include $(RDI_MAKEROOT)/subdir.mk
include $(RDI_MAKEROOT)/bottom.mk
