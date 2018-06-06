include $(RDI_MAKEROOT)/top.mk

include $(SSROOT)/Boost/rdi-boost.mk
MYINCLUDES := $(BOOST_INC_DIR)

# NOTE, DATE_TIME_INLINE does not have to be defined.
CPP_SUFFIX := .cpp
include $(RDI_MAKEROOT)/objs.mk

LIBRARY_NAME := boost_date_time
LIBRARIES :=
include $(RDI_MAKEROOT)/shlib.mk

include $(RDI_MAKEROOT)/bottom.mk
