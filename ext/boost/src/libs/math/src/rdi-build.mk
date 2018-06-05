include $(RDI_MAKEROOT)/top.mk
SUBDIR_MAKEFILE_NAME := rdi-build.mk
include $(RDI_MAKEROOT)/subdir.mk

include $(SSROOT)/Boost/rdi-boost.mk
MYINCLUDES := $(BOOST_INC_DIR)
MYCFLAGS := $(RDI_BOOST_CFLAGS)

# All the sources in tr1/
ALL_SRCS := $(patsubst %.cpp,%,$(patsubst $(SSCURDIR)/%,%,$(wildcard $(SSCURDIR)/tr1/*.cpp)))

# The subset of sources that make ip libboost_math_c99 (see ../build/JamFile.v2)
C99_SRCS := $(addprefix tr1/,\
acosh \
asinh \
atanh \
cbrt \
copysign \
erfc \
erf \
expm1 \
fmax \
fmin \
fpclassify \
hypot \
lgamma \
llround \
log1p \
lround \
nextafter \
nexttoward \
round \
tgamma \
trunc )

# The objs to exclude when building libboost_math_c99
LIBS_EXCLUDE_OBJS := $(filter-out $(C99_SRCS),$(ALL_SRCS))

LIBRARY_NAME := boost_math_c99
SHLIBRARIES :=
include $(RDI_MAKEROOT)/shlib.mk

include $(RDI_MAKEROOT)/bottom.mk

