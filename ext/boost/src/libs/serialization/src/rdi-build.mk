include $(RDI_MAKEROOT)/top.mk

# SOURCES listed in  ../build/Jamfile.v2)
BS_SOURCES := $(addsuffix .cpp,\
basic_archive \
basic_iarchive \
basic_iserializer \
basic_oarchive \
basic_oserializer \
basic_pointer_iserializer \
basic_pointer_oserializer \
basic_serializer_map \
basic_text_iprimitive \
basic_text_oprimitive \
basic_xml_archive \
binary_iarchive \
binary_oarchive \
extended_type_info \
extended_type_info_typeid \
extended_type_info_no_rtti \
polymorphic_iarchive \
polymorphic_oarchive \
stl_port \
text_iarchive \
text_oarchive \
void_cast \
archive_exception \
xml_grammar \
xml_iarchive \
xml_oarchive \
xml_archive_exception \
)

# WSOURCES listed in  ../build/Jamfile.v2)
BS_WSOURCES := $(addsuffix .cpp,\
basic_text_wiprimitive \
basic_text_woprimitive \
text_wiarchive \
text_woarchive \
utf8_codecvt_facet \
xml_wgrammar \
xml_wiarchive \
xml_woarchive \
codecvt_null \
)

# We build only the serialiazation library, not the wserialization
#BS_ALL_SOURCES := $(BS_SOURCES) $(BS_WSOURCES)
BS_ALL_SOURCES := $(BS_SOURCES)

boost-source := yes
include $(SSROOT)/Boost/rdi-boost.mk
MYINCLUDES := $(BOOST_INC_DIR)
MYCFLAGS := $(RDI_BOOST_CFLAGS)

include $(RDI_MAKEROOT)/platform.mk
ifeq ($(RDI_OS),windows)
 MYCFLAGS += -DBOOST_SERIALIZATION_DYN_LINK
endif
ifeq ($(RDI_OS),gnu/linux)
endif

OBJS_EXCLUDE_SRCS := $(filter-out $(BS_ALL_SOURCES),$(notdir $(wildcard $(SSCURDIR)/*.cpp)))
CPP_SUFFIX := .cpp
include $(RDI_MAKEROOT)/objs.mk

LIBRARY_NAME := boost_serialization
LIBRARIES :=
include $(RDI_MAKEROOT)/libs.mk

include $(RDI_MAKEROOT)/bottom.mk

