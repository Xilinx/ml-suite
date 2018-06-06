################################################################
# Makefile for releasing boost runtime and link libraries
################################################################
include $(RDI_MAKEROOT)/top.mk
include $(RDI_MAKEROOT)/platform.mk

BOOST_COMPILER_VERSION := $(RDI_COMPILER_VERSION)

ifeq ($(boost-debug),yes)
 BOOST_LIB_DIR := $(SSCURDIR)/$(BOOST_COMPILER_VERSION)/$(RDI_PLATFORM).g
else
 BOOST_LIB_DIR := $(SSCURDIR)/$(BOOST_COMPILER_VERSION)/$(RDI_PLATFORM).o
endif

ifeq ($(RDI_OS),windows)
 BOOST_LINK_LIBS := $(wildcard $(BOOST_LIB_DIR)/*.lib)
 BOOST_RUNTIME_LIBS := $(wildcard $(BOOST_LIB_DIR)/*.dll)
endif

ifeq ($(RDI_OS),gnu/linux)
 BOOST_STATIC_LIBS := $(wildcard $(BOOST_LIB_DIR)/*.a)
 BOOST_LINK_LIBS := $(wildcard $(BOOST_LIB_DIR)/*.so)
 BOOST_RUNTIME_LIBS := $(BOOST_LINK_LIBS)
 BOOST_DEPS := $(wildcard $(BOOST_LIB_DIR)/*.dep)
endif

LINK_TARGET_DIR := $(RDI_LINK_DIR)/$(SSNAME)
LINK_TARGETS := $(addprefix $(LINK_TARGET_DIR)/,$(notdir $(BOOST_LINK_LIBS)) $(notdir $(BOOST_STATIC_LIBS)))
LINK_SOURCES := $(BOOST_LINK_LIBS)

DEP_TARGET_FILES := $(addprefix $(LINK_TARGET_DIR)/,$(notdir $(BOOST_DEPS)))
DEP_SOURCE_FILES := $(BOOST_DEPS)

libs: $(LINK_TARGETS) 
dirs: $(LINK_TARGET_DIR)/.rdi
depend: $(DEP_TARGET_FILES)

# Release dependecy files
$(DEP_TARGET_FILES): $(LINK_TARGET_DIR)/.rdi
$(DEP_TARGET_FILES): $(LINK_TARGET_DIR)/% : $(BOOST_LIB_DIR)/%
	$(CP) $< $@

# Release link libraries
$(LINK_TARGETS): $(LINK_TARGET_DIR)/.rdi
$(LINK_TARGETS): $(LINK_TARGET_DIR)/% : $(BOOST_LIB_DIR)/%
	$(CP) $< $@

# Release the runtime libraries
RELEASE_LIBS_EXACT := $(BOOST_RUNTIME_LIBS)
RELEASE_TYPE := customer
include $(RDI_MAKEROOT)/release.mk


include $(RDI_MAKEROOT)/bottom.mk
