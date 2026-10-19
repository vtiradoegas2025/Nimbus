# Build profiles. CPU_ONLY=1 creates an isolated release artifact without the
# Vulkan implementation, shader build, or optional compression by default.
CPU_ONLY ?= 0
ALLOW_SERIAL ?= 0
REQUIRE_FULL_STACK ?= 0
LTO ?= 0
CHECKED ?= 0
PORTABLE ?= 0
# Distribution packagers set this separately from PORTABLE so a forgotten
# portability flag fails closed instead of encoding the build runner's ISA in
# an installer. Ordinary source builds leave both defaults alone and retain
# their host-native tuning.
DISTRIBUTION ?= 0
WARNINGS ?= 1
WERROR ?= 0
SANITIZE ?= none
COMPILER_CACHE ?= auto
UNAME_S := $(shell uname -s)
ifneq ($(filter MINGW% MSYS% CYGWIN%,$(UNAME_S)),)
  WINDOWS_HOST := 1
  EXEEXT := .exe
else
  WINDOWS_HOST := 0
  EXEEXT :=
endif
# Interface defaults and callback adapters intentionally leave parameters
# unused; keep the rest of -Wextra actionable without forcing dummy casts into
# public APIs.
WARNING_SUPPRESSIONS ?= -Wno-unused-parameter -Wno-missing-field-initializers \
	-Wno-overloaded-virtual

VALID_SANITIZERS := none address undefined address-undefined
ifeq ($(filter $(SANITIZE),$(VALID_SANITIZERS)),)
  $(error *** SANITIZE must be one of: $(VALID_SANITIZERS))
endif
ifneq ($(SANITIZE),none)
  override CHECKED := 1
endif

VALID_BINARY_SWITCHES := 0 1
ifeq ($(filter $(DISTRIBUTION),$(VALID_BINARY_SWITCHES)),)
  $(error *** DISTRIBUTION must be 0 or 1)
endif
ifeq ($(DISTRIBUTION),1)
  ifneq ($(PORTABLE),1)
    $(error *** DISTRIBUTION=1 requires PORTABLE=1; refusing a host-native release artifact)
  endif
endif

# Inherited Windows PATH entries can contain a compiler for another MSYS2 ABI.
ifeq ($(origin CXX),default)
  ifeq ($(WINDOWS_HOST):$(MSYSTEM),1:CLANGARM64)
    CXX := /clangarm64/bin/clang++
  else ifeq ($(WINDOWS_HOST):$(MSYSTEM),1:UCRT64)
    CXX := /ucrt64/bin/g++
  else ifneq ($(shell command -v g++ 2>/dev/null),)
    CXX := g++
  else ifneq ($(shell command -v clang++ 2>/dev/null),)
    CXX := clang++
  else
    CXX := c++
  endif
endif

# Sanitizer builds trade optimization for actionable diagnostics. OPTIMIZE is
# still overridable for toolchain experiments and release qualification.
ifeq ($(SANITIZE),none)
  OPTIMIZE ?= -O3
else
  OPTIMIZE ?= -O1
endif
CXXFLAGS := -std=c++17 $(OPTIMIZE) -I include -I src
ifeq ($(UNAME_S),Darwin)
  MACOSX_DEPLOYMENT_TARGET ?= 11.0
  export MACOSX_DEPLOYMENT_TARGET
  CXXFLAGS += -mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET)
  ifeq ($(DISTRIBUTION),1)
    CXXFLAGS += -Werror=unguarded-availability -Werror=unguarded-availability-new
  endif
endif
ifeq ($(WARNINGS),1)
  CXXFLAGS += -Wall -Wextra -Wpedantic $(WARNING_SUPPRESSIONS)
endif
ifeq ($(WERROR),1)
  CXXFLAGS += -Werror
endif
ifeq ($(PORTABLE),1)
  CXXFLAGS += -DNIMBUS_PORTABLE_BUILD
else
  CXXFLAGS += -march=native -mtune=native
endif
ifeq ($(DISTRIBUTION),1)
  ifneq ($(filter -march=% -mtune=%,$(CXXFLAGS)),)
    $(error *** distribution CXXFLAGS must not contain -march or -mtune host targeting)
  endif
  ifeq ($(findstring -DNIMBUS_PORTABLE_BUILD,$(CXXFLAGS)),)
    $(error *** distribution CXXFLAGS must retain -DNIMBUS_PORTABLE_BUILD evidence)
  endif
endif
ifeq ($(CPU_ONLY),1)
  CXXFLAGS += -DNIMBUS_CPU_ONLY
else
  CXXFLAGS += -I vulkan/include
endif
ifeq ($(LTO),1)
  CXXFLAGS += -flto
  LDFLAGS += -flto
endif

SANITIZER_FLAGS :=
ifeq ($(SANITIZE),address)
  SANITIZER_FLAGS := -fsanitize=address
else ifeq ($(SANITIZE),undefined)
  SANITIZER_FLAGS := -fsanitize=undefined
else ifeq ($(SANITIZE),address-undefined)
  SANITIZER_FLAGS := -fsanitize=address,undefined
endif
ifneq ($(SANITIZER_FLAGS),)
  SANITIZER_FLAGS += -fno-omit-frame-pointer -fno-sanitize-recover=all
  CXXFLAGS += $(SANITIZER_FLAGS)
  LDFLAGS += $(SANITIZER_FLAGS)
endif

# Prefer sccache when both wrappers are installed. COMPILER_CACHE=none disables
# wrapping; an explicit executable name/path is validated up front.
ifeq ($(COMPILER_CACHE),auto)
  CACHE_PROGRAM := $(firstword \
    $(shell command -v sccache 2>/dev/null) \
    $(shell command -v ccache 2>/dev/null))
else ifeq ($(COMPILER_CACHE),none)
  CACHE_PROGRAM :=
else
  CACHE_PROGRAM := $(shell command -v $(COMPILER_CACHE) 2>/dev/null)
  ifeq ($(CACHE_PROGRAM),)
    $(error *** COMPILER_CACHE=$(COMPILER_CACHE) was requested but not found)
  endif
endif
CXX_COMPILE = $(strip $(CACHE_PROGRAM) $(CXX))
# Physical aggregate constituents use this path for the public simulator. The
# aggregate overrides it with a receipt-bound exact snapshot.
PHYSICAL_QUALIFICATION_BINARY ?= bin/nimbus_sim
