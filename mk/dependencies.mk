# Native Metal is a production backend only on macOS. Keep the opt-in define
# on production objects so monolithic portable/CPU tests continue to link the
# fail-closed C++ factory stub without Apple framework dependencies.
ifeq ($(UNAME_S),Darwin)
  ifneq ($(CPU_ONLY),1)
    CXXFLAGS += -I metal/include
    METAL_MAIN_FLAGS := -DNIMBUS_ENABLE_METAL
    METAL_CXX := xcrun clang++
    METAL_LDLIBS := -framework Foundation -framework Metal
  endif
endif

# Vulkan headers: Homebrew (macOS) or system packages (Linux)
ifneq ($(CPU_ONLY),1)
BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
ifneq ($(BREW_PREFIX),)
  VULKAN_HEADERS_PREFIX := $(shell brew --prefix vulkan-headers 2>/dev/null)
  ifneq ($(VULKAN_HEADERS_PREFIX),)
    CXXFLAGS += -I$(VULKAN_HEADERS_PREFIX)/include
  endif
else ifeq ($(UNAME_S),Linux)
  # Linux: check VULKAN_SDK env var (LunarG SDK), then standard system paths
  ifneq ($(VULKAN_SDK),)
    CXXFLAGS += -I$(VULKAN_SDK)/include
    LDFLAGS += -L$(VULKAN_SDK)/lib
  else ifneq ($(wildcard /usr/include/vulkan/vulkan.h),)
    # System headers already on default include path
  else ifneq ($(wildcard /usr/local/include/vulkan/vulkan.h),)
    CXXFLAGS += -I/usr/local/include
  endif
endif
endif
# Detect OpenMP support - handle both g++ and clang++
# On macOS, g++ is actually clang++, so check for libomp
# Try to find libomp via brew
LIBOMP_PATH ?= $(shell brew --prefix libomp 2>/dev/null)
ifeq ($(LIBOMP_PATH),)
  # Try alternative path
  LIBOMP_PATH := $(shell test -d /opt/homebrew/opt/libomp && echo /opt/homebrew/opt/libomp || echo "")
endif
ifneq ($(LIBOMP_PATH),)
  # libomp found - use it for clang
  # Check if include directory exists, if not try to find omp.h elsewhere
  ifeq ($(shell test -d $(LIBOMP_PATH)/include && echo "yes"),yes)
    LIBOMP_INCLUDE := -I$(LIBOMP_PATH)/include
  else
    # Try to find omp.h in common locations
    OMP_H_PATH := $(shell find $(LIBOMP_PATH) -name "omp.h" 2>/dev/null | head -1)
    ifneq ($(OMP_H_PATH),)
      LIBOMP_INCLUDE := -I$(shell dirname $(OMP_H_PATH))
    else
      # No include found, disable OpenMP
      LIBOMP_PATH :=
    endif
  endif
  ifneq ($(LIBOMP_PATH),)
    OPENMP_FLAG := -Xpreprocessor -fopenmp
    OPENMP_LIB := -L$(LIBOMP_PATH)/lib -lomp
    CXXFLAGS += $(OPENMP_FLAG) $(LIBOMP_INCLUDE)
  endif
endif
# If OpenMP not found above, try standard -fopenmp (works for GCC)
# If that also fails, OpenMP will be disabled (pragmas will be ignored)
ifeq ($(LIBOMP_PATH),)
  # Test if -fopenmp works (GCC) or fails (clang without libomp)
  OPENMP_TEST := $(shell bash scripts/dependencies/probe_openmp.sh $(CXX) && echo available)
  ifeq ($(OPENMP_TEST),available)
    # -fopenmp works (GCC)
    OPENMP_FLAG := -fopenmp
    OPENMP_LIB := -fopenmp
    CXXFLAGS += $(OPENMP_FLAG)
  else
    # -fopenmp doesn't work, OpenMP disabled
    OPENMP_FLAG :=
    OPENMP_LIB :=
    ifeq ($(CPU_ONLY),1)
      ifneq ($(ALLOW_SERIAL),1)
        $(error *** CPU_ONLY release requires OpenMP. Install OpenMP or set ALLOW_SERIAL=1 explicitly.)
      endif
    endif
    $(warning *** OpenMP not found -- simulation will run SINGLE-THREADED.)
    $(warning *** Install: macOS: brew install libomp | Ubuntu/Debian: sudo apt install libomp-dev | Fedora: sudo dnf install libomp-devel)
  endif
endif
OPENMP_LDLIBS := $(OPENMP_LIB)
# Optional: enable periodic NPY slice export during simulation
EXPORT_NPY ?= 1
ifeq ($(EXPORT_NPY),1)
  CXXFLAGS += -DEXPORT_NPY
endif
# Optional: ZFP scientific compression
# Auto-detected from: brew, ~/.local, /usr/local, /usr, or ZFP_PREFIX env var.
# Set ZFP=0 to explicitly disable even when installed.
#
# Note: `brew --prefix zfp` returns a path (/opt/homebrew/opt/zfp) even when
# the package is NOT installed -- the directory simply does not exist on
# disk. We therefore validate every candidate prefix by checking that
# <prefix>/include/zfp.h actually exists before accepting it.
ifeq ($(CPU_ONLY),1)
  ifeq ($(origin ZFP),undefined)
    ZFP := 0
  endif
endif
ifeq ($(origin ZFP_PREFIX), undefined)
  ZFP_BREW_PREFIX := $(shell brew --prefix zfp 2>/dev/null)
  ifneq ($(wildcard $(ZFP_BREW_PREFIX)/include/zfp.h),)
    ZFP_PREFIX := $(ZFP_BREW_PREFIX)
  else ifneq ($(wildcard $(HOME)/.local/include/zfp.h),)
    ZFP_PREFIX := $(HOME)/.local
  else ifneq ($(wildcard /usr/local/include/zfp.h),)
    ZFP_PREFIX := /usr/local
  else ifneq ($(wildcard /usr/include/zfp.h),)
    ZFP_PREFIX := /usr
  else ifeq ($(WINDOWS_HOST),1)
    ifneq ($(wildcard /ucrt64/include/zfp.h),)
      ZFP_PREFIX := /ucrt64
    else ifneq ($(wildcard /clangarm64/include/zfp.h),)
      ZFP_PREFIX := /clangarm64
    endif
  endif
endif
ifneq ($(ZFP),0)
  ifneq ($(ZFP_PREFIX),)
    # The Windows setup may live below a user-selected checkout path containing
    # spaces. Keep the externally supplied prefix one shell argument all the
    # way through compile and link recipes.
    CXXFLAGS += -DHAVE_ZFP -I"$(ZFP_PREFIX)/include"
    ifeq ($(WINDOWS_HOST),1)
      ZFP_LDLIBS := -L"$(ZFP_PREFIX)/lib" -lzfp
    else
      ZFP_LDLIBS := -L"$(ZFP_PREFIX)/lib" -Wl,-rpath,"$(ZFP_PREFIX)/lib" -lzfp
    endif
  endif
endif

# Optional: self-describing netCDF4 history output.  Detection is compile-time
# so output capability reports the truth for the exact Nimbus binary.  Set
# NETCDF=0 to produce a binary that explicitly rejects output.format=netcdf.
NETCDF ?= 1
ifneq ($(NETCDF),0)
  NETCDF_CONFIG ?= $(shell command -v nc-config 2>/dev/null)
  ifneq ($(NETCDF_CONFIG),)
    NETCDF_CFLAGS := $(shell $(NETCDF_CONFIG) --cflags 2>/dev/null)
    NETCDF_LDLIBS := $(shell $(NETCDF_CONFIG) --libs 2>/dev/null)
    CXXFLAGS += -DHAVE_NETCDF $(NETCDF_CFLAGS)
  endif
endif

CPPFLAGS :=
LDLIBS := $(OPENMP_LDLIBS) $(ZFP_LDLIBS) $(NETCDF_LDLIBS)
ifeq ($(UNAME_S),Linux)
  LDLIBS += -ldl
endif
ifeq ($(WINDOWS_HOST),1)
  LDLIBS += -ladvapi32 -lpsapi
endif

# Setup and release packaging promise one capability contract on every desktop
# OS.  Make's optional dependency discovery remains useful for focused
# developer builds, but the product route must fail closed instead of quietly
# producing a serial/no-codec binary that happens to have the same filename.
ifeq ($(REQUIRE_FULL_STACK),1)
  ifeq ($(strip $(OPENMP_FLAG)),)
    $(error *** full Nimbus product requires OpenMP)
  endif
  ifeq ($(strip $(ZFP_LDLIBS)),)
    $(error *** full Nimbus product requires ZFP; install it or set ZFP_PREFIX)
  endif
  ifeq ($(strip $(NETCDF_LDLIBS)),)
    $(error *** full Nimbus product requires netCDF and nc-config)
  endif
  ifeq ($(CPU_ONLY),1)
    $(error *** full Nimbus product cannot be built with CPU_ONLY=1)
  endif
endif
