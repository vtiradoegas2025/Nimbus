# Auto-detect compiler: prefer CXX from environment, then g++, then clang++
ifeq ($(origin CXX),default)
  ifneq ($(shell command -v g++ 2>/dev/null),)
    CXX := g++
  else ifneq ($(shell command -v clang++ 2>/dev/null),)
    CXX := clang++
  else
    CXX := c++
  endif
endif
CXXFLAGS := -std=c++17 -O3 -march=native -mtune=native -I include -I src -I vulkan/include
UNAME_S := $(shell uname -s)

# Vulkan headers: Homebrew (macOS) or system packages (Linux)
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
# Detect OpenMP support - handle both g++ and clang++
# On macOS, g++ is actually clang++, so check for libomp
# Try to find libomp via brew
LIBOMP_PATH := $(shell brew --prefix libomp 2>/dev/null)
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
  OPENMP_TEST := $(shell echo 'int main(){return 0;}' | $(CXX) -x c++ -fopenmp - -o /dev/null 2>&1)
  ifeq ($(OPENMP_TEST),)
    # -fopenmp works (GCC)
    OPENMP_FLAG := -fopenmp
    OPENMP_LIB :=
    CXXFLAGS += $(OPENMP_FLAG)
  else
    # -fopenmp doesn't work, OpenMP disabled
    OPENMP_FLAG :=
    OPENMP_LIB :=
    $(warning *** OpenMP not found -- simulation will run SINGLE-THREADED.)
    $(warning *** Install: macOS: brew install libomp | Ubuntu/Debian: sudo apt install libomp-dev | Fedora: sudo dnf install libomp-devel)
  endif
endif
OPENMP_LDLIBS := $(OPENMP_LIB)
# Optional GUI (SFML) support; default off
GUI ?= 0
# Optional: enable slice export from GUI with S key
EXPORT_NPY ?= 1
ifeq ($(EXPORT_NPY),1)
  CXXFLAGS += -DEXPORT_NPY
endif
# Optional: ZFP scientific compression
# Auto-detected from: brew, ~/.local, /usr/local, or ZFP_PREFIX env var.
# Set ZFP=0 to explicitly disable even when installed.
#
# Note: `brew --prefix zfp` returns a path (/opt/homebrew/opt/zfp) even when
# the package is NOT installed -- the directory simply does not exist on
# disk. We therefore validate every candidate prefix by checking that
# <prefix>/include/zfp.h actually exists before accepting it.
ifeq ($(origin ZFP_PREFIX), undefined)
  ZFP_BREW_PREFIX := $(shell brew --prefix zfp 2>/dev/null)
  ifneq ($(wildcard $(ZFP_BREW_PREFIX)/include/zfp.h),)
    ZFP_PREFIX := $(ZFP_BREW_PREFIX)
  else ifneq ($(wildcard $(HOME)/.local/include/zfp.h),)
    ZFP_PREFIX := $(HOME)/.local
  else ifneq ($(wildcard /usr/local/include/zfp.h),)
    ZFP_PREFIX := /usr/local
  endif
endif
ifneq ($(ZFP),0)
  ifneq ($(ZFP_PREFIX),)
    CXXFLAGS += -DHAVE_ZFP -I$(ZFP_PREFIX)/include
    ZFP_LDLIBS := -L$(ZFP_PREFIX)/lib -Wl,-rpath,$(ZFP_PREFIX)/lib -lzfp
  endif
endif

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
VALIDATION_SRCS := src/diagnostics/field_contract.cpp src/diagnostics/field_validation.cpp
SRCS := src/core/orchestration/dynamics/equations.cpp src/core/orchestration/dynamics/dynamics.cpp src/core/infra/field_sanitization.cpp src/core/infra/rayleigh_damping.cpp src/boundary_conditions/boundary_conditions_cartesian.cpp src/boundary_conditions/boundary_conditions_cylindrical.cpp src/boundary_conditions/boundary_conditions_cylindrical_cgrid.cpp src/boundary_conditions/factory.cpp src/core/orchestration/physics/diffusion_step.cpp src/core/orchestration/physics/microphysics_step.cpp src/core/orchestration/physics/radar_step.cpp src/core/infra/nested_grid.cpp src/core/orchestration/dynamics/initial_conditions_cartesian.cpp src/core/orchestration/dynamics/initial_conditions_cylindrical_cgrid.cpp src/core/runtime/tornado_sim.cpp src/core/runtime/headless_runtime.cpp src/core/runtime/runtime_config.cpp src/core/infra/coordinate_system.cpp src/compute/compute_backend.cpp src/compute/compute_kernel_template.cpp src/core/infra/hardware_info.cpp src/core/output/npy_writer.cpp src/core/output/output_config.cpp src/core/output/output_writer.cpp src/core/output/shm_writer.cpp src/core/orchestration/physics/radiation.cpp src/core/orchestration/physics/boundary_layer.cpp src/core/orchestration/physics/turbulence.cpp src/core/orchestration/dynamics/numerics.cpp src/core/infra/simd_utils.cpp \
         src/numerics/advection/advection.cpp \
         src/numerics/advection/advection_cartesian.cpp \
         src/core/orchestration/physics/radar.cpp \
         src/radar/base/radar_base.cpp \
         src/radar/factory.cpp \
         src/radar/schemes/reflectivity/reflectivity.cpp \
         src/radar/schemes/velocity/velocity.cpp \
         src/radar/schemes/zdr/zdr.cpp \
         src/soundings/soundings.cpp \
         src/soundings/factory.cpp \
         src/soundings/base/soundings_base.cpp \
         src/soundings/schemes/sharpy/sharpy_sounding.cpp \
         src/microphysics/base/thermodynamics.cpp \
         src/microphysics/factory.cpp \
         src/microphysics/schemes/kessler/kessler.cpp \
         src/microphysics/schemes/lin/lin.cpp \
         src/microphysics/schemes/thompson/thompson.cpp \
         src/microphysics/schemes/milbrandt/milbrandt.cpp \
         src/dynamics/factory.cpp \
         src/dynamics/schemes/cartesian/cartesian.cpp \
         src/dynamics/schemes/supercell/supercell.cpp \
         src/dynamics/schemes/tornado/tornado.cpp \
         src/dynamics/schemes/tornado/tornado_cgrid.cpp \
         src/radiation/base/radiative_transfer.cpp \
         src/radiation/factory.cpp \
         src/radiation/schemes/simple_grey/simple_grey.cpp \
         src/boundary_layer/base/surface_fluxes.cpp \
         src/boundary_layer/factory.cpp \
         src/boundary_layer/schemes/slab/slab.cpp \
         src/boundary_layer/schemes/ysu/ysu.cpp \
         src/boundary_layer/schemes/mynn/mynn.cpp \
         src/turbulence/base/eddy_viscosity.cpp \
         src/turbulence/factory.cpp \
         src/turbulence/schemes/smagorinsky/smagorinsky.cpp \
         src/turbulence/schemes/tke/tke.cpp \
         src/numerics/advection/factory.cpp \
         src/numerics/advection/schemes/tvd/tvd.cpp \
         src/numerics/advection/schemes/weno5/weno5.cpp \
         src/numerics/diffusion/factory.cpp \
         src/numerics/diffusion/schemes/explicit/explicit.cpp \
         src/numerics/diffusion/schemes/implicit/implicit.cpp \
         src/numerics/time_stepping/factory.cpp \
         src/numerics/time_stepping/schemes/rk3/rk3.cpp \
         src/numerics/time_stepping/schemes/rk4/rk4.cpp \
         src/numerics/time_stepping/schemes/split_explicit/split_explicit.cpp \
         src/chaos/chaos.cpp \
         src/chaos/base/random_generator.cpp \
         src/chaos/base/perturbation_field.cpp \
         src/chaos/base/correlation_filter.cpp \
         src/chaos/factory.cpp \
         src/chaos/schemes/none/none.cpp \
         src/chaos/schemes/initial_conditions/initial_conditions.cpp \
         src/chaos/schemes/boundary_layer/boundary_layer.cpp \
         src/chaos/schemes/full_stochastic/full_stochastic.cpp \
         src/core/orchestration/physics/terrain.cpp \
         src/terrain/base/topography.cpp \
         src/terrain/factory.cpp \
         src/terrain/schemes/bell/bell.cpp \
         src/terrain/schemes/schar/schar.cpp \
         src/terrain/schemes/none.cpp \
         vulkan/src/compute/compute_backend_vulkan.cpp \
         src/diagnostics/conservation_budget.cpp \
         $(VALIDATION_SRCS)
FIELD_VALIDATOR_SRCS := src/tools/field_validator.cpp $(VALIDATION_SRCS) vulkan/src/data/npy_reader.cpp
BACKEND_COMMON_SRCS := src/compute/compute_kernel_template.cpp src/compute/compute_backend.cpp src/core/infra/hardware_info.cpp src/core/infra/simd_utils.cpp vulkan/src/compute/compute_backend_vulkan.cpp

CPPFLAGS :=
LDLIBS := $(OPENMP_LDLIBS) $(ZFP_LDLIBS)
ifeq ($(UNAME_S),Linux)
  LDLIBS += -ldl
endif
ifeq ($(GUI),1)
  PKG_CONFIG := $(shell command -v pkg-config 2>/dev/null)
  ifeq ($(PKG_CONFIG),)
    SFML_PREFIX ?= $(shell brew --prefix sfml 2>/dev/null)
    SFML_CFLAGS := -I$(SFML_PREFIX)/include
    SFML_LIBS := -L$(SFML_PREFIX)/lib -lsfml-graphics -lsfml-window -lsfml-system
  else
    SFML_CFLAGS := $(shell pkg-config --cflags sfml-graphics)
    SFML_LIBS := $(shell pkg-config --libs sfml-graphics)
  endif
  CPPFLAGS += $(SFML_CFLAGS) -DENABLE_GUI=1
  LDLIBS += $(SFML_LIBS)
  SRCS += src/core/runtime/gui.cpp
endif

# ---------------------------------------------------------------------------
# Main binary (incremental object-file build)
# ---------------------------------------------------------------------------
BIN := bin/tornado_sim
FIELD_VALIDATOR := bin/field_validator
BUILDDIR := build

# Object files mirror source tree under build/
OBJS := $(patsubst %.cpp,$(BUILDDIR)/%.o,$(SRCS))
FIELD_VALIDATOR_OBJS := $(patsubst %.cpp,$(BUILDDIR)/%.o,$(FIELD_VALIDATOR_SRCS))

# Auto-dependency tracking (-MMD -MP generates .d files alongside .o files)
DEPFLAGS = -MMD -MP
DEPS := $(OBJS:.o=.d) $(FIELD_VALIDATOR_OBJS:.o=.d)

# Default target: build simulation + compile compute shaders if glslangValidator is available
.DEFAULT_GOAL := all
all: $(BIN) vulkan-compute-shaders
-include $(DEPS)

# Compile each .cpp to a .o (incremental, parallelizable with -j)
$(BUILDDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

# Link
$(BIN): $(OBJS) | bin
	$(CXX) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $(BIN)

$(FIELD_VALIDATOR): $(FIELD_VALIDATOR_OBJS) | bin
	$(CXX) $(FIELD_VALIDATOR_OBJS) $(LDFLAGS) $(LDLIBS) -o $(FIELD_VALIDATOR)

bin:
	mkdir -p bin

run: $(BIN)
	./$(BIN)

# ---------------------------------------------------------------------------
# Vulkan viewer and compute shaders
# ---------------------------------------------------------------------------
vulkan-compute-shaders:
	@GLSLANG=$$(command -v glslangValidator 2>/dev/null); \
	if [ -n "$$GLSLANG" ]; then \
		recompiled=0; \
		for comp in vulkan/shaders/compute/*.comp; do \
			[ -f "$$comp" ] || continue; \
			spv="$$comp.spv"; \
			if [ ! -f "$$spv" ] || [ "$$comp" -nt "$$spv" ]; then \
				$$GLSLANG -V -S comp -Os "$$comp" -o "$$spv" && recompiled=$$((recompiled+1)); \
			fi; \
		done; \
		if [ $$recompiled -gt 0 ]; then \
			echo "[SHADERS] Compiled $$recompiled compute shader(s)"; \
		fi; \
	fi

vulkan:
	$(MAKE) -C vulkan

run-vulkan: vulkan
	./bin/vulkan_viewer --dry-run

validate-fields: $(FIELD_VALIDATOR)
	./$(FIELD_VALIDATOR) --input data/exports --contract cm1 --mode report --json data/exports/validation_dataset_report.json

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
.PHONY: check-deps
check-deps:
	@echo "=== Nimbus dependency check ==="
	@printf "C++ compiler:   "; $(CXX) --version 2>&1 | head -1 || echo "NOT FOUND"
	@printf "GNU Make:       "; make --version 2>&1 | head -1
	@printf "OpenMP:         "; \
	  if echo 'int main(){return 0;}' | $(CXX) -x c++ -fopenmp - -o /dev/null 2>/dev/null; then \
	    echo "available"; \
	  elif [ -n "$$(brew --prefix libomp 2>/dev/null)" ]; then \
	    echo "available (Homebrew libomp)"; \
	  else \
	    echo "NOT FOUND -- simulation will run single-threaded"; \
	    echo "  Install: macOS: brew install libomp | Ubuntu: sudo apt install libomp-dev | Fedora: sudo dnf install libomp-devel"; \
	  fi
	@printf "Vulkan headers: "; \
	  if [ -f /usr/include/vulkan/vulkan.h ] || [ -f /usr/local/include/vulkan/vulkan.h ] || \
	     [ -n "$$(brew --prefix vulkan-headers 2>/dev/null)" ]; then \
	    echo "found"; \
	  else \
	    echo "NOT FOUND (optional -- needed for GPU compute and viewer)"; \
	    echo "  Install: macOS: brew install vulkan-headers vulkan-loader molten-vk | Ubuntu: sudo apt install libvulkan-dev"; \
	  fi
	@printf "glslangValidator: "; \
	  if command -v glslangValidator >/dev/null 2>&1; then \
	    glslangValidator --version 2>&1 | head -1; \
	  else \
	    echo "NOT FOUND (optional -- needed to compile GPU shaders)"; \
	    echo "  Install: macOS: brew install glslang | Ubuntu: sudo apt install glslang-tools"; \
	  fi
	@printf "GLFW:           "; \
	  if pkg-config --exists glfw3 2>/dev/null; then \
	    echo "found (pkg-config)"; \
	  elif [ -n "$$(brew --prefix glfw 2>/dev/null)" ]; then \
	    echo "found (Homebrew)"; \
	  else \
	    echo "NOT FOUND (optional -- needed for windowed viewer)"; \
	    echo "  Install: macOS: brew install glfw | Ubuntu: sudo apt install libglfw3-dev"; \
	  fi
	@echo "=== Done ==="

# ---------------------------------------------------------------------------
# Catch2 Test Suite
# ---------------------------------------------------------------------------
TEST_INFRA := tests/test_main.cpp tests/test_harness.cpp
TEST_CXXFLAGS := $(CXXFLAGS) -O0 -I tests

# Core tests
bin/test_core_field: $(TEST_INFRA) tests/core/test_field3d.cpp tests/core/test_field_pool.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_core_output: $(TEST_INFRA) tests/core/test_output_config.cpp src/core/output/output_config.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_core_hardware: $(TEST_INFRA) tests/core/test_hardware_info.cpp src/core/infra/hardware_info.cpp src/core/infra/simd_utils.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_core_npy: $(TEST_INFRA) tests/core/test_npy_writer.cpp src/core/output/npy_writer.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_core_output_writer: $(TEST_INFRA) tests/core/test_output_writer.cpp src/core/output/output_writer.cpp src/core/output/output_config.cpp src/core/output/npy_writer.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_core_zfp_roundtrip: $(TEST_INFRA) tests/core/test_zfp_roundtrip.cpp src/core/output/zfp_reader.cpp src/core/output/output_writer.cpp src/core/output/output_config.cpp src/core/output/npy_writer.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_core_zfp_benchmark: $(TEST_INFRA) tests/core/test_zfp_benchmark.cpp src/core/output/zfp_reader.cpp src/core/output/output_writer.cpp src/core/output/output_config.cpp src/core/output/npy_writer.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_core_shm: $(TEST_INFRA) tests/core/test_shm_transport.cpp src/core/output/shm_writer.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_core_coordinate_system: $(TEST_INFRA) tests/core/test_coordinate_system.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

# Diagnostics tests
bin/test_diagnostics_contract: $(TEST_INFRA) tests/diagnostics/test_field_contract.cpp src/diagnostics/field_contract.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_diagnostics_validation: $(TEST_INFRA) tests/diagnostics/test_field_validation.cpp src/diagnostics/field_validation.cpp src/diagnostics/field_contract.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_diagnostics_logging_ratchet: $(TEST_INFRA) tests/diagnostics/test_logging_ratchet.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

# Dynamics tests
bin/test_dynamics_cartesian: $(TEST_INFRA) tests/dynamics/test_cartesian_dynamics.cpp src/dynamics/schemes/cartesian/cartesian.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_dynamics_cartesian_bcs: $(TEST_INFRA) tests/dynamics/test_cartesian_boundary_conditions.cpp src/dynamics/schemes/cartesian/cartesian.cpp src/boundary_conditions/boundary_conditions_cartesian.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_dynamics_cylindrical_cgrid_bcs: $(TEST_INFRA) tests/dynamics/test_cylindrical_cgrid_boundary_conditions.cpp src/boundary_conditions/boundary_conditions_cylindrical_cgrid.cpp src/boundary_conditions/factory.cpp src/boundary_conditions/boundary_conditions_cylindrical.cpp src/boundary_conditions/boundary_conditions_cartesian.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_dynamics_cartesian_ic: $(TEST_INFRA) tests/dynamics/test_cartesian_initial_conditions.cpp src/core/orchestration/dynamics/initial_conditions_cartesian.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_dynamics_cylindrical_cgrid_ic: $(TEST_INFRA) tests/dynamics/test_cylindrical_cgrid_initial_conditions.cpp src/core/orchestration/dynamics/initial_conditions_cylindrical_cgrid.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_dynamics_tornado_cgrid: $(TEST_INFRA) tests/dynamics/test_tornado_cgrid_dynamics.cpp src/dynamics/schemes/tornado/tornado_cgrid.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

# Numerics tests
bin/test_numerics_advection: $(TEST_INFRA) tests/numerics/test_advection_tvd.cpp src/numerics/advection/factory.cpp src/numerics/advection/schemes/tvd/tvd.cpp src/numerics/advection/schemes/weno5/weno5.cpp src/numerics/advection/advection.cpp src/numerics/advection/advection_cartesian.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_numerics_tvd_monotonicity: $(TEST_INFRA) tests/numerics/test_tvd_monotonicity.cpp src/numerics/advection/factory.cpp src/numerics/advection/schemes/tvd/tvd.cpp src/numerics/advection/schemes/weno5/weno5.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_numerics_advection_cartesian: $(TEST_INFRA) tests/numerics/test_advection_cartesian.cpp src/numerics/advection/factory.cpp src/numerics/advection/schemes/tvd/tvd.cpp src/numerics/advection/schemes/weno5/weno5.cpp src/numerics/advection/advection.cpp src/numerics/advection/advection_cartesian.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_numerics_diffusion: $(TEST_INFRA) tests/numerics/test_diffusion.cpp src/numerics/diffusion/factory.cpp src/numerics/diffusion/schemes/explicit/explicit.cpp src/numerics/diffusion/schemes/implicit/implicit.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_numerics_timestepping: $(TEST_INFRA) tests/numerics/test_time_stepping.cpp src/numerics/time_stepping/factory.cpp src/numerics/time_stepping/schemes/rk3/rk3.cpp src/numerics/time_stepping/schemes/rk4/rk4.cpp src/numerics/time_stepping/schemes/split_explicit/split_explicit.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_numerics_staggered_derivatives: $(TEST_INFRA) tests/numerics/test_staggered_derivatives.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

# Physics tests
bin/test_physics_microphysics: $(TEST_INFRA) tests/physics/test_microphysics.cpp src/microphysics/factory.cpp src/microphysics/schemes/kessler/kessler.cpp src/microphysics/schemes/lin/lin.cpp src/microphysics/schemes/thompson/thompson.cpp src/microphysics/schemes/milbrandt/milbrandt.cpp src/microphysics/base/thermodynamics.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_physics_radiation: $(TEST_INFRA) tests/physics/test_radiation.cpp src/radiation/factory.cpp src/radiation/base/radiative_transfer.cpp src/radiation/schemes/simple_grey/simple_grey.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_physics_terrain: $(TEST_INFRA) tests/physics/test_terrain.cpp src/terrain/base/topography.cpp src/terrain/factory.cpp src/terrain/schemes/bell/bell.cpp src/terrain/schemes/schar/schar.cpp src/terrain/schemes/none.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

# Data tests
bin/test_data_soundings: $(TEST_INFRA) tests/data/test_soundings.cpp src/soundings/factory.cpp src/soundings/base/soundings_base.cpp src/soundings/schemes/sharpy/sharpy_sounding.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

# Vulkan/compute tests
bin/test_vulkan_backend: $(TEST_INFRA) tests/vulkan/test_compute_backend.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_vulkan_gpu_parity: $(TEST_INFRA) tests/vulkan/test_gpu_parity.cpp $(BACKEND_COMMON_SRCS) src/microphysics/factory.cpp src/microphysics/schemes/kessler/kessler.cpp src/microphysics/schemes/lin/lin.cpp src/microphysics/schemes/thompson/thompson.cpp src/microphysics/schemes/milbrandt/milbrandt.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

# Integration tests
bin/test_integration: $(TEST_INFRA) tests/integration/test_config_presets.cpp tests/integration/test_performance.cpp src/core/output/output_config.cpp src/core/output/npy_writer.cpp src/core/infra/hardware_info.cpp src/core/infra/simd_utils.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

bin/test_shm_e2e: $(TEST_INFRA) tests/integration/test_shm_e2e.cpp src/core/output/shm_writer.cpp vulkan/src/data/shm_dataset.cpp | bin
	$(CXX) $(TEST_CXXFLAGS) $(CPPFLAGS) $^ $(LDLIBS) -o $@

CATCH2_BINS := bin/test_core_field bin/test_core_output bin/test_core_hardware bin/test_core_npy \
               bin/test_core_output_writer bin/test_core_shm bin/test_core_coordinate_system \
               bin/test_diagnostics_contract bin/test_diagnostics_validation \
               bin/test_dynamics_cartesian bin/test_dynamics_cartesian_bcs bin/test_dynamics_cartesian_ic \
               bin/test_dynamics_cylindrical_cgrid_bcs bin/test_dynamics_cylindrical_cgrid_ic bin/test_dynamics_tornado_cgrid \
               bin/test_numerics_advection bin/test_numerics_advection_cartesian bin/test_numerics_diffusion bin/test_numerics_timestepping bin/test_numerics_staggered_derivatives \
               bin/test_physics_microphysics bin/test_physics_radiation bin/test_physics_terrain \
               bin/test_data_soundings bin/test_vulkan_backend bin/test_vulkan_gpu_parity \
               bin/test_integration bin/test_shm_e2e

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------
.PHONY: all run clean clean-vulkan vulkan vulkan-compute-shaders run-vulkan validate-fields \
        test test-all test-core test-diagnostics test-dynamics test-numerics test-physics test-data \
        test-vulkan test-integration test-shm-e2e smoke-test smoke-test-e2e benchmark-point2

ifeq ($(ZFP),1)
  CORE_ZFP_TESTS := bin/test_core_zfp_roundtrip bin/test_core_zfp_benchmark
  CORE_ZFP_RUN := && ./bin/test_core_zfp_roundtrip && ./bin/test_core_zfp_benchmark
else
  CORE_ZFP_TESTS :=
  CORE_ZFP_RUN :=
endif

test-core: bin/test_core_field bin/test_core_output bin/test_core_hardware bin/test_core_npy bin/test_core_output_writer bin/test_core_shm bin/test_core_coordinate_system $(CORE_ZFP_TESTS)
	./bin/test_core_field && ./bin/test_core_output && ./bin/test_core_hardware && ./bin/test_core_npy && ./bin/test_core_output_writer && ./bin/test_core_shm && ./bin/test_core_coordinate_system $(CORE_ZFP_RUN)

test-diagnostics: bin/test_diagnostics_contract bin/test_diagnostics_validation bin/test_diagnostics_logging_ratchet
	./bin/test_diagnostics_contract && ./bin/test_diagnostics_validation && ./bin/test_diagnostics_logging_ratchet

test-dynamics: bin/test_dynamics_cartesian bin/test_dynamics_cartesian_bcs bin/test_dynamics_cartesian_ic bin/test_dynamics_cylindrical_cgrid_bcs bin/test_dynamics_cylindrical_cgrid_ic bin/test_dynamics_tornado_cgrid
	./bin/test_dynamics_cartesian && ./bin/test_dynamics_cartesian_bcs && ./bin/test_dynamics_cartesian_ic && ./bin/test_dynamics_cylindrical_cgrid_bcs && ./bin/test_dynamics_cylindrical_cgrid_ic && ./bin/test_dynamics_tornado_cgrid

test-numerics: bin/test_numerics_advection bin/test_numerics_tvd_monotonicity bin/test_numerics_advection_cartesian bin/test_numerics_diffusion bin/test_numerics_timestepping bin/test_numerics_staggered_derivatives
	./bin/test_numerics_advection && ./bin/test_numerics_tvd_monotonicity && ./bin/test_numerics_advection_cartesian && ./bin/test_numerics_diffusion && ./bin/test_numerics_timestepping && ./bin/test_numerics_staggered_derivatives

test-physics: bin/test_physics_microphysics bin/test_physics_radiation bin/test_physics_terrain
	./bin/test_physics_microphysics && ./bin/test_physics_radiation && ./bin/test_physics_terrain

test-data: bin/test_data_soundings
	./bin/test_data_soundings

test-vulkan: bin/test_vulkan_backend bin/test_vulkan_gpu_parity
	./bin/test_vulkan_backend && ./bin/test_vulkan_gpu_parity

test-integration: bin/test_integration
	./bin/test_integration

test-shm-e2e: bin/test_shm_e2e
	./bin/test_shm_e2e

test: test-core test-diagnostics test-dynamics test-numerics test-physics test-data test-vulkan test-integration test-shm-e2e
	@echo "=== All tests passed ==="

smoke-test: $(BIN)
	./$(BIN) --headless --config=configs/student/student.yaml --duration=5 2>/dev/null && echo "[smoke-test] PASS"

smoke-test-e2e: $(BIN) $(VULKAN_BIN)
	bash ./tools/smoke_test_e2e.sh

test-all: test smoke-test smoke-test-e2e

benchmark-point2: $(BIN)
	bash ./tools/benchmark_point2_first_kernel_offload.sh

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean-vulkan:
	$(MAKE) -C vulkan clean

clean:
	rm -rf $(BUILDDIR)
	rm -f $(BIN) $(FIELD_VALIDATOR) $(CATCH2_BINS)
	@$(MAKE) -C vulkan clean >/dev/null 2>&1 || true
