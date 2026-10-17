# Native-Metal qualification target. Production runtime integration remains
# platform-gated in the repository root build.

UNAME_S := $(shell uname -s)
ifeq ($(origin CXX),default)
CXX := xcrun clang++
endif
ROOT := $(abspath ..)
BIN := $(ROOT)/bin/test_metal_compute_backend
METAL_QUALIFICATION_DIR ?= $(ROOT)/reports/qualification/metal
METAL_QUALIFICATION_REPORT := $(METAL_QUALIFICATION_DIR)/bootstrap-current.json
METAL_QUALIFICATION_LOG := $(METAL_QUALIFICATION_DIR)/bootstrap-current.log

.PHONY: all test runtime-test domain-recycling-test moving-nest-test moving-nest-admission-test live-recycling-test qualify clean

ifeq ($(UNAME_S),Darwin)
METAL_TEST_SOURCES := \
	$(ROOT)/tests/test_main.cpp \
	$(ROOT)/tests/metal/test_metal_compute_backend.cpp \
	$(ROOT)/metal/src/compute/metal_compute_backend.mm \
	$(ROOT)/src/compute/backend_runtime.cpp \
	$(ROOT)/src/compute/resident_field_store.cpp \
	$(ROOT)/src/compute/kernel_manifest.cpp \
	$(ROOT)/src/chaos/base/earth_correlated_noise.cpp \
	$(ROOT)/src/core/infra/domain_motion.cpp \
	$(ROOT)/src/core/infra/domain_recycling.cpp \
	$(ROOT)/src/terrain/base/conservative_transport.cpp \
	$(ROOT)/src/terrain/base/topography.cpp \
	$(ROOT)/src/terrain/base/native_mapped_transaction.cpp \
	$(ROOT)/src/core/runtime/simulation_state.cpp \
	$(ROOT)/src/core/orchestration/dynamics/eos_apply.cpp \
	$(ROOT)/src/diagnostics/budgets/energy_budget.cpp \
	$(ROOT)/src/dynamics/cartesian_cgrid_momentum_flux.cpp \
	$(ROOT)/src/dynamics/device_slow_energy_attribution.cpp \
	$(ROOT)/src/dynamics/native_cgrid_boundary.cpp \
	$(ROOT)/src/numerics/advection/advection_cartesian.cpp \
	$(ROOT)/src/numerics/diffusion/schemes/explicit/explicit.cpp \
	$(ROOT)/src/microphysics/schemes/lin/lin.cpp \
	$(ROOT)/src/turbulence/base/eddy_viscosity.cpp \
	$(ROOT)/src/turbulence/base/collocated_eddy_viscosity.cpp \
	$(ROOT)/src/boundary_layer/base/surface_fluxes.cpp \
	$(ROOT)/src/boundary_layer/schemes/mynn/mynn_stability.cpp \
	$(ROOT)/src/boundary_layer/schemes/mynn/mynn.cpp \
	$(ROOT)/src/boundary_layer/schemes/mynn/mynn_batch.cpp \
	$(ROOT)/src/radiation/schemes/rrtmg/rrtmg_longwave_transfer.cpp \
	$(ROOT)/src/radiation/schemes/rrtmg/rrtmg_shortwave_transfer.cpp
METAL_TEST_HEADERS := $(shell find $(ROOT)/metal/include -type f -name '*.hpp') \
	$(shell find $(ROOT)/metal/src/compute/backend -type f -name '*.inc') \
	$(shell find $(ROOT)/metal/src/compute/internal -type f -name '*.hpp') \
	$(shell find $(ROOT)/include/compute -type f -name '*.hpp') \
	$(ROOT)/src/compute/compute_backend_factory.hpp \
	$(ROOT)/tests/compute/support/agrid_complete_native_fixture.hpp \
	$(ROOT)/tests/compute/support/sgs_scalar_boundary.hpp \
	$(ROOT)/tests/compute/support/acoustic_unrepaired.hpp \
	$(ROOT)/include/dynamics/device_acoustic_energy_attribution.hpp \
	$(ROOT)/tests/catch2/catch.hpp

all: $(BIN)

$(BIN): $(ROOT)/vulkan/shaders/compute/acoustic/pressure_eos.glsl $(ROOT)/vulkan/shaders/compute/acoustic/momentum_precision.glsl $(ROOT)/scripts/dev/shaders/generate_acoustic_eos_msl.py

$(BIN): $(ROOT)/vulkan/shaders/compute/chaos/earth_control_evolution.comp $(ROOT)/scripts/dev/shaders/generate_earth_control_msl.py

$(BIN): $(ROOT)/vulkan/shaders/compute/dynamics/conservative_reduction.comp $(ROOT)/scripts/dev/shaders/generate_conservative_reduction_msl.py $(ROOT)/vulkan/shaders/compute/dynamics/conservative_momentum.comp $(ROOT)/scripts/dev/shaders/generate_conservative_momentum_msl.py $(ROOT)/vulkan/shaders/compute/microphysics/lin_pointwise.comp $(ROOT)/vulkan/shaders/compute/microphysics/lin_sedimentation.comp $(ROOT)/scripts/dev/shaders/generate_lin_msl.py $(ROOT)/vulkan/shaders/compute/microphysics/kessler_sedimentation.comp $(ROOT)/scripts/dev/shaders/generate_kessler_sedimentation_msl.py $(ROOT)/vulkan/shaders/compute/dynamics/acoustic_bc_face_cartesian_cgrid.comp $(ROOT)/vulkan/shaders/compute/dynamics/boundary_float32.glsl $(ROOT)/scripts/dev/shaders/generate_acoustic_bc_face_cartesian_cgrid_msl.py $(ROOT)/vulkan/shaders/compute/advection/advect_conservative_scalar_flux_cartesian_cgrid.comp $(ROOT)/scripts/dev/shaders/generate_advect_conservative_scalar_flux_cartesian_cgrid_msl.py $(ROOT)/vulkan/shaders/compute/terrain/held_sources.comp $(ROOT)/scripts/dev/shaders/generate_held_sources_msl.py $(ROOT)/vulkan/shaders/compute/boundary_layer/slab_column.comp $(ROOT)/scripts/dev/shaders/generate_slab_column_msl.py $(ROOT)/vulkan/shaders/compute/radiation/simple_grey.comp $(ROOT)/scripts/dev/shaders/generate_simple_grey_msl.py $(ROOT)/vulkan/shaders/compute/terrain/column_remap.comp $(ROOT)/scripts/dev/shaders/generate_column_remap_msl.py $(ROOT)/vulkan/shaders/compute/terrain/mapped_boundary.comp $(ROOT)/scripts/dev/shaders/generate_mapped_boundary_msl.py $(ROOT)/vulkan/shaders/compute/terrain/reference_atmosphere.comp $(ROOT)/scripts/dev/shaders/generate_reference_atmosphere_msl.py $(ROOT)/vulkan/shaders/compute/terrain/moving_energy.comp $(ROOT)/scripts/dev/shaders/generate_moving_energy_msl.py $(ROOT)/vulkan/shaders/compute/terrain/moving_momentum.comp $(ROOT)/vulkan/shaders/compute/terrain/compensated_storage.glsl $(ROOT)/scripts/dev/shaders/generate_moving_momentum_msl.py $(ROOT)/vulkan/shaders/compute/terrain/earth_geometry.comp $(ROOT)/scripts/dev/shaders/generate_earth_geometry_msl.py $(ROOT)/vulkan/shaders/compute/terrain/conservative_remap.comp $(ROOT)/scripts/dev/shaders/generate_remap_msl.py $(ROOT)/vulkan/shaders/compute/turbulence/collocated_sgs.comp $(ROOT)/vulkan/shaders/compute/time_stepping/rk_tendency.comp $(ROOT)/vulkan/shaders/compute/time_stepping/rk_combination.comp $(ROOT)/scripts/dev/shaders/generate_rk_msl.py $(ROOT)/vulkan/shaders/compute/microphysics/milbrandt_sedimentation.comp $(ROOT)/vulkan/shaders/compute/microphysics/milbrandt_cold.comp $(ROOT)/vulkan/shaders/compute/microphysics/compensated_fma.glsl $(ROOT)/vulkan/shaders/compute/microphysics/milbrandt_melting.comp $(ROOT)/vulkan/shaders/compute/microphysics/milbrandt_cloud_adjustment.comp $(ROOT)/vulkan/shaders/compute/microphysics/milbrandt_thermodynamics.glsl $(ROOT)/vulkan/shaders/compute/microphysics/milbrandt_warm_rain.comp $(ROOT)/vulkan/shaders/compute/microphysics/thompson_compensated.glsl $(ROOT)/scripts/dev/shaders/generate_milbrandt_msl.py $(ROOT)/vulkan/shaders/compute/diffusion/implicit_diffusion_stencil.comp $(ROOT)/vulkan/shaders/compute/diffusion/implicit_diffusion_vector.comp $(ROOT)/scripts/dev/shaders/generate_implicit_msl.py $(METAL_TEST_SOURCES) $(METAL_TEST_HEADERS) $(ROOT)/vulkan/shaders/compute/turbulence/tke_closure.comp $(ROOT)/vulkan/shaders/compute/turbulence/tke_tendency.comp $(ROOT)/scripts/dev/shaders/generate_tke_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_earth_control_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_acoustic_eos_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_implicit_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_tke_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_mapped_boundary_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_slab_column_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_simple_grey_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_lin_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_kessler_sedimentation_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_acoustic_bc_face_cartesian_cgrid_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_advect_conservative_scalar_flux_cartesian_cgrid_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_held_sources_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_column_remap_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_reference_atmosphere_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_moving_energy_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_conservative_reduction_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_conservative_momentum_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_moving_momentum_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_earth_geometry_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_remap_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_rk_msl.py
	python3 $(ROOT)/scripts/dev/shaders/generate_milbrandt_msl.py
	$(CXX) -std=c++17 -O0 -g -fobjc-arc -Wall -Wextra -Wpedantic \
		-Wno-unused-parameter -Wno-missing-field-initializers \
		-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
		-I$(ROOT)/include -I$(ROOT)/src -I$(ROOT)/metal/include -I$(ROOT)/tests \
		$(METAL_TEST_SOURCES) -L/opt/homebrew/opt/libomp/lib -lomp \
		-framework Foundation -framework Metal -o $@

test: $(BIN) runtime-test domain-recycling-test moving-nest-test moving-nest-admission-test
	cd $(ROOT) && ./bin/test_metal_compute_backend
	cd $(ROOT) && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/metal/test_qualify_metal.py

runtime-test:
	$(MAKE) -C $(ROOT) bin/nimbus_sim
	cd $(ROOT) && NIMBUS_GPU_REQUIRED=1 PYTHONDONTWRITEBYTECODE=1 \
		python3 -m unittest tests/metal/test_metal_runtime.py

domain-recycling-test:
	$(MAKE) -C $(ROOT) bin/test_core_nested_production_step
	cd $(ROOT) && NIMBUS_GPU_REQUIRED=1 ./bin/test_core_nested_production_step \
		"[metal][domain_recycling][production]"

moving-nest-test:
	$(MAKE) -C $(ROOT) bin/test_core_nested_production_step
	cd $(ROOT) && NIMBUS_GPU_REQUIRED=1 ./bin/test_core_nested_production_step \
		"[metal][movement][tracker]"

moving-nest-admission-test:
	$(MAKE) -C $(ROOT) bin/nimbus_sim_internal_qualification bin/nimbus_sim
	cd $(ROOT) && PYTHONDONTWRITEBYTECODE=1 python3 \
		scripts/tests/gpu/test_moving_nested_vulkan_admission.py

live-recycling-test:
	$(MAKE) -C $(ROOT) qualify-domain-recycling-metal-live

qualify: $(BIN) runtime-test domain-recycling-test moving-nest-test moving-nest-admission-test live-recycling-test
	cd $(ROOT) && python3 scripts/qualification/accelerators/qualify_metal.py \
		--binary bin/test_metal_compute_backend \
		--output $(METAL_QUALIFICATION_REPORT) --log $(METAL_QUALIFICATION_LOG)

clean:
	$(RM) $(BIN)
else
all test runtime-test domain-recycling-test moving-nest-test moving-nest-admission-test live-recycling-test qualify:
	@echo "[metal] SKIP: native Metal qualification requires macOS"

clean:
	@true
endif
