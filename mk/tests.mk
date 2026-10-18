# ---------------------------------------------------------------------------
# Catch2 Test Suite
# ---------------------------------------------------------------------------
# Header sensitivity: test binaries compile monolithically (no .d
# tracking), so header-only edits must still invalidate them. A stale
# suite green-lit pre-fix physics once (2026-06-12, science audit M7:
# only thermodynamics.hpp changed; the GPU loaded the recompiled shader
# at runtime while the CPU reference baked into the binary kept the old
# formula). Recipes filter the headers out of the compile line.
TEST_HDRS := $(shell find include src vulkan/include tests -name '*.hpp' -o -name '*.h' 2>/dev/null)
TEST_INFRA := tests/test_main.cpp tests/test_harness.cpp src/microphysics/grid_geometry.cpp $(TEST_HDRS) \
	src/core/runtime/simulation_state.cpp \
	$(COMPUTE_RUNTIME_SRC) $(SOURCE_MANIFEST) $(TEST_FLAGS_STAMP)
# Several CPU-reference tests live beside the viewer and include viewer-local
# headers that do not themselves require the Vulkan SDK. Keep that local include
# available without adding it to CPU-only production artifacts.
TEST_CXXFLAGS := $(CXXFLAGS) -O0 -I tests -I vulkan/include

$(TEST_FLAGS_STAMP): FORCE
	@mkdir -p $(dir $@)
	@tmp="$@.tmp"; \
	  compiler_path="$$(command -v $(firstword $(CXX)) 2>/dev/null || printf '%s' '$(firstword $(CXX))')"; \
	  compiler_version="$$($(CXX) --version 2>/dev/null | sed -n '1p')"; \
	  { printf '%s\n' 'compiler=$(CXX)'; \
	    printf '%s\n' "compiler_path=$$compiler_path"; \
	    printf '%s\n' "compiler_version=$$compiler_version"; \
	    printf '%s\n' 'cxxflags=$(TEST_CXXFLAGS) $(CPPFLAGS)'; \
	    printf '%s\n' 'ldflags=$(LDFLAGS) $(LDLIBS)'; } > "$$tmp"; \
	  if ! cmp -s "$$tmp" "$@"; then mv "$$tmp" "$@"; else rm -f "$$tmp"; fi

# Core tests
bin/test_core_field: $(TEST_INFRA) tests/core/test_field3d.cpp tests/core/test_field_pool.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_parallel_copy: $(TEST_INFRA) tests/core/test_parallel_copy.cpp include/util/parallel_copy.hpp $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/benchmark_acoustic_snapshot_copy: scripts/benchmarks/cpu/benchmark_acoustic_snapshot_copy.cpp include/util/parallel_copy.hpp include/core/field/field3d.hpp $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_output: $(TEST_INFRA) tests/core/test_output_config.cpp src/core/output/output_config.cpp src/core/output/npy_writer.cpp src/diagnostics/derived/derived_slice_compute.cpp src/diagnostics/contracts/field_contract.cpp src/terrain/base/conservative_transport.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/benchmark_checkpoint_publication: scripts/benchmarks/storage/benchmark_checkpoint_publication.cpp src/core/infra/sha256.cpp include/core/runtime/checkpoint_format.hpp $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_checkpoint_wire: $(TEST_INFRA) tests/core/test_checkpoint_wire.cpp src/core/infra/sha256.cpp include/core/runtime/checkpoint_format.hpp | bin
	$(CXX_COMPILE) $(filter-out -O0,$(TEST_CXXFLAGS)) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_durable_file: $(TEST_INFRA) tests/core/test_durable_file.cpp src/core/infra/sha256.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_thompson_collection_tables: $(TEST_INFRA) tests/physics/test_thompson_collection_tables.cpp src/microphysics/schemes/thompson/collection_tables.cpp src/core/infra/sha256.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_thompson_activation_table: $(TEST_INFRA) tests/physics/test_thompson_activation_table.cpp src/microphysics/schemes/thompson/activation_table.cpp src/core/infra/sha256.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_thompson_aerosol_scavenging: $(TEST_INFRA) tests/physics/test_thompson_aerosol_scavenging.cpp src/microphysics/schemes/thompson/aerosol_scavenging.cpp src/microphysics/schemes/thompson/particle_distributions.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_thompson_cloud_evaporation: $(TEST_INFRA) tests/physics/test_thompson_cloud_evaporation.cpp src/microphysics/schemes/thompson/cloud_evaporation.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_thompson_cold_process_tables: $(TEST_INFRA) tests/physics/test_thompson_cold_process_tables.cpp src/microphysics/schemes/thompson/cold_process_tables.cpp src/core/infra/sha256.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_thompson_cold_processes: $(TEST_INFRA) tests/physics/test_thompson_cold_processes.cpp src/microphysics/schemes/thompson/cold_processes.cpp src/microphysics/schemes/thompson/cold_process_tables.cpp src/core/infra/sha256.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_thompson_collection_processes: $(TEST_INFRA) tests/physics/test_thompson_collection_processes.cpp src/microphysics/schemes/thompson/collection_processes.cpp src/microphysics/schemes/thompson/collection_tables.cpp src/core/infra/sha256.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_thompson_frozen_melting: $(TEST_INFRA) tests/physics/test_thompson_frozen_melting.cpp src/microphysics/schemes/thompson/frozen_melting.cpp src/microphysics/schemes/thompson/thermodynamic_helpers.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/thompson_activation_table_tool: src/tools/thompson_activation_table_tool.cpp src/microphysics/schemes/thompson/activation_table.cpp src/core/infra/sha256.cpp $(TEST_HDRS) | bin
	$(CXX_COMPILE) $(CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/thompson_collection_table_tool: src/tools/thompson_collection_table_tool.cpp src/microphysics/schemes/thompson/collection_tables.cpp src/core/infra/sha256.cpp $(TEST_HDRS) | bin
	$(CXX_COMPILE) $(CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/thompson_cold_table_tool: src/tools/thompson_cold_table_tool.cpp src/microphysics/schemes/thompson/cold_process_tables.cpp src/core/infra/sha256.cpp $(TEST_HDRS) | bin
	$(CXX_COMPILE) $(CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

THOMPSON_TABLE_THREADS ?= 8
.PHONY: thompson-data
thompson-data:
	python3 scripts/dependencies/fetch_thompson_tables.py

.PHONY: thompson-generate-data
thompson-generate-data: bin/thompson_activation_table_tool bin/thompson_collection_table_tool bin/thompson_cold_table_tool
	mkdir -p data/thompson
	./bin/thompson_activation_table_tool tests/fixtures/reference/thompson/CCN_ACTIVATE.BIN data/thompson/ccn_activation.ntact
	OMP_NUM_THREADS=$(THOMPSON_TABLE_THREADS) ./bin/thompson_collection_table_tool data/thompson/collection_v1.ntctbl
	OMP_NUM_THREADS=$(THOMPSON_TABLE_THREADS) ./bin/thompson_cold_table_tool data/thompson/cold_process_v1.ntcold

.PHONY: test-thompson-data-determinism
test-thompson-data-determinism: bin/thompson_collection_table_tool
	mkdir -p /tmp/nimbus-thompson-table-determinism
	OMP_NUM_THREADS=1 ./bin/thompson_collection_table_tool /tmp/nimbus-thompson-table-determinism/collection-t1.ntctbl
	OMP_NUM_THREADS=4 ./bin/thompson_collection_table_tool /tmp/nimbus-thompson-table-determinism/collection-t4.ntctbl
	cmp /tmp/nimbus-thompson-table-determinism/collection-t1.ntctbl /tmp/nimbus-thompson-table-determinism/collection-t4.ntctbl

bin/test_physics_thompson_nucleation: $(TEST_INFRA) tests/physics/test_thompson_nucleation.cpp src/microphysics/schemes/thompson/nucleation.cpp src/microphysics/schemes/thompson/moment_initialization.cpp src/microphysics/schemes/thompson/activation_table.cpp src/core/infra/sha256.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_thompson_particle_distributions: $(TEST_INFRA) tests/physics/test_thompson_particle_distributions.cpp src/microphysics/schemes/thompson/particle_distributions.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_thompson_prognostic_state: $(TEST_INFRA) tests/physics/test_thompson_prognostic_state.cpp src/microphysics/schemes/thompson/prognostic_state.cpp src/microphysics/schemes/thompson/moment_initialization.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_thompson_required_packaged_data: $(TEST_INFRA) tests/physics/test_thompson_required_packaged_data.cpp src/microphysics/schemes/thompson/required_packaged_data.cpp src/microphysics/schemes/thompson/activation_table.cpp src/microphysics/schemes/thompson/collection_tables.cpp src/microphysics/schemes/thompson/cold_process_tables.cpp src/core/infra/sha256.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_thompson_thermodynamic_helpers: $(TEST_INFRA) tests/physics/test_thompson_thermodynamic_helpers.cpp src/microphysics/schemes/thompson/thermodynamic_helpers.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_thompson_warm_rain_processes: $(TEST_INFRA) tests/physics/test_thompson_warm_rain_processes.cpp src/microphysics/schemes/thompson/warm_rain_processes.cpp src/microphysics/schemes/thompson/thermodynamic_helpers.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

.PHONY: test-thompson-warm-correctness
test-thompson-warm-correctness: bin/nimbus_sim bin/test_physics_thompson_warm_rain_processes bin/test_physics_thompson_activation_table bin/test_physics_thompson_cloud_evaporation bin/test_physics_microphysics
	./bin/test_physics_thompson_warm_rain_processes
	./bin/test_physics_thompson_activation_table
	./bin/test_physics_thompson_cloud_evaporation
	./bin/test_physics_microphysics "[thompson]"
	python3 scripts/tests/science/test_thompson_cpu_ready.py --binary $(PHYSICAL_QUALIFICATION_BINARY)
	python3 scripts/tests/science/test_agrid_thompson_ready.py --binary $(PHYSICAL_QUALIFICATION_BINARY)
	python3 scripts/tests/science/test_thompson_maintained_profiles.py --binary $(PHYSICAL_QUALIFICATION_BINARY) --report reports/pre_beta/thompson_warm_regression_v1.json

bin/test_core_hardware: $(TEST_INFRA) tests/core/test_hardware_info.cpp src/core/infra/hardware_info.cpp src/core/infra/simd_utils.cpp include/core/infra/hardware_info.hpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_packaged_runtime: tests/test_main.cpp tests/core/test_packaged_runtime.cpp include/util/packaged_runtime.hpp $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_numeric_text: tests/test_main.cpp tests/core/test_numeric_text.cpp include/util/numeric_text.hpp $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

ifeq ($(UNAME_S),Darwin)
bin/test_core_numeric_text_oracle: tests/test_main.cpp tests/core/test_numeric_text.cpp include/util/numeric_text.hpp $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) -mmacosx-version-min=13.3 $(filter %.cpp,$^) $(LDLIBS) -o $@
endif

bin/test_core_runtime_cadence: $(TEST_INFRA) tests/core/test_runtime_cadence.cpp src/core/runtime/simulation_state.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_scheme_catalog: $(TEST_INFRA) tests/core/test_scheme_catalog.cpp src/core/runtime/readiness.cpp src/core/runtime/scheme_catalog.cpp src/core/output/output_config.cpp src/diagnostics/contracts/field_contract.cpp src/core/infra/sha256.cpp include/core/runtime/flat_mixed_metal_scope_v1.hpp tests/fixtures/reference/catalog/flat_mixed_metal_cadence1_scope_v1.json | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/nimbus_generated_backend_tuples: tests/compute/generated_backend_tuple_catalog_main.cpp src/core/infra/sha256.cpp $(TEST_HDRS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_simulation_checkpoint: $(TEST_INFRA) tests/core/test_simulation_checkpoint.cpp src/core/infra/domain_recycling.cpp src/core/infra/domain_recycling_provenance.cpp src/core/infra/nested_execution.cpp src/core/infra/nested_terrain_geometry.cpp src/core/infra/nested_grid.cpp src/core/infra/nested_feedback.cpp src/core/infra/nested_scalar_feedback.cpp src/core/infra/nested_boundary_forcing.cpp src/core/infra/sha256.cpp src/core/output/npy_writer.cpp src/diagnostics/budgets/energy_budget.cpp src/diagnostics/trajectories/trajectories.cpp $(CHECKPOINT_SCHEME_SRCS) $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

# This target intentionally uses the production optimization level. Its exact
# alias patterns previously invoked undefined behavior only visible to an
# optimizing compiler, so compiling it through TEST_CXXFLAGS (-O0) defeats the
# regression.
bin/test_core_simd_aliasing: $(TEST_INFRA) tests/core/test_simd_aliasing.cpp src/core/infra/simd_utils.cpp | bin
	$(CXX_COMPILE) $(CXXFLAGS) -DNDEBUG -I tests $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_eos_pressure: $(TEST_INFRA) tests/core/test_eos_pressure.cpp \
	src/core/orchestration/dynamics/eos_apply.cpp \
	src/core/runtime/simulation_state.cpp \
	src/core/infra/coordinate_system.cpp \
	$(TERRAIN_TRANSPORT_SRC) | bin
	$(CXX_COMPILE) $(CXXFLAGS) -DNDEBUG -I tests $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_npy: $(TEST_INFRA) tests/core/test_npy_writer.cpp tests/core/test_npy_reader.cpp src/core/io/npy_reader.cpp src/core/output/npy_writer.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_output_writer_lifetime: tests/core/output_writer_lifetime_probe.cpp $(filter-out tests/test_main.cpp,$(TEST_INFRA)) $(OUTPUT_WRITER_SRCS) $(filter-out src/core/infra/allocation_telemetry.cpp,$(OPT_IN_SRCS)) src/diagnostics/derived/tranche_product_registry.cpp src/diagnostics/derived/column_water_paths.cpp src/terrain/base/conservative_transport.cpp | bin
	$(CXX_COMPILE) $(CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_sha256_probe: tests/core/sha256_probe.cpp src/core/infra/sha256.cpp include/core/infra/sha256.hpp $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_sha256_portable: tests/core/sha256_probe.cpp src/core/infra/sha256.cpp include/core/infra/sha256.hpp $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(CXXFLAGS) $(CPPFLAGS) -DNIMBUS_SHA256_PORTABLE_ONLY $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_output_writer: $(TEST_INFRA) tests/core/test_output_writer.cpp $(OUTPUT_WRITER_SRCS) $(OPT_IN_SRCS) src/diagnostics/derived/tranche_product_registry.cpp src/diagnostics/derived/column_water_paths.cpp src/terrain/base/conservative_transport.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_netcdf_history_writer: $(TEST_INFRA) tests/core/test_netcdf_history_writer.cpp src/core/output/netcdf_history_writer.cpp src/diagnostics/contracts/field_contract.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/netcdf_history_fixture: tests/qualification/netcdf_history_fixture.cpp src/core/output/netcdf_history_writer.cpp src/core/output/npy_writer.cpp src/core/output/history_index.cpp src/core/infra/sha256.cpp src/diagnostics/contracts/field_contract.cpp include/core/output/netcdf_history_writer.hpp include/core/output/npy_writer.hpp include/core/output/step_manifest.hpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_zfp_roundtrip: $(TEST_INFRA) tests/core/test_zfp_roundtrip.cpp $(OPT_IN_SRCS) $(OUTPUT_WRITER_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_zfp_benchmark: $(TEST_INFRA) tests/core/test_zfp_benchmark.cpp $(OPT_IN_SRCS) $(OUTPUT_WRITER_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_shm: $(TEST_INFRA) tests/core/test_shm_transport.cpp src/core/output/shm_writer.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_coordinate_system: $(TEST_INFRA) tests/core/test_coordinate_system.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_domain_motion: $(TEST_INFRA) tests/core/test_domain_motion.cpp src/core/infra/domain_motion.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@


bin/test_core_domain_recycling_swath: $(TEST_INFRA) tests/core/test_domain_recycling_swath.cpp src/core/output/domain_recycling_swath.cpp src/core/infra/domain_recycling.cpp src/core/infra/domain_recycling_provenance.cpp src/core/infra/domain_motion.cpp src/core/output/zfp_reader.cpp $(OUTPUT_WRITER_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_stagger_interpolation: $(TEST_INFRA) tests/core/test_stagger_interpolation.cpp src/core/output/stagger_interpolation.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_rayleigh_damping_yaml: $(TEST_INFRA) tests/core/test_rayleigh_damping_yaml.cpp src/core/infra/rayleigh_damping.cpp src/core/runtime/simulation_state.cpp src/core/infra/coordinate_system.cpp src/terrain/base/conservative_transport.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_nested_grid: $(TEST_INFRA) tests/core/test_nested_grid.cpp src/core/infra/nested_execution.cpp src/core/infra/nested_terrain_geometry.cpp src/core/infra/nested_grid.cpp src/core/infra/nested_boundary_forcing.cpp src/core/infra/nested_feedback.cpp src/core/infra/nested_scalar_feedback.cpp src/numerics/advection/cartesian_interface_flux_register.cpp src/core/runtime/simulation_state.cpp src/core/infra/coordinate_system.cpp src/core/infra/domain_motion.cpp src/diagnostics/budgets/energy_budget.cpp src/terrain/base/topography.cpp src/terrain/base/earth_fixed_geometry.cpp src/terrain/base/earth_fixed_volume_geometry.cpp src/terrain/base/conservative_transport.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_nested_terrain_geometry: $(TEST_INFRA) tests/core/test_nested_terrain_geometry.cpp src/core/infra/nested_terrain_geometry.cpp src/core/infra/nested_grid.cpp src/core/runtime/simulation_state.cpp src/core/infra/coordinate_system.cpp src/core/infra/domain_motion.cpp src/terrain/base/earth_fixed_geometry.cpp src/terrain/base/earth_fixed_volume_geometry.cpp src/terrain/base/topography.cpp src/terrain/base/conservative_transport.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

NESTED_PRODUCTION_TEST_OBJS := $(filter-out \
	$(BUILDDIR)/src/core/runtime/nimbus_sim.o,$(OBJS))

bin/test_core_nested_level_runtime: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/core/test_nested_level_runtime.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_physics_richardson_smagorinsky: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_richardson_smagorinsky.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_physics_native_tke: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_tke.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-thompson-phase
test-native-thompson-phase: bin/test_physics_native_thompson_phase bin/test_physics_native_thompson_sedimentation
	./bin/test_physics_native_thompson_phase "[native-thompson-phase][cpu]"
	./bin/test_physics_native_thompson_phase "[native-thompson-phase][metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_phase "[native-thompson-phase][vulkan]"
	./scripts/run_with_vulkan_env.sh env NIMBUS_VULKAN_MEMORY_POLICY=staged ./bin/test_physics_native_thompson_phase "[native-thompson-phase][vulkan]"
	./bin/test_physics_native_thompson_sedimentation "[native-thompson][metal][coupling]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_sedimentation "[native-thompson][vulkan][coupling]"

.PHONY: test-native-thompson-freezing
test-native-thompson-freezing: bin/test_physics_native_thompson_freezing bin/test_physics_native_thompson_sedimentation
	./bin/test_physics_native_thompson_freezing "[native-thompson-freezing][cpu]"
	./bin/test_physics_native_thompson_freezing "[native-thompson-freezing][metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_freezing "[native-thompson-freezing][vulkan]"
	./scripts/run_with_vulkan_env.sh env NIMBUS_VULKAN_MEMORY_POLICY=staged ./bin/test_physics_native_thompson_freezing "[native-thompson-freezing][vulkan]"
	./bin/test_physics_native_thompson_sedimentation "[native-thompson][metal][coupling]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_sedimentation "[native-thompson][vulkan][coupling]"

.PHONY: test-native-thompson-deposition
test-native-thompson-deposition: bin/test_physics_native_thompson_deposition bin/test_physics_native_thompson_sedimentation bin/test_physics_thompson_cold_processes
	./bin/test_physics_thompson_cold_processes
	./bin/test_physics_native_thompson_deposition "[native-thompson-deposition][cpu]"
	./bin/test_physics_native_thompson_deposition "[native-thompson-deposition][metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_deposition "[native-thompson-deposition][vulkan]"
	./scripts/run_with_vulkan_env.sh env NIMBUS_VULKAN_MEMORY_POLICY=staged ./bin/test_physics_native_thompson_deposition "[native-thompson-deposition][vulkan]"
	./bin/test_physics_native_thompson_sedimentation "[native-thompson][metal][coupling]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_sedimentation "[native-thompson][vulkan][coupling]"

bin/test_physics_native_thompson_deposition: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_thompson_deposition.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_physics_native_thompson_freezing: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_thompson_freezing.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_physics_native_thompson_phase: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_thompson_phase.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_physics_native_thompson_warm: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_thompson_warm.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-thompson-warm
test-native-thompson-warm: bin/test_physics_native_thompson_warm bin/test_physics_native_thompson_sedimentation
	./bin/test_physics_native_thompson_warm "[native-thompson-warm][cpu]"
	./bin/test_physics_native_thompson_warm "[native-thompson-warm][metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_warm "[native-thompson-warm][vulkan]"
	./scripts/run_with_vulkan_env.sh env NIMBUS_VULKAN_MEMORY_POLICY=staged ./bin/test_physics_native_thompson_warm "[native-thompson-warm][vulkan]"
	./bin/test_physics_native_thompson_sedimentation "[native-thompson][metal][coupling]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_sedimentation "[native-thompson][vulkan][coupling]"

bin/test_physics_native_thompson_cloud: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_thompson_cloud.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-thompson-cloud
test-native-thompson-cloud: bin/test_physics_native_thompson_cloud bin/test_physics_native_thompson_sedimentation
	./bin/test_physics_native_thompson_cloud "[native-thompson-cloud][contract]"
	./bin/test_physics_native_thompson_cloud "[native-thompson-cloud][metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_cloud "[native-thompson-cloud][vulkan]"
	./scripts/run_with_vulkan_env.sh env NIMBUS_VULKAN_MEMORY_POLICY=staged ./bin/test_physics_native_thompson_cloud "[native-thompson-cloud][vulkan]"
	./bin/test_physics_native_thompson_sedimentation "[native-thompson][metal][coupling]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_sedimentation "[native-thompson][vulkan][coupling]"
	python3 scripts/dev/shaders/generate_thompson_msl.py --check

bin/test_physics_native_thompson_sedimentation: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_thompson_sedimentation.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_numerics_native_implicit: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/numerics/test_native_implicit.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_core_earth_fixed_step_transaction: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/core/test_earth_fixed_step_transaction.cpp tests/core/test_earth_fixed_remap_reference.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/nimbus_native_admission_batch: tests/compute/native_admission_batch_main.cpp src/core/runtime/nimbus_sim.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) \
	  tests/compute/native_admission_batch_main.cpp $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-generated-admission-audit
test-generated-admission-audit: bin/nimbus_sim bin/nimbus_native_admission_batch
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_generated_admission_audit.py

.PHONY: test-generated-metal-east-npy-scope
test-generated-metal-east-npy-scope: bin/nimbus_sim bin/nimbus_native_admission_batch bin/test_init_config_v1 bin/test_core_scheme_catalog
	./bin/test_init_config_v1 '[generated-metal-east-npy]'
	./bin/test_core_scheme_catalog '[generated-metal-zfp]'
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_generated_metal_east_npy_scope.py --binary ./bin/nimbus_sim --helper ./bin/nimbus_native_admission_batch

.PHONY: test-generated-metal-public-scope
test-generated-metal-public-scope: bin/nimbus_sim test-qualified-zfp-explicit-tolerances test-generated-metal-east-npy-scope
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_generated_metal_public_scope.py --binary ./bin/nimbus_sim

.PHONY: test-generated-metal-cli-scope
test-generated-metal-cli-scope: bin/nimbus_sim test-generated-metal-east-npy-scope
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_generated_metal_cli_scope.py --binary ./bin/nimbus_sim

.PHONY: test-qualified-zfp-explicit-tolerances
test-qualified-zfp-explicit-tolerances: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_qualified_zfp_explicit_tolerances.py --binary ./bin/nimbus_sim

.PHONY: test-retained-fixture-materialization
test-retained-fixture-materialization: bin/nimbus_sim bin/nimbus_native_admission_batch
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest scripts.tests.catalog.test_retained_fixture_shards scripts.tests.catalog.test_retained_fixture_materializer scripts.tests.catalog.test_backend_completion_inventory scripts.tests.catalog.test_loader_categorical_axes scripts.tests.catalog.test_config_source_axis_survey
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_retained_fixture_materialization.py --binary ./bin/nimbus_sim --native ./bin/nimbus_native_admission_batch

.PHONY: test-closed-sgs-boundary
test-closed-sgs-boundary: bin/nimbus_sim bin/nimbus_native_admission_batch bin/test_physics_turbulence_sgs bin/test_vulkan_gpu_parity
	./bin/test_physics_turbulence_sgs '[closed_scalar]'
	./scripts/run_with_vulkan_env.sh ./bin/test_vulkan_gpu_parity '[sgs_scalar_boundary]'
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_closed_sgs_boundary.py --binary bin/nimbus_sim

bin/benchmark_checkpoint_restore: scripts/benchmarks/storage/benchmark_checkpoint_restore.cpp tests/support/production_wind_profile.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/benchmark_checkpoint_read: scripts/benchmarks/storage/benchmark_checkpoint_read.cpp tests/support/production_wind_profile.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/nimbus_private_checkpoint_probe: tests/core/private_checkpoint_probe.cpp tests/support/production_wind_profile.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_native_domain_recycling: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/compute/test_native_domain_recycling.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-domain-recycling
test-native-domain-recycling: bin/test_native_domain_recycling
	./scripts/run_with_vulkan_env.sh ./bin/test_native_domain_recycling
	NIMBUS_VULKAN_MEMORY_POLICY=staged ./scripts/run_with_vulkan_env.sh ./bin/test_native_domain_recycling


.PHONY: test-native-slow-attribution
test-native-slow-attribution: bin/test_core_nested_production_step bin/test_dynamics_cartesian_cgrid_slow_tendencies_gpu_parity
	./bin/test_core_nested_production_step "[native_slow][contract],[native_slow][atomicity]"
	./scripts/run_with_vulkan_env.sh ./bin/test_core_nested_production_step "[native_slow][device]"
	NIMBUS_VULKAN_MEMORY_POLICY=staged ./scripts/run_with_vulkan_env.sh ./bin/test_core_nested_production_step "[native_slow][device]"
	./scripts/run_with_vulkan_env.sh ./bin/test_dynamics_cartesian_cgrid_slow_tendencies_gpu_parity

MAPPED_RECYCLING_EVIDENCE ?= /tmp/nimbus-mapped-parent-recycling-ready
.PHONY: test-mapped-nested-recycling-ready
test-mapped-nested-recycling-ready: test-native-slow-attribution bin/nimbus_sim bin/test_native_domain_recycling bin/test_core_nested_grid bin/test_core_nested_terrain_geometry bin/test_core_domain_recycling bin/test_core_nested_production_step bin/test_physics_chaos_runtime_state
	./bin/test_core_nested_grid
	./bin/test_core_nested_terrain_geometry
	./bin/test_core_domain_recycling
	./bin/test_core_nested_production_step "[chaos]"
	./bin/test_physics_chaos_runtime_state "[atomicity]"
	./scripts/run_with_vulkan_env.sh ./bin/test_native_domain_recycling
	NIMBUS_VULKAN_MEMORY_POLICY=staged ./scripts/run_with_vulkan_env.sh ./bin/test_native_domain_recycling
	python3 scripts/tests/science/test_mapped_nested_recycling_ready.py --binary bin/nimbus_sim --output $(MAPPED_RECYCLING_EVIDENCE)

bin/test_core_domain_recycling: tests/test_main.cpp tests/core/test_domain_recycling.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_core_nested_production_step: tests/test_main.cpp \
	tests/core/test_nested_production_step.cpp \
	$(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) \
		$(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

$(BUILDDIR)/qualification/flat_mixed_preallocation/equations.o: src/core/orchestration/dynamics/equations.cpp $(TEST_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) -Dresize_fields=nimbus_test_preserved_resize_fields -c $< -o $@

bin/test_core_flat_mixed_preallocation: tests/test_main.cpp \
	tests/support/production_wind_profile.cpp tests/core/test_flat_mixed_preallocation.cpp \
	$(filter-out $(BUILDDIR)/src/core/orchestration/dynamics/equations.o $(BUILDDIR)/metal/src/compute/metal_compute_backend.o,$(NESTED_PRODUCTION_TEST_OBJS)) \
	$(BUILDDIR)/qualification/flat_mixed_preallocation/equations.o $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) -DNIMBUS_ENABLE_METAL \
		$(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) -o $@

bin/test_core_flat_mixed_metal_admission: tests/test_main.cpp \
	tests/core/test_flat_mixed_metal_admission.cpp src/init/config/v1_loader.cpp \
	$(CONFIG_CODEC_SRCS) $(CONFIG_MIGRATION_SRCS) src/core/runtime/yaml_helpers.cpp \
	src/core/infra/coordinate_system.cpp src/core/infra/sha256.cpp \
	include/core/runtime/flat_mixed_metal_admission.hpp \
	include/core/runtime/qualified_generated_metal_cli_scope.hpp $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_validate_only_context: tests/test_main.cpp tests/core/test_validate_only_context.cpp \
	src/init/config/v1_loader.cpp $(CONFIG_CODEC_SRCS) $(CONFIG_MIGRATION_SRCS) \
	src/core/runtime/yaml_helpers.cpp src/core/infra/coordinate_system.cpp src/core/infra/sha256.cpp \
	include/core/runtime/validate_only_context.hpp include/core/runtime/yaml_helpers.hpp $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_radar_zfp_v4_integration: tests/test_main.cpp \
	tests/support/production_wind_profile.cpp \
	tests/core/test_radar_zfp_v4_integration.cpp \
	src/core/output/zfp_reader.cpp \
	$(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) \
		$(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_core_nested_boundary_forcing: $(TEST_INFRA) tests/core/test_nested_boundary_forcing.cpp src/core/infra/nested_boundary_forcing.cpp src/core/infra/nested_grid.cpp src/core/runtime/simulation_state.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

# Diagnostics tests
bin/test_diagnostics_storm_tracker: $(TEST_INFRA) tests/diagnostics/test_storm_tracker.cpp src/diagnostics/storm_tracker.cpp src/diagnostics/storm_rotation.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_diagnostics_contract: $(TEST_INFRA) tests/diagnostics/test_field_contract.cpp src/diagnostics/contracts/field_contract.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_diagnostics_cartesian_derivatives: $(TEST_INFRA) tests/diagnostics/test_cartesian_horizontal_derivatives.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_diagnostics_column_water_paths: $(TEST_INFRA) tests/diagnostics/test_column_water_paths.cpp src/diagnostics/derived/column_water_paths.cpp src/core/runtime/simulation_state.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_diagnostics_potential_vorticity: $(TEST_INFRA) tests/diagnostics/test_potential_vorticity.cpp src/terrain/base/conservative_transport.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_diagnostics_validation: $(TEST_INFRA) tests/diagnostics/test_field_validation.cpp $(VALIDATION_SRCS) src/core/infra/field_sanitization.cpp src/core/runtime/simulation_state.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_diagnostics_logging_ratchet: $(TEST_INFRA) tests/diagnostics/test_logging_ratchet.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_diagnostics_trajectories: $(TEST_INFRA) tests/diagnostics/test_trajectories.cpp src/diagnostics/trajectories/trajectories.cpp src/core/infra/domain_motion.cpp src/core/output/npy_writer.cpp src/terrain/base/conservative_transport.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_diagnostics_conservation_budget: $(TEST_INFRA) tests/diagnostics/test_conservation_budget.cpp src/diagnostics/budgets/conservation_budget.cpp src/diagnostics/budgets/moisture_budget.cpp src/core/runtime/simulation_state.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_diagnostics_momentum_budget: $(TEST_INFRA) tests/diagnostics/test_momentum_budget.cpp src/diagnostics/budgets/momentum_budget.cpp src/core/runtime/simulation_state.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_diagnostics_energy_budget: $(TEST_INFRA) tests/diagnostics/test_energy_budget.cpp src/core/orchestration/dynamics/pressure_cell_energy.hpp src/diagnostics/budgets/energy_budget.cpp src/diagnostics/budgets/velocity_energy_pair.cpp src/core/runtime/simulation_state.cpp src/terrain/base/topography.cpp src/terrain/base/conservative_transport.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_diagnostics_energy_pair_optimized: $(TEST_INFRA) tests/diagnostics/test_energy_budget.cpp src/diagnostics/budgets/energy_budget.cpp src/diagnostics/budgets/velocity_energy_pair.cpp src/core/runtime/simulation_state.cpp src/terrain/base/topography.cpp src/terrain/base/conservative_transport.cpp | bin
	$(CXX_COMPILE) $(filter-out -O0,$(TEST_CXXFLAGS)) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_diagnostics_moisture_budget: $(TEST_INFRA) tests/diagnostics/test_moisture_budget.cpp src/diagnostics/budgets/moisture_budget.cpp src/core/runtime/simulation_state.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

# Dynamics tests
bin/test_dynamics_cartesian: $(TEST_INFRA) tests/dynamics/test_cartesian_dynamics.cpp src/dynamics/schemes/klemp_wilhelmson_split/cartesian_agrid.cpp src/diagnostics/derived/resolved_diagnostics.cpp src/diagnostics/derived/pressure_projection.cpp $(TERRAIN_TRANSPORT_SRC) src/core/infra/coordinate_system.cpp src/core/infra/domain_motion.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_dynamics_cartesian_bcs: $(TEST_INFRA) tests/dynamics/test_cartesian_boundary_conditions.cpp src/dynamics/schemes/klemp_wilhelmson_split/cartesian_agrid.cpp src/boundary_conditions/boundary_conditions_cartesian.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_dynamics_collocated_mapped_timestep: $(TEST_INFRA) tests/dynamics/test_collocated_mapped_timestep.cpp $(TERRAIN_COLLOCATED_SRCS) $(TERRAIN_TRANSPORT_SRC) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_dynamics_cartesian_lateral_bc_yaml: $(TEST_INFRA) tests/dynamics/test_cartesian_lateral_bc_yaml.cpp $(BOUNDARY_CONDITION_SCHEME_SRCS) src/boundary_conditions/runtime.cpp $(VALIDATION_SRCS) src/core/infra/field_sanitization.cpp src/core/runtime/simulation_state.cpp src/core/infra/coordinate_system.cpp $(TERRAIN_TRANSPORT_SRC) src/dynamics/cartesian_cgrid_momentum_flux.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_dynamics_cartesian_lateral_bc_optimized: $(TEST_INFRA) tests/dynamics/test_cartesian_lateral_bc_yaml.cpp $(BOUNDARY_CONDITION_SCHEME_SRCS) src/boundary_conditions/runtime.cpp $(VALIDATION_SRCS) src/core/infra/field_sanitization.cpp src/core/runtime/simulation_state.cpp src/core/infra/coordinate_system.cpp $(TERRAIN_TRANSPORT_SRC) src/dynamics/cartesian_cgrid_momentum_flux.cpp | bin
	$(CXX_COMPILE) $(filter-out -O0,$(TEST_CXXFLAGS)) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_dynamics_cartesian_vertical_bc_yaml: $(TEST_INFRA) tests/dynamics/test_cartesian_vertical_bc_yaml.cpp $(BOUNDARY_CONDITION_SCHEME_SRCS) src/terrain/base/topography.cpp $(TERRAIN_TRANSPORT_SRC) src/dynamics/cartesian_cgrid_momentum_flux.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_dynamics_coriolis_yaml: $(TEST_INFRA) tests/dynamics/test_coriolis_yaml.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

DYNAMICS_FACTORY_TEST_SUPPORT_SRCS := src/numerics/time_stepping/native_rk.cpp \
	src/core/infra/domain_motion.cpp \
	src/terrain/base/topography.cpp \
	src/terrain/base/conservative_transport.cpp \
	src/terrain/base/earth_fixed_geometry.cpp \
	src/terrain/base/earth_fixed_volume_geometry.cpp \
	src/terrain/base/native_mapped_transaction.cpp \
	src/core/orchestration/dynamics/eos_apply.cpp \
	src/dynamics/device_slow_energy_attribution.cpp \
	src/diagnostics/budgets/energy_budget.cpp \
	src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp

bin/test_dynamics_factory_v1: $(TEST_INFRA) tests/dynamics/test_dynamics_factory_v1.cpp $(DYNAMICS_KERNEL_SRCS) $(DYNAMICS_FACTORY_TEST_SUPPORT_SRCS) $(CONFIG_CODEC_SRCS) src/core/infra/coordinate_system.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_dynamics_cartesian_ic: $(TEST_INFRA) tests/dynamics/test_cartesian_initial_conditions.cpp src/core/orchestration/dynamics/initial_conditions_cartesian.cpp src/core/runtime/simulation_state.cpp src/terrain/base/topography.cpp $(TERRAIN_TRANSPORT_SRC) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_dynamics_acoustic_cartesian_cgrid_gpu_parity: $(TEST_INFRA) tests/dynamics/test_acoustic_cartesian_cgrid_gpu_parity.cpp $(BOUNDARY_CONDITION_SCHEME_SRCS) src/dynamics/cartesian_cgrid_momentum_flux.cpp src/dynamics/device_acoustic_energy_attribution.cpp src/core/runtime/simulation_state.cpp src/core/infra/coordinate_system.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_dynamics_cartesian_cgrid_slow_tendencies_gpu_parity: $(TEST_INFRA) tests/dynamics/test_cartesian_cgrid_slow_tendencies_gpu_parity.cpp src/numerics/time_stepping/native_rk.cpp src/dynamics/cartesian_cgrid_momentum_flux.cpp src/dynamics/moving_cgrid_transport.cpp src/dynamics/schemes/klemp_wilhelmson_split/cartesian_cgrid_3d.cpp src/dynamics/device_slow_energy_attribution.cpp src/core/orchestration/dynamics/thermodynamic_closure.cpp src/core/runtime/simulation_state.cpp src/terrain/base/topography.cpp $(TERRAIN_TRANSPORT_SRC) src/core/infra/coordinate_system.cpp $(BACKEND_COMMON_SRCS) src/terrain/base/earth_fixed_geometry.cpp src/terrain/base/earth_fixed_volume_geometry.cpp src/core/infra/domain_motion.cpp src/terrain/base/native_mapped_transaction.cpp src/diagnostics/budgets/energy_budget.cpp src/dynamics/native_cgrid_boundary.cpp src/core/orchestration/dynamics/eos_apply.cpp src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

# Numerics tests
bin/test_numerics_advection: $(TEST_INFRA) tests/numerics/test_advection_tvd.cpp $(ADVECTION_SRCS) src/core/runtime/simulation_state.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_numerics_tvd_monotonicity: $(TEST_INFRA) tests/numerics/test_tvd_monotonicity.cpp $(ADVECTION_SCHEME_SRCS) $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_numerics_advection_cartesian bin/test_numerics_advection_cartesian_optimized: $(TEST_INFRA) tests/numerics/test_advection_cartesian.cpp $(ADVECTION_SRCS) src/core/runtime/simulation_state.cpp src/microphysics/base/thermodynamics.cpp src/microphysics/schemes/milbrandt/milbrandt.cpp src/microphysics/schemes/milbrandt/native.cpp src/microphysics/schemes/milbrandt/triple_moment.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/schemes/milbrandt/triple_internal.cpp src/microphysics/schemes/milbrandt/triple_warm_internal.cpp src/microphysics/schemes/milbrandt/triple_cold_freezing.cpp src/microphysics/schemes/milbrandt/triple_cold_internal.cpp src/microphysics/schemes/milbrandt/triple_cold_diffusion.cpp src/microphysics/schemes/milbrandt/triple_warm_events.cpp src/microphysics/schemes/milbrandt/triple_warm_phase.cpp src/microphysics/schemes/milbrandt/triple_warm_adapter.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(if $(filter %_optimized,$@),$(filter-out -O0,$(TEST_CXXFLAGS)),$(TEST_CXXFLAGS)) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_numerics_advection_cartesian bin/test_numerics_advection_cartesian_optimized: src/microphysics/schemes/milbrandt/triple_cold_diffusion_internal.cpp src/microphysics/schemes/milbrandt/triple_cold_joint_internal.cpp src/microphysics/schemes/milbrandt/triple_melting.cpp src/microphysics/schemes/milbrandt/triple_melting_internal.cpp src/microphysics/schemes/milbrandt/triple_ice_melting.cpp src/microphysics/schemes/milbrandt/triple_ice_melting_internal.cpp src/microphysics/schemes/milbrandt/triple_snow_melting.cpp src/microphysics/schemes/milbrandt/triple_snow_melting_internal.cpp src/microphysics/schemes/milbrandt/triple_warm_primary.cpp src/microphysics/schemes/milbrandt/triple_warm_primary_internal.cpp
bin/test_numerics_advection_cartesian bin/test_numerics_advection_cartesian_optimized: src/microphysics/schemes/milbrandt/triple_cloud_adjustment.cpp src/microphysics/schemes/milbrandt/triple_cloud_adjustment_internal.cpp
bin/test_numerics_advection_cartesian bin/test_numerics_advection_cartesian_optimized: src/microphysics/schemes/milbrandt/triple_warm_cloud.cpp src/microphysics/schemes/milbrandt/triple_warm_cloud_internal.cpp
bin/test_numerics_advection_cartesian bin/test_numerics_advection_cartesian_optimized: src/microphysics/schemes/milbrandt/triple_frozen_collection.cpp src/microphysics/schemes/milbrandt/triple_frozen_collection_internal.cpp
bin/test_numerics_advection_cartesian bin/test_numerics_advection_cartesian_optimized: src/microphysics/schemes/milbrandt/triple_cold_cloud_collection.cpp src/microphysics/schemes/milbrandt/triple_cold_cloud_collection_internal.cpp
bin/test_numerics_advection_cartesian bin/test_numerics_advection_cartesian_optimized: src/microphysics/schemes/milbrandt/triple_cold_cloud_diffusion.cpp src/microphysics/schemes/milbrandt/triple_cold_cloud_diffusion_internal.cpp
bin/test_numerics_advection_cartesian bin/test_numerics_advection_cartesian_optimized: src/microphysics/schemes/milbrandt/triple_ice_cloud_collection.cpp src/microphysics/schemes/milbrandt/triple_rime_splintering.cpp src/microphysics/schemes/milbrandt/triple_cold_riming.cpp src/microphysics/schemes/milbrandt/triple_cold_riming_internal.cpp
bin/test_numerics_advection_cartesian bin/test_numerics_advection_cartesian_optimized: src/microphysics/schemes/milbrandt/triple_ice_graupel_conversion.cpp
bin/test_numerics_advection_cartesian bin/test_numerics_advection_cartesian_optimized: src/microphysics/schemes/milbrandt/triple_snow_graupel_conversion.cpp
bin/test_numerics_advection_cartesian bin/test_numerics_advection_cartesian_optimized: src/microphysics/schemes/milbrandt/triple_homogeneous_cloud_freezing.cpp src/microphysics/schemes/milbrandt/triple_graupel_hail_conversion.cpp

# weno5_face_value is a dependency-free header (weno5_face.hpp), so this test
# needs no advection sources - just the test harness.
bin/test_numerics_weno5_reconstruction: $(TEST_INFRA) tests/numerics/test_weno5_reconstruction.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_numerics_advect_conservative_cartesian_cgrid_gpu_parity: $(TEST_INFRA) tests/numerics/test_advect_conservative_cartesian_cgrid_gpu_parity.cpp src/numerics/advection/advection_cartesian.cpp src/core/infra/coordinate_system.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_numerics_diffusion: $(TEST_INFRA) tests/numerics/test_diffusion.cpp $(DIFFUSION_SCHEME_SRCS) $(TERRAIN_TRANSPORT_SRC) src/turbulence/base/eddy_viscosity.cpp src/turbulence/base/collocated_eddy_viscosity.cpp src/core/orchestration/physics/diffusion_step.cpp src/core/runtime/simulation_state.cpp src/core/infra/coordinate_system.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_numerics_timestepping: $(TEST_INFRA) tests/numerics/test_time_stepping.cpp $(TIME_STEPPING_SCHEME_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

RUNTIME_TIMESTEP_TEST_SRC ?= src/numerics/runtime.cpp
bin/test_numerics_runtime_timestep bin/test_numerics_runtime_timestep_optimized: tests/test_main.cpp $(TEST_HDRS) \
	$(COMPUTE_RUNTIME_SRC) $(SOURCE_MANIFEST) $(TEST_FLAGS_STAMP) \
	tests/numerics/test_runtime_timestep.cpp $(RUNTIME_TIMESTEP_TEST_SRC) src/core/runtime/simulation_state.cpp \
	src/numerics/diffusion/linear_height_profile.cpp \
	src/core/infra/coordinate_system.cpp \
	src/terrain/base/conservative_transport.cpp | bin
	$(CXX_COMPILE) $(if $(filter %_optimized,$@),$(filter-out -O0,$(TEST_CXXFLAGS)) -DNDEBUG,$(TEST_CXXFLAGS)) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_numerics_staggered_derivatives: $(TEST_INFRA) tests/numerics/test_staggered_derivatives.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_domain_recycling bin/test_core_nested_level_runtime \
bin/test_numerics_advection_cartesian bin/test_physics_microphysics bin/test_vulkan_gpu_parity: \
    src/microphysics/schemes/milbrandt/milbrandt.hpp \
    src/microphysics/schemes/milbrandt/warm_rain.hpp

# Physics tests
bin/test_physics_microphysics: $(TEST_INFRA) tests/physics/test_microphysics.cpp $(MICROPHYSICS_SCHEME_SRCS) src/core/infra/sha256.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_candidate: $(TEST_INFRA) tests/physics/test_milbrandt_triple_moment_candidate.cpp src/microphysics/schemes/milbrandt/triple_moment.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_sedimentation: $(TEST_INFRA) tests/physics/test_milbrandt_triple_sedimentation.cpp src/microphysics/schemes/milbrandt/triple_moment.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_five_sedimentation: $(TEST_INFRA) tests/physics/test_milbrandt_triple_five_sedimentation.cpp src/microphysics/schemes/milbrandt/triple_moment.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_internal: $(TEST_INFRA) tests/physics/test_milbrandt_triple_internal.cpp $(MICROPHYSICS_SCHEME_SRCS) src/core/infra/sha256.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_internal_runtime: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_milbrandt_triple_internal_runtime.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_internal_runtime: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_milbrandt_triple_cold_internal_runtime.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_diffusion_internal_runtime: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_milbrandt_triple_cold_diffusion_internal_runtime.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_joint_internal_runtime: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_milbrandt_triple_cold_joint_internal_runtime.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_diffusion: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cold_diffusion.cpp src/microphysics/schemes/milbrandt/triple_cold_diffusion.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_frozen_composition: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cold_frozen_composition.cpp src/microphysics/schemes/milbrandt/triple_cold_diffusion.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_melting: $(TEST_INFRA) tests/physics/test_milbrandt_triple_melting.cpp src/microphysics/schemes/milbrandt/triple_melting.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_ice_melting: $(TEST_INFRA) tests/physics/test_milbrandt_triple_ice_melting.cpp src/microphysics/schemes/milbrandt/triple_ice_melting.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_snow_melting: $(TEST_INFRA) tests/physics/test_milbrandt_triple_snow_melting.cpp src/microphysics/schemes/milbrandt/triple_snow_melting.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_warm_events: $(TEST_INFRA) tests/physics/test_milbrandt_triple_warm_events.cpp src/microphysics/schemes/milbrandt/triple_warm_events.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cloud_adjustment: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cloud_adjustment.cpp tests/support/milbrandt_cloud_adjustment_reference.hpp src/microphysics/schemes/milbrandt/triple_cloud_adjustment.hpp src/microphysics/schemes/milbrandt/activation.hpp src/microphysics/base/thermodynamics.hpp src/microphysics/schemes/milbrandt/triple_cloud_adjustment.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_warm_cloud: $(TEST_INFRA) tests/physics/test_milbrandt_triple_warm_cloud.cpp tests/support/milbrandt_cloud_adjustment_reference.hpp src/microphysics/schemes/milbrandt/triple_warm_cloud.hpp src/microphysics/schemes/milbrandt/triple_warm_primary.hpp src/microphysics/schemes/milbrandt/warm_rain.hpp src/microphysics/schemes/milbrandt/triple_warm_cloud.cpp src/microphysics/schemes/milbrandt/triple_cloud_adjustment.cpp src/microphysics/schemes/milbrandt/triple_warm_primary.cpp src/microphysics/schemes/milbrandt/triple_warm_events.cpp src/microphysics/schemes/milbrandt/triple_warm_phase.cpp src/microphysics/schemes/milbrandt/triple_warm_adapter.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

MILBRANDT_DYNAMIC_WARM_SRCS := src/microphysics/schemes/milbrandt/triple_warm_dynamic.cpp src/microphysics/schemes/milbrandt/triple_warm_dynamic_adapter.cpp src/microphysics/schemes/milbrandt/triple_warm_spectrum.cpp

MILBRANDT_DYNAMIC_WARM_SRCS += src/microphysics/schemes/milbrandt/triple_warm_subcycle.cpp

bin/test_physics_milbrandt_triple_frozen_collection: $(TEST_INFRA) tests/physics/test_milbrandt_triple_frozen_collection.cpp tests/support/milbrandt_frozen_collection_reference.hpp src/microphysics/schemes/milbrandt/triple_frozen_collection.hpp src/microphysics/schemes/milbrandt/cold.hpp src/microphysics/schemes/milbrandt/collision.hpp src/microphysics/schemes/milbrandt/distribution.hpp include/microphysics/milbrandt_cold_parameters.hpp src/microphysics/schemes/milbrandt/triple_frozen_collection.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_cloud_collection: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cold_cloud_collection.cpp tests/support/milbrandt_cold_cloud_collection_reference.hpp src/microphysics/schemes/milbrandt/triple_cold_cloud_collection.hpp src/microphysics/schemes/milbrandt/triple_frozen_collection.hpp src/microphysics/schemes/milbrandt/cold.hpp src/microphysics/schemes/milbrandt/collision.hpp src/microphysics/schemes/milbrandt/distribution.hpp include/microphysics/milbrandt_cold_parameters.hpp src/microphysics/schemes/milbrandt/triple_cold_cloud_collection.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_cloud_collection: $(wildcard src/microphysics/schemes/milbrandt/*.hpp) src/microphysics/base/thermodynamics.hpp include/core/infra/physical_constants.hpp tests/support/milbrandt_frozen_collection_reference.hpp

bin/test_physics_milbrandt_triple_cold_cloud_diffusion: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cold_cloud_diffusion.cpp tests/support/milbrandt_cold_cloud_diffusion_reference.hpp tests/support/milbrandt_cold_cloud_collection_reference.hpp tests/support/milbrandt_frozen_collection_reference.hpp $(wildcard src/microphysics/schemes/milbrandt/*.hpp) src/microphysics/base/thermodynamics.hpp include/core/infra/physical_constants.hpp include/microphysics/milbrandt_cold_parameters.hpp src/microphysics/schemes/milbrandt/triple_cold_cloud_diffusion.cpp src/microphysics/schemes/milbrandt/triple_cold_cloud_collection.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_warm_cloud bin/test_physics_milbrandt_triple_warm_phase_adapter bin/test_numerics_advection_cartesian bin/test_numerics_advection_cartesian_optimized: $(MILBRANDT_DYNAMIC_WARM_SRCS) src/microphysics/schemes/milbrandt/triple_warm_dynamic.hpp

bin/test_physics_milbrandt_triple_dynamic_warm: $(TEST_INFRA) tests/physics/test_milbrandt_triple_dynamic_warm.cpp tests/support/milbrandt_dynamic_warm_reference.hpp src/microphysics/schemes/milbrandt/warm_rain.hpp src/microphysics/schemes/milbrandt/triple_warm_dynamic.hpp src/microphysics/schemes/milbrandt/triple_warm_primary.hpp $(MILBRANDT_DYNAMIC_WARM_SRCS) src/microphysics/schemes/milbrandt/triple_warm_primary.cpp src/microphysics/schemes/milbrandt/triple_warm_events.cpp src/microphysics/schemes/milbrandt/triple_warm_phase.cpp src/microphysics/schemes/milbrandt/triple_warm_adapter.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_dynamic_warm: tests/support/milbrandt_cloud_adjustment_reference.hpp src/microphysics/schemes/milbrandt/triple_warm_cloud.hpp src/microphysics/schemes/milbrandt/triple_warm_cloud.cpp src/microphysics/schemes/milbrandt/triple_cloud_adjustment.cpp

bin/test_physics_milbrandt_triple_warm_subcycle: $(TEST_INFRA) tests/physics/test_milbrandt_triple_warm_subcycle.cpp tests/support/milbrandt_subcycle_reference.hpp src/microphysics/schemes/milbrandt/triple_warm_subcycle.hpp src/microphysics/schemes/milbrandt/triple_warm_primary.hpp src/microphysics/schemes/milbrandt/triple_warm_dynamic.hpp src/microphysics/schemes/milbrandt/warm_rain.hpp $(MILBRANDT_DYNAMIC_WARM_SRCS) src/microphysics/schemes/milbrandt/triple_warm_primary.cpp src/microphysics/schemes/milbrandt/triple_warm_events.cpp src/microphysics/schemes/milbrandt/triple_warm_phase.cpp src/microphysics/schemes/milbrandt/triple_warm_adapter.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp src/microphysics/schemes/milbrandt/triple_warm_cloud.cpp src/microphysics/schemes/milbrandt/triple_cloud_adjustment.cpp tests/support/milbrandt_cloud_adjustment_reference.hpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_warm_spectrum: $(TEST_INFRA) tests/physics/test_milbrandt_triple_warm_spectrum.cpp src/microphysics/schemes/milbrandt/triple_warm_spectrum.hpp src/microphysics/schemes/milbrandt/triple_warm_phase.hpp src/microphysics/schemes/milbrandt/triple_sedimentation.hpp include/microphysics/milbrandt_warm_parameters.hpp src/microphysics/schemes/milbrandt/triple_warm_spectrum.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_warm_observer: $(TEST_INFRA) tests/physics/test_milbrandt_warm_observer.cpp src/microphysics/schemes/milbrandt/warm_rain.hpp include/microphysics/milbrandt_warm_parameters.hpp src/microphysics/base/thermodynamics.cpp src/core/infra/sha256.cpp src/core/infra/allocation_telemetry.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_cold_observer: $(TEST_INFRA) tests/physics/test_milbrandt_cold_observer.cpp src/microphysics/base/thermodynamics.cpp src/core/infra/sha256.cpp src/core/infra/allocation_telemetry.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_cold_riming_repair: $(TEST_INFRA) tests/physics/test_milbrandt_cold_riming_repair.cpp tests/support/milbrandt_cold_cloud_collection_reference.hpp tests/support/milbrandt_frozen_collection_reference.hpp $(wildcard src/microphysics/schemes/milbrandt/*.hpp) include/microphysics/milbrandt_cold_parameters.hpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_cold_riming_repair bin/test_physics_native_milbrandt_cold: tests/support/milbrandt_double_riming_reference.hpp tests/support/milbrandt_cold_cloud_diffusion_reference.hpp tests/support/milbrandt_cold_cloud_collection_reference.hpp tests/support/milbrandt_frozen_collection_reference.hpp

bin/test_physics_milbrandt_triple_cold_freezing: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cold_freezing.cpp src/microphysics/schemes/milbrandt/triple_cold_freezing.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_warm_phase_adapter: $(TEST_INFRA) tests/physics/test_milbrandt_triple_warm_phase_adapter.cpp src/microphysics/schemes/milbrandt/warm_rain.hpp src/microphysics/schemes/milbrandt/triple_warm_primary.hpp src/microphysics/schemes/milbrandt/triple_warm_events.cpp src/microphysics/schemes/milbrandt/triple_warm_phase.cpp src/microphysics/schemes/milbrandt/triple_warm_adapter.cpp src/microphysics/schemes/milbrandt/triple_warm_primary.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_init_milbrandt_triple_scope: $(TEST_INFRA) tests/init/config/test_milbrandt_triple_scope.cpp src/init/config/v1_loader.cpp $(CONFIG_CODEC_SRCS) $(CONFIG_MIGRATION_SRCS) src/core/runtime/yaml_helpers.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_qualification_kessler_reference_column: $(TEST_INFRA) tests/qualification/test_kessler_reference_column.cpp src/microphysics/base/thermodynamics.cpp src/microphysics/schemes/kessler/kessler.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_qualification_lin_reference_processes: $(TEST_INFRA) tests/qualification/test_lin_reference_processes.cpp src/microphysics/base/thermodynamics.cpp src/microphysics/schemes/lin/lin.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_qualification_storm_history_replay: $(TEST_INFRA) tests/qualification/test_storm_history_replay.cpp src/diagnostics/storm_tracker.cpp src/diagnostics/storm_rotation.cpp src/core/infra/domain_motion.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

.PHONY: qualify-storm-history-replay-available
qualify-storm-history-replay-available: bin/test_qualification_storm_history_replay
	python3 scripts/qualification/domain_motion/qualify_storm_history_replay.py --binary bin/test_qualification_storm_history_replay $(STORM_HISTORY_ARGS)

bin/test_physics_microphysics_runtime_state bin/test_physics_microphysics_runtime_state_optimized: $(TEST_INFRA) tests/physics/test_microphysics_runtime_state.cpp src/core/orchestration/physics/microphysics_step.cpp src/microphysics/grid_velocity.cpp src/terrain/base/conservative_transport.cpp src/core/runtime/simulation_state.cpp src/core/infra/simd_utils.cpp src/radiation/energy_ledger.cpp | bin
	$(CXX_COMPILE) $(if $(filter %_optimized,$@),$(filter-out -O0,$(TEST_CXXFLAGS)),$(TEST_CXXFLAGS)) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_radiation: $(TEST_INFRA) tests/physics/test_radiation.cpp $(RADIATION_SCHEME_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_radiation_runtime_state: tests/test_main.cpp tests/physics/test_radiation_runtime_state.cpp src/radiation/runtime.cpp src/radiation/native_batch_runtime.cpp src/radiation/energy_ledger.cpp src/radiation/column_evaluation.cpp src/core/runtime/simulation_state.cpp $(RADIATION_SCHEME_SRCS) src/core/infra/coordinate_system.cpp $(COMPUTE_RUNTIME_SRC) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_boundary_layer: $(TEST_INFRA) tests/physics/test_boundary_layer.cpp $(BOUNDARY_LAYER_SCHEME_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_boundary_layer_runtime_state: tests/test_main.cpp tests/physics/test_boundary_layer_runtime_state.cpp src/boundary_layer/runtime.cpp src/core/runtime/native_held_sources.cpp src/boundary_layer/native_batch_runtime.cpp src/boundary_layer/column_evaluation.cpp src/core/runtime/simulation_state.cpp src/core/runtime/boundary_layer_checkpoint.cpp src/core/infra/sha256.cpp $(BOUNDARY_LAYER_SCHEME_SRCS) src/core/infra/coordinate_system.cpp src/terrain/base/conservative_transport.cpp $(COMPUTE_RUNTIME_SRC) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_chaos_runtime_state: tests/test_main.cpp tests/physics/test_chaos_runtime_state.cpp $(CHAOS_SRCS) src/core/runtime/simulation_state.cpp src/core/orchestration/dynamics/thermodynamic_closure.cpp src/core/infra/coordinate_system.cpp src/core/infra/domain_motion.cpp src/core/infra/simd_utils.cpp $(COMPUTE_RUNTIME_SRC) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_reflectivity: $(TEST_INFRA) tests/physics/test_reflectivity.cpp src/radar/schemes/reflectivity/reflectivity.cpp src/radar/base/radar_base.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_radar_velocity: $(TEST_INFRA) tests/physics/test_radar_velocity.cpp src/radar/schemes/velocity/velocity.cpp src/radar/base/radar_base.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_radar_zdr: $(TEST_INFRA) tests/physics/test_radar_zdr.cpp src/radar/schemes/zdr/zdr.cpp src/radar/base/radar_base.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_radar_runtime_state: $(TEST_INFRA) tests/physics/test_radar_runtime_state.cpp $(RADAR_SYSTEM_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_turbulence_topology: $(TEST_INFRA) tests/physics/test_turbulence_topology.cpp src/turbulence/base/eddy_viscosity.cpp src/turbulence/base/collocated_eddy_viscosity.cpp $(TERRAIN_TRANSPORT_SRC) src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_turbulence_sgs: $(TEST_INFRA) tests/physics/test_turbulence_sgs.cpp src/turbulence/schemes/smagorinsky/dynamic_smagorinsky.cpp src/turbulence/base/eddy_viscosity.cpp src/turbulence/base/collocated_eddy_viscosity.cpp $(TERRAIN_TRANSPORT_SRC) src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_tke_scheme: $(TEST_INFRA) tests/physics/test_tke_scheme.cpp src/turbulence/base/native_collocated_eddy_viscosity.cpp src/turbulence/schemes/tke/tke.cpp src/turbulence/base/eddy_viscosity.cpp src/turbulence/base/collocated_eddy_viscosity.cpp $(TERRAIN_TRANSPORT_SRC) src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_turbulence_runtime_state: tests/test_main.cpp tests/physics/test_turbulence_runtime_state.cpp src/turbulence/runtime.cpp src/core/runtime/simulation_state.cpp src/core/infra/coordinate_system.cpp src/core/infra/simd_utils.cpp $(COMPUTE_RUNTIME_SRC) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_sgs_apply_runtime_state bin/test_physics_sgs_apply_runtime_state_optimized: tests/test_main.cpp tests/physics/test_sgs_apply_runtime_state.cpp src/core/orchestration/dynamics/sgs_apply.cpp src/core/runtime/simulation_state.cpp src/turbulence/base/eddy_viscosity.cpp src/turbulence/base/collocated_eddy_viscosity.cpp $(TERRAIN_TRANSPORT_SRC) src/core/infra/coordinate_system.cpp $(COMPUTE_RUNTIME_SRC) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(if $(filter %_optimized,$@),$(filter-out -O0,$(TEST_CXXFLAGS)) -DNDEBUG,$(TEST_CXXFLAGS)) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_terrain: $(TEST_INFRA) tests/physics/test_terrain.cpp $(TERRAIN_SCHEME_SRCS) $(TERRAIN_TRANSPORT_SRC) src/numerics/advection/advection_cartesian.cpp src/numerics/advection/advection_perf.cpp src/core/infra/coordinate_system.cpp src/core/infra/domain_motion.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

# Data tests
bin/test_data_soundings: $(TEST_INFRA) tests/data/test_soundings.cpp $(SOUNDING_BACKEND_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

# Init tests (sounding sources, hodograph sources, trigger sources)
bin/test_init_parametric_cape_sounding: $(TEST_INFRA) tests/init/test_parametric_cape_sounding.cpp src/init/sounding/sources/parametric_cape.cpp src/init/sounding/diagnostics.cpp src/init/sounding/sounding_source.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_init_parametric_targets_sounding: $(TEST_INFRA) tests/init/test_parametric_targets_sounding.cpp src/init/sounding/sources/parametric_targets.cpp src/init/sounding/sources/parametric_cape.cpp src/init/sounding/diagnostics.cpp src/init/sounding/sounding_source.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_init_sounding_diagnostics: $(TEST_INFRA) tests/init/test_sounding_diagnostics.cpp src/init/sounding/diagnostics.cpp src/init/sounding/sounding_source.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_init_file_sounding: $(TEST_INFRA) tests/init/test_file_sounding.cpp src/init/sounding/sources/file_sounding.cpp src/init/sounding/diagnostics.cpp src/init/sounding/sounding_source.cpp $(SOUNDING_READER_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_init_hodograph_sources: $(TEST_INFRA) tests/init/test_hodograph_sources.cpp $(INIT_HODOGRAPH_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_init_vortex_seed_trigger: $(TEST_INFRA) tests/init/test_vortex_seed_trigger.cpp src/init/trigger/sources/vortex_seed.cpp src/core/runtime/simulation_state.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_init_b3_scheme_trigger_decoupling: $(TEST_INFRA) tests/init/test_b3_scheme_trigger_decoupling.cpp $(DYNAMICS_KERNEL_SRCS) $(DYNAMICS_FACTORY_TEST_SUPPORT_SRCS) $(INIT_TRIGGER_SRCS) $(CONFIG_CODEC_SRCS) src/core/runtime/simulation_state.cpp src/core/infra/coordinate_system.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_init_scheme_profile_validator: $(TEST_INFRA) tests/init/test_scheme_profile_validator.cpp src/init/scheme_profile.cpp $(INIT_SOUNDING_SRCS) $(INIT_HODOGRAPH_SRCS) $(INIT_TRIGGER_SRCS) $(SOUNDING_READER_SRCS) src/core/runtime/simulation_state.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_init_config_v1: $(TEST_INFRA) tests/init/config/test_v1_loader.cpp tests/init/config/test_configuration_compiler.cpp src/init/config/v1_loader.cpp $(CONFIG_CODEC_SRCS) $(CONFIG_MIGRATION_SRCS) src/core/runtime/yaml_helpers.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_core_generated_metal_unserialized_controls: $(TEST_INFRA) tests/core/test_generated_metal_unserialized_controls.cpp src/init/config/v1_loader.cpp $(CONFIG_CODEC_SRCS) $(CONFIG_MIGRATION_SRCS) src/core/runtime/yaml_helpers.cpp src/core/infra/coordinate_system.cpp $(wildcard tests/fixtures/reference/catalog/generated_metal_unserialized_controls/*.yaml) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_vulkan_memory_window_telemetry: tests/test_main.cpp tests/vulkan/test_memory_window_telemetry.cpp vulkan/include/compute/vulkan_compute_telemetry.hpp $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_vulkan_memory_plan: tests/test_main.cpp tests/vulkan/test_memory_plan.cpp vulkan/include/compute/vulkan_memory_plan.hpp $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_vulkan_acoustic_input_reuse: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/vulkan/test_acoustic_input_reuse.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

RETAINED_FLUX_NO_CONTRACT_OBJ := $(BUILDDIR)/arithmetic/advection_no_contract.o
-include $(RETAINED_FLUX_NO_CONTRACT_OBJ:.o=.d)
$(RETAINED_FLUX_NO_CONTRACT_OBJ): src/numerics/advection/advection_cartesian.cpp $(MAIN_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(CXXFLAGS) $(CPPFLAGS) $(MAIN_ASSERT_FLAGS) -ffp-contract=off -MMD -MP -c $< -o $@

bin/test_native_retained_flux: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/numerics/test_native_retained_flux.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_native_retained_flux_no_contract: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/numerics/test_native_retained_flux.cpp $(filter-out $(BUILDDIR)/src/numerics/advection/advection_cartesian.o,$(NESTED_PRODUCTION_TEST_OBJS)) $(RETAINED_FLUX_NO_CONTRACT_OBJ) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) -ffp-contract=off $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_init_radar_zfp_profile: $(TEST_INFRA) tests/init/config/test_radar_zfp_profile.cpp src/init/config/v1_loader.cpp $(CONFIG_CODEC_SRCS) $(CONFIG_MIGRATION_SRCS) src/core/runtime/yaml_helpers.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_init_vertical_only_profile: $(TEST_INFRA) tests/init/config/test_vertical_only_profile.cpp src/init/config/v1_loader.cpp $(CONFIG_CODEC_SRCS) $(CONFIG_MIGRATION_SRCS) src/core/runtime/yaml_helpers.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_init_config_migrator: $(TEST_INFRA) tests/init/config/test_v0_to_v1.cpp $(CONFIG_MIGRATION_SRCS) src/init/config/v1_loader.cpp $(CONFIG_CODEC_SRCS) src/core/runtime/yaml_helpers.cpp src/core/infra/coordinate_system.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

# Backend-neutral compute tests
bin/test_compute_backend_types: $(TEST_INFRA) tests/compute/test_backend_types.cpp src/compute/backend_types.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_compute_backend_contracts: $(TEST_INFRA) tests/compute/test_backend_contracts.cpp src/compute/kernel_manifest.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_compute_cpu_backend: $(TEST_INFRA) tests/compute/test_cpu_backend.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_compute_memory_admission: $(TEST_INFRA) tests/compute/test_memory_admission.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_compute_residency: $(TEST_INFRA) tests/compute/test_residency.cpp $(COMPUTE_RESIDENCY_SRC) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_parity_check: $(TEST_INFRA) tests/compute/test_parity.cpp tests/compute/support/parity_registry.cpp tests/compute/support/parity_registry_microphysics_anchors.cpp src/core/infra/coordinate_system.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

KERNEL_OWNERSHIP_METAL_HEADERS := metal/src/compute/internal/metal_source_catalog.hpp \
    $(wildcard metal/include/compute/metal/shaders/*.hpp metal/include/compute/metal/shaders/*/*.hpp)

bin/nimbus_kernel_ownership: tests/compute/kernel_ownership_main.cpp tests/compute/support/kernel_ownership_catalog.cpp tests/compute/support/parity_registry.cpp tests/compute/support/parity_registry_microphysics_anchors.cpp src/compute/kernel_manifest.cpp $(KERNEL_OWNERSHIP_METAL_HEADERS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) -I metal/src $(filter %.cpp,$^) $(LDLIBS) -o $@

# Vulkan implementation tests
bin/test_vulkan_startup_plan: tests/test_main.cpp tests/vulkan/test_kernel_startup_plan.cpp tests/vulkan/test_kernel_startup_profile_loader.cpp src/compute/kernel_manifest.cpp src/init/config/v1_loader.cpp $(CONFIG_CODEC_SRCS) src/core/runtime/yaml_helpers.cpp src/core/infra/coordinate_system.cpp $(TEST_HDRS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_vulkan_backend: $(TEST_INFRA) tests/vulkan/test_compute_backend.cpp tests/vulkan/test_memory_plan.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_vulkan_required_dispatch: $(TEST_INFRA) tests/vulkan/test_gpu_required_dispatch.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_vulkan_gpu_parity: $(TEST_INFRA) tests/vulkan/test_gpu_parity.cpp tests/vulkan/test_mapped_mass_continuity.cpp src/terrain/base/topography.cpp src/terrain/base/native_mapped_transaction.cpp src/core/runtime/simulation_state.cpp src/core/orchestration/dynamics/eos_apply.cpp src/diagnostics/budgets/energy_budget.cpp src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp src/dynamics/cartesian_cgrid_momentum_flux.cpp src/dynamics/device_slow_energy_attribution.cpp $(BACKEND_COMMON_SRCS) src/core/infra/domain_motion.cpp src/core/infra/domain_recycling.cpp src/core/infra/sha256.cpp src/chaos/base/earth_correlated_noise.cpp $(MICROPHYSICS_SCHEME_SRCS) $(DIFFUSION_SCHEME_SRCS) src/turbulence/base/eddy_viscosity.cpp src/turbulence/base/collocated_eddy_viscosity.cpp src/boundary_layer/base/surface_fluxes.cpp src/boundary_layer/schemes/mynn/mynn_stability.cpp src/boundary_layer/schemes/mynn/mynn.cpp src/boundary_layer/schemes/mynn/mynn_batch.cpp src/radiation/schemes/rrtmg/rrtmg_longwave_transfer.cpp src/radiation/schemes/rrtmg/rrtmg_shortwave_transfer.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_vulkan_gpu_parity: src/dynamics/native_cgrid_boundary.cpp tests/compute/support/sgs_scalar_boundary.hpp

bin/test_vulkan_mapped_mass_continuity: $(TEST_INFRA) tests/vulkan/test_mapped_mass_continuity.cpp src/terrain/base/topography.cpp src/terrain/base/native_mapped_transaction.cpp src/dynamics/native_cgrid_boundary.cpp src/core/infra/domain_motion.cpp src/core/runtime/simulation_state.cpp src/core/orchestration/dynamics/eos_apply.cpp src/diagnostics/budgets/energy_budget.cpp src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp src/dynamics/cartesian_cgrid_momentum_flux.cpp src/dynamics/device_slow_energy_attribution.cpp $(BACKEND_COMMON_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

# S7.1 scientific colormap LUTs (standalone module — no Vulkan/GPU deps).
bin/test_vulkan_colormap_lut: $(TEST_INFRA) tests/vulkan/test_colormap_lut.cpp vulkan/src/rendering/colormap_lut.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_vulkan_capture_format: $(TEST_INFRA) tests/vulkan/test_capture_format.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_vulkan_grid_geometry: $(TEST_INFRA) tests/vulkan/test_grid_geometry.cpp vulkan/src/data/grid_geometry.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

# S7.3 CPU marching-cubes reference (standalone module — no Vulkan/GPU deps).
MILBRANDT_COLD_RIMING_SRCS := src/microphysics/schemes/milbrandt/triple_cold_riming.cpp src/microphysics/schemes/milbrandt/triple_ice_graupel_conversion.cpp src/microphysics/schemes/milbrandt/triple_rime_splintering.cpp src/microphysics/schemes/milbrandt/triple_ice_cloud_collection.cpp src/microphysics/schemes/milbrandt/triple_cold_cloud_collection.cpp src/microphysics/schemes/milbrandt/triple_cold_cloud_diffusion.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp
MILBRANDT_COLD_RIMING_SRCS += src/microphysics/schemes/milbrandt/triple_snow_graupel_conversion.cpp
MILBRANDT_COLD_RIMING_SRCS += src/microphysics/schemes/milbrandt/triple_homogeneous_cloud_freezing.cpp
MILBRANDT_COLD_RIMING_SRCS += src/microphysics/schemes/milbrandt/triple_graupel_hail_conversion.cpp
MILBRANDT_COLD_RIMING_SRCS += src/microphysics/schemes/milbrandt/triple_frozen_collection.cpp

bin/test_physics_milbrandt_triple_ice_cloud_collection: $(TEST_INFRA) tests/physics/test_milbrandt_triple_ice_cloud_collection.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) src/microphysics/schemes/milbrandt/triple_ice_cloud_collection.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_rime_splintering: $(TEST_INFRA) tests/physics/test_milbrandt_triple_rime_splintering.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) src/microphysics/schemes/milbrandt/triple_rime_splintering.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_riming: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cold_riming.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(MILBRANDT_COLD_RIMING_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_ice_graupel: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cold_ice_graupel.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(MILBRANDT_COLD_RIMING_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_graupel_conversions: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cold_graupel_conversions.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(MILBRANDT_COLD_RIMING_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_snow_graupel_conversion: $(TEST_INFRA) tests/physics/test_milbrandt_triple_snow_graupel_conversion.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(MILBRANDT_COLD_RIMING_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_homogeneous_cloud_freezing: $(TEST_INFRA) tests/physics/test_milbrandt_triple_homogeneous_cloud_freezing.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) src/microphysics/schemes/milbrandt/triple_homogeneous_cloud_freezing.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_homogeneous: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cold_homogeneous.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(MILBRANDT_COLD_RIMING_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_graupel_hail_conversion: $(TEST_INFRA) tests/physics/test_milbrandt_triple_graupel_hail_conversion.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(MILBRANDT_COLD_RIMING_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_sll: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cold_sll.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(MILBRANDT_COLD_RIMING_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_collected_sll: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cold_collected_sll.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(MILBRANDT_COLD_RIMING_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_rain_sll: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cold_rain_sll.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(MILBRANDT_COLD_RIMING_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_rain_collision: $(TEST_INFRA) tests/physics/test_milbrandt_triple_rain_collision.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(MILBRANDT_COLD_RIMING_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_hail_wet_growth: $(TEST_INFRA) tests/physics/test_milbrandt_triple_hail_wet_growth.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(MILBRANDT_COLD_RIMING_SRCS) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_source_eos_energy: $(TEST_INFRA) tests/physics/test_milbrandt_triple_source_eos_energy.cpp \
	src/microphysics/schemes/milbrandt/triple_source_eos_energy.hpp \
	src/core/orchestration/dynamics/eos_apply.cpp src/core/runtime/simulation_state.cpp \
	src/core/infra/coordinate_system.cpp $(TERRAIN_TRANSPORT_SRC) | bin
	$(CXX_COMPILE) $(CXXFLAGS) -DNDEBUG -I tests $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_hail_surface_condensation: $(TEST_INFRA) tests/physics/test_milbrandt_triple_hail_surface_condensation.cpp \
	src/microphysics/schemes/milbrandt/triple_hail_surface_condensation.hpp \
	src/microphysics/schemes/milbrandt/triple_source_eos_energy.hpp \
	tests/support/milbrandt_frozen_collection_reference.hpp \
	src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp \
	src/core/orchestration/dynamics/eos_apply.cpp src/core/runtime/simulation_state.cpp \
	src/core/infra/coordinate_system.cpp $(TERRAIN_TRANSPORT_SRC) | bin
	$(CXX_COMPILE) $(CXXFLAGS) -DNDEBUG -I tests $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_hail_surface_liquid: $(TEST_INFRA) tests/physics/test_milbrandt_triple_hail_surface_liquid.cpp \
	src/microphysics/schemes/milbrandt/triple_hail_surface_liquid.hpp \
	$(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(wildcard tests/support/milbrandt*.hpp) \
	src/microphysics/schemes/milbrandt/triple_cold_cloud_collection.cpp \
	src/microphysics/schemes/milbrandt/triple_frozen_collection.cpp \
	src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp \
	src/core/orchestration/dynamics/eos_apply.cpp src/core/runtime/simulation_state.cpp \
	src/core/infra/coordinate_system.cpp $(TERRAIN_TRANSPORT_SRC) | bin
	$(CXX_COMPILE) $(CXXFLAGS) -DNDEBUG -I tests $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_hail_surface_frozen: $(TEST_INFRA) tests/physics/test_milbrandt_triple_hail_surface_frozen.cpp \
	tests/physics/test_milbrandt_triple_hail_surface_liquid.cpp \
	$(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(wildcard tests/support/milbrandt*.hpp) \
	src/microphysics/schemes/milbrandt/triple_cold_cloud_collection.cpp \
	src/microphysics/schemes/milbrandt/triple_frozen_collection.cpp \
	src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp \
	src/core/orchestration/dynamics/eos_apply.cpp src/core/runtime/simulation_state.cpp \
	src/core/infra/coordinate_system.cpp $(TERRAIN_TRANSPORT_SRC) | bin
	$(CXX_COMPILE) $(CXXFLAGS) -DNDEBUG -I tests $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_hail_shared_vapor: $(TEST_INFRA) tests/physics/test_milbrandt_triple_hail_shared_vapor.cpp \
	tests/physics/test_milbrandt_triple_hail_surface_liquid.cpp tests/physics/test_milbrandt_triple_hail_surface_frozen.cpp \
	$(wildcard src/microphysics/schemes/milbrandt/*.hpp) $(wildcard tests/support/milbrandt*.hpp) \
	src/microphysics/schemes/milbrandt/triple_cold_cloud_collection.cpp src/microphysics/schemes/milbrandt/triple_cold_cloud_diffusion.cpp \
	src/microphysics/schemes/milbrandt/triple_frozen_collection.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp \
	src/microphysics/base/thermodynamics.cpp src/core/orchestration/dynamics/eos_apply.cpp src/core/runtime/simulation_state.cpp \
	src/core/infra/coordinate_system.cpp $(TERRAIN_TRANSPORT_SRC) | bin
	$(CXX_COMPILE) $(CXXFLAGS) -DNDEBUG -I tests $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_physics_milbrandt_triple_cold_spectrum_domain: $(TEST_INFRA) tests/physics/test_milbrandt_triple_cold_spectrum_domain.cpp $(wildcard tests/support/milbrandt*.hpp) $(wildcard src/microphysics/schemes/milbrandt/*.hpp) src/microphysics/schemes/milbrandt/triple_cold_cloud_collection.cpp src/microphysics/schemes/milbrandt/triple_cold_cloud_diffusion.cpp src/microphysics/schemes/milbrandt/triple_rime_splintering.cpp src/microphysics/schemes/milbrandt/triple_sedimentation.cpp src/microphysics/base/thermodynamics.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_marching_cubes_cpu: $(TEST_INFRA) tests/vulkan/test_marching_cubes_cpu.cpp vulkan/src/rendering/marching_cubes_cpu.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

# Integration tests
bin/test_integration: $(TEST_INFRA) tests/integration/test_config_presets.cpp tests/integration/test_performance.cpp src/core/output/output_config.cpp src/core/output/npy_writer.cpp src/core/infra/hardware_info.cpp src/core/infra/simd_utils.cpp src/diagnostics/contracts/field_contract.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_shm_e2e: $(TEST_INFRA) tests/integration/test_shm_e2e.cpp src/core/output/shm_writer.cpp src/core/output/npy_writer.cpp vulkan/src/data/shm_dataset.cpp vulkan/src/data/export_dataset.cpp vulkan/src/data/grid_geometry.cpp src/core/io/npy_reader.cpp | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

CATCH2_BINS := bin/test_core_checkpoint_wire bin/test_core_field bin/test_core_parallel_copy bin/test_core_output bin/test_core_durable_file bin/test_core_hardware bin/test_core_runtime_cadence bin/test_core_scheme_catalog bin/test_core_simulation_checkpoint bin/test_core_simd_aliasing bin/test_core_eos_pressure bin/test_core_npy \
               bin/test_core_output_writer bin/test_core_netcdf_history_writer bin/test_core_shm bin/test_core_coordinate_system bin/test_core_stagger_interpolation \
               bin/test_core_domain_motion bin/test_core_domain_recycling bin/test_core_domain_recycling_swath bin/test_core_nested_grid bin/test_core_nested_terrain_geometry \
               bin/test_diagnostics_contract bin/test_diagnostics_validation bin/test_diagnostics_conservation_budget bin/test_diagnostics_momentum_budget bin/test_diagnostics_energy_budget bin/test_diagnostics_moisture_budget bin/test_diagnostics_cartesian_derivatives bin/test_diagnostics_column_water_paths bin/test_diagnostics_potential_vorticity bin/test_diagnostics_storm_tracker \
               bin/test_dynamics_cartesian bin/test_dynamics_cartesian_bcs bin/test_dynamics_cartesian_ic bin/test_dynamics_collocated_mapped_timestep bin/test_dynamics_cartesian_cgrid_slow_tendencies_gpu_parity \
               bin/test_numerics_advection bin/test_numerics_advection_cartesian bin/test_numerics_diffusion bin/test_numerics_timestepping bin/test_numerics_staggered_derivatives \
               bin/test_physics_microphysics bin/test_qualification_kessler_reference_column bin/test_qualification_lin_reference_processes bin/test_physics_microphysics_runtime_state bin/test_physics_radiation bin/test_physics_radiation_runtime_state bin/test_physics_boundary_layer_runtime_state bin/test_physics_chaos_runtime_state bin/test_physics_turbulence_runtime_state bin/test_physics_tke_scheme bin/test_physics_sgs_apply_runtime_state bin/test_physics_radar_velocity bin/test_physics_radar_zdr bin/test_physics_radar_runtime_state bin/test_physics_terrain \
               bin/test_data_soundings bin/test_compute_backend_types bin/test_compute_backend_contracts bin/test_compute_cpu_backend bin/test_compute_memory_admission bin/test_compute_residency bin/test_parity_check \
               bin/test_vulkan_backend bin/test_vulkan_gpu_parity bin/test_vulkan_colormap_lut bin/test_vulkan_capture_format bin/test_vulkan_grid_geometry bin/test_marching_cubes_cpu \
               bin/test_integration bin/test_shm_e2e

CATCH2_BINS += bin/test_physics_milbrandt_triple_warm_spectrum bin/test_physics_milbrandt_triple_dynamic_warm bin/test_physics_milbrandt_triple_warm_subcycle
CATCH2_BINS += bin/test_physics_milbrandt_triple_frozen_collection
CATCH2_BINS += bin/test_physics_milbrandt_triple_cold_cloud_collection
CATCH2_BINS += bin/test_physics_milbrandt_triple_cold_cloud_diffusion
CATCH2_BINS += bin/test_physics_milbrandt_cold_riming_repair
CATCH2_BINS += bin/test_physics_milbrandt_triple_ice_cloud_collection bin/test_physics_milbrandt_triple_rime_splintering bin/test_physics_milbrandt_triple_cold_riming
CATCH2_BINS += bin/test_physics_milbrandt_triple_cold_ice_graupel
CATCH2_BINS += bin/test_physics_milbrandt_triple_cold_spectrum_domain
CATCH2_BINS += bin/test_physics_milbrandt_triple_cold_graupel_conversions bin/test_physics_milbrandt_triple_snow_graupel_conversion
CATCH2_BINS += bin/test_physics_milbrandt_triple_homogeneous_cloud_freezing bin/test_physics_milbrandt_triple_cold_homogeneous bin/test_physics_milbrandt_triple_graupel_hail_conversion
CATCH2_BINS += bin/test_physics_milbrandt_triple_cold_sll
CATCH2_BINS += bin/test_physics_milbrandt_triple_cold_collected_sll bin/test_physics_milbrandt_triple_cold_rain_sll bin/test_physics_milbrandt_triple_rain_collision bin/test_physics_milbrandt_triple_hail_wet_growth
CATCH2_BINS += bin/test_physics_milbrandt_triple_source_eos_energy
CATCH2_BINS += bin/test_physics_milbrandt_triple_hail_surface_condensation
CATCH2_BINS += bin/test_physics_milbrandt_triple_hail_surface_liquid
CATCH2_BINS += bin/test_physics_milbrandt_triple_hail_surface_frozen
CATCH2_BINS += bin/test_physics_milbrandt_triple_hail_shared_vapor
CATCH2_BINS += bin/test_core_generated_metal_unserialized_controls bin/test_vulkan_memory_window_telemetry

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------
.PHONY: all run clean clean-vulkan vulkan vulkan-compute-shaders run-vulkan validate-fields \
        test test-all test-script-suite test-build-manifest test-domain-motion-inventory test-module-ownership test-public-factory-headers test-public-module-headers test-scheme-descriptors test-external-scheme test-compute test-core test-diagnostics test-dynamics test-numerics test-physics test-data test-init \
		test-vulkan test-vulkan-required test-integration test-studio-native-launch test-benchmark-runner test-qualification-tools test-release-provenance test-shm-e2e test-parity test-catalog-authority test-catalog-reachable-matrix test-catalog-backend-ownership qualify-catalog-backend-ownership test-catalog-backend-process-ownership qualify-catalog-backend-process-ownership test-catalog-backend-kernel-ownership qualify-catalog-backend-kernel-ownership test-catalog-domain-recycling-ownership qualify-catalog-domain-recycling-ownership test-catalog-composite-runtime-ownership qualify-catalog-composite-runtime-ownership test-catalog-physical-topology-ownership qualify-catalog-physical-topology-ownership test-catalog-backend-io-ownership qualify-catalog-backend-io-ownership test-catalog-generated-backend-tuples qualify-catalog-generated-backend-tuples test-catalog-native-admission qualify-catalog-native-admission test-catalog-native-full-preflight qualify-catalog-native-full-preflight test-catalog-native-yaml-reload qualify-catalog-native-yaml-reload test-catalog-backend-execution-surface seal-catalog-backend-execution-surface qualify-catalog-backend-operation-observations seal-catalog-backend-kernel-dispatch-coverage qualify-catalog-backend-kernel-dispatch-coverage test-catalog-studio-native-parity qualify-catalog-bounded-runs qualify-catalog-restart-threading qualify-catalog-physical-backends test-cpu-only-profile test-pbl-rrtmg-ready test-cloud-top-radiative-ysu-ready test-pbl-top-ready test-lin-rrtmg-ready test-thompson-cpu-ready test-lin-rrtmg-vulkan-ready test-lin-rrtmg-metal-ready test-native-chaos-vulkan-ready test-native-chaos-metal-ready test-variable-diffusion-ready test-radar-ready test-simple-grey-absorbing-ready test-tke-rrtmg-ready test-turbulence-closure-ready test-user-filter-width-ready test-coupled-rk-ready test-two-way-nested-ready test-active-terrain-nested-engineering test-active-terrain-nested-ready test-ready-admission-matrix test-public-config-admission test-weno5-rrtmg-ready test-implicit-rrtmg-ready test-mapped-terrain-ready test-earth-fixed-terrain-startup test-native-mapped-terrain-ready test-domain-recycling-ready test-tracked-window-ready test-agrid-core-ready test-agrid-native-ready \
		test-nested-grid-recycling-ready \
		test-collocated-mapped-terrain-ready test-release-docs-ui qualify-release-final-surface \
        smoke-test smoke-test-e2e qualify-io qualify-live-shm qualify-history-index qualify-netcdf-history qualify-zfp-core-history qualify-multiresolution-output qualify-zfp-bricked-windows qualify-zfp-codec-workspace qualify-cm1-guided-smoke qualify-science-hydrostatic qualify-science-core qualify-science-agrid qualify-science-kessler-reference qualify-science-milbrandt-coupled qualify-science-terrain-coupled qualify-science-radiation qualify-science-diffusion qualify-science-diffusion-gpu qualify-gpu-residency-equivalence qualify-vulkan qualify-vulkan-strict qualify-vulkan-report-bearing qualify-vulkan-restart qualify-vulkan-scalar-owner-parity qualify-vulkan-acoustic-owner-parity qualify-vulkan-slow-path-parity qualify-vulkan-slow-owner-parity qualify-vulkan-sponge-owner-parity qualify-vulkan-mixed-physics benchmark-vulkan-report-overhead benchmark-vulkan-minimum-acoustic-report-overhead benchmark-cpu benchmark-cpu-scaling benchmark-terrain-shortwave-cpu benchmark-terrain-shortwave-cpu-scaling check-cpu-benchmark \
        benchmark-active-physics-allocations benchmark-milbrandt-allocations benchmark-smagorinsky-filter-width benchmark-sgs-vertical-stress-reuse benchmark-nested-relocation test-nested-relocation-benchmark benchmark-zfp-history-working-set benchmark-zfp-field-parallel benchmark-zfp-brick-strides benchmark-zfp-codec-workspace benchmark-vulkan-scalar-attribution benchmark-vulkan-scalar-regions benchmark-vulkan-sgs-apply benchmark-vulkan-noop-budget-scans benchmark-vulkan-endpoint-moisture benchmark-vulkan-microphysics-gate benchmark-vulkan-microphysics-validation benchmark-vulkan-microphysics-input-validation \
        verify-shaders compile-commands format format-check print-simulator-sources print-simulator-objects \
        print-field-validator-sources print-opt-in-sources print-default-artifacts

verify-shaders:
	python3 scripts/dev/shaders/test_division_workaround.py
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py

qualify-netcdf-history: bin/netcdf_history_fixture
	CARGO_TARGET_DIR=$(CATALOG_RUST_TARGET_DIR) python3 scripts/qualification/output/qualify_netcdf_history.py

qualify-zfp-core-history: bin/nimbus_sim bin/test_core_zfp_roundtrip bin/test_core_output_writer bin/test_core_radar_zfp_v4_integration
	./bin/test_core_zfp_roundtrip "[core][zfp][roundtrip][qualified][science]"
	./bin/test_core_zfp_roundtrip "[core][zfp][writer][workspace]"
	./bin/test_core_zfp_roundtrip "[core][zfp][parallel]"
	./bin/test_core_output_writer "[core][output_writer][zfp][parallel]"
	cd ui/src-tauri && cargo test zfp3d --lib
	python3 scripts/qualification/output/qualify_zfp_core_history.py

qualify-multiresolution-output: qualify-netcdf-history qualify-zfp-core-history bin/test_core_output_writer
	./bin/test_core_output_writer "[brick-hierarchy]"
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/integration/test_multiresolution_contract.py

benchmark-zfp-field-parallel: bin/nimbus_sim
	python3 scripts/benchmarks/benchmark_zfp_field_parallel.py

qualify-history-index: bin/test_core_output_writer
	./bin/test_core_output_writer "[history_index]"
	cd ui/src-tauri && cargo test indexed_ --lib

qualify-live-shm: bin/nimbus_sim
	python3 scripts/qualification/output/qualify_live_shm.py

qualify-io: qualify-history-index qualify-multiresolution-output qualify-live-shm \
		bin/test_core_durable_file bin/test_core_output_writer \
		bin/test_core_simulation_checkpoint bin/test_diagnostics_trajectories
	./bin/test_core_durable_file
	./bin/test_core_output_writer
	./bin/test_core_simulation_checkpoint
	./bin/test_diagnostics_trajectories
	cd ui/src-tauri && cargo test --lib
	cd ui && npm run typecheck
	cd ui && npm run test:comparison-series-cache
	cd ui && npm run test:field-cancellation

qualify-zfp-bricked-windows: bin/test_core_zfp_roundtrip bin/test_core_zfp_benchmark bin/test_core_output \
		bin/test_core_output_writer
	./bin/test_core_zfp_roundtrip
	./bin/test_core_zfp_benchmark
	./bin/test_core_output
	./bin/test_core_output_writer
	cd ui/src-tauri && cargo test zfp3d --lib

qualify-science-kessler-reference: bin/test_qualification_kessler_reference_column
	./bin/test_qualification_kessler_reference_column

qualify-science-agrid: bin/nimbus_sim bin/nimbus_sim_agrid_qualification
	./bin/test_qualification_kessler_reference_column
	python3 tests/science/run.py suite agrid

qualify-science-milbrandt-coupled: bin/nimbus_sim bin/nimbus_sim_internal_qualification
	python3 ./tests/science/cgrid/physics/qualify_cgrid_milbrandt_coupling.py

qualify-science-terrain-coupled: bin/nimbus_sim
	python3 ./tests/science/cgrid/terrain/qualify_cgrid_terrain_coupling.py

# Dedicated sustained gate: six production-binary A/C-grid members spanning
# timestep and radiation-cadence sensitivity for one simulated hour. Keep the
# shorter matrix in qualify-science-agrid for ordinary development feedback.
qualify-science-radiation: bin/nimbus_sim
	python3 ./tests/science/agrid/physics/qualify_agrid_radiation_coupling.py --duration 3600 --jobs 2 --threads-per-case 1

test-public-factory-headers:
	python3 scripts/run.py test public_factory_headers -- --compiler "$(CXX)"

test-public-module-headers:
	python3 scripts/run.py test public_module_headers -- --compiler "$(CXX)"

test-scheme-descriptors:
	python3 scripts/run.py test scheme_descriptors -- --compiler "$(CXX)"

test-external-scheme:
	python3 scripts/run.py test external_scheme_fixture -- --compiler "$(CXX)"

test-module-ownership:
	python3 scripts/run.py test module_ownership

test-build-manifest:
	python3 scripts/run.py test build_manifest

test-domain-motion-inventory:
	python3 scripts/run.py test domain_motion_inventory

test-script-suite:
	python3 scripts/run.py test static --compiler "$(CXX)"
	python3 scripts/dev/configs/generate_resolution_catalog.py --check
	python3 -m unittest tests/integration/test_public_config_catalog.py

test-compute: bin/test_compute_backend_types bin/test_compute_backend_contracts bin/test_compute_cpu_backend bin/test_compute_memory_admission bin/test_compute_residency bin/test_parity_check
	./bin/test_compute_backend_types && ./bin/test_compute_backend_contracts && ./bin/test_compute_cpu_backend && ./bin/test_compute_memory_admission && ./bin/test_compute_residency && ./bin/test_parity_check

print-simulator-sources:
	@printf '%s\n' $(SRCS)

print-simulator-objects:
	@printf '%s\n' $(OBJS)

print-field-validator-sources:
	@printf '%s\n' $(FIELD_VALIDATOR_SRCS)

print-opt-in-sources:
	@printf '%s\n' $(OPT_IN_SRCS) $(STANDALONE_TOOL_SRCS)

print-default-artifacts:
	@printf '%s\n' $(DEFAULT_ARTIFACTS)

# Follow the exact build capability, including auto-detected ZFP. Checking only
# `ZFP=1` let a default HAVE_ZFP production build silently omit both codec
# suites from `make test-core`.
ifneq ($(strip $(ZFP_LDLIBS)),)
  CORE_ZFP_TESTS := bin/test_core_zfp_roundtrip bin/test_core_zfp_benchmark
  CORE_ZFP_RUN := && ./bin/test_core_zfp_roundtrip && ./bin/test_core_zfp_benchmark
else
  CORE_ZFP_TESTS :=
  CORE_ZFP_RUN :=
endif

test-core: bin/test_output_writer_lifetime bin/test_core_sha256_probe bin/test_core_sha256_portable bin/test_core_checkpoint_wire bin/test_core_field bin/test_core_parallel_copy bin/test_core_output bin/test_core_durable_file bin/test_core_hardware bin/test_core_runtime_cadence bin/test_core_scheme_catalog bin/test_core_simulation_checkpoint bin/test_core_simd_aliasing bin/test_core_eos_pressure bin/test_core_npy bin/test_core_output_writer bin/test_core_netcdf_history_writer bin/test_core_shm bin/test_core_coordinate_system bin/test_core_domain_motion bin/test_core_domain_recycling bin/test_core_domain_recycling_swath bin/test_core_stagger_interpolation bin/test_core_rayleigh_damping_yaml bin/test_core_nested_grid bin/test_core_nested_terrain_geometry bin/test_core_nested_level_runtime bin/test_core_nested_production_step bin/test_core_nested_boundary_forcing $(CORE_ZFP_TESTS)
	python3 scripts/tests/core/test_sha256_paths.py --auto bin/test_core_sha256_probe --portable bin/test_core_sha256_portable --output $(BUILDDIR)/sha256-paths.json
	./bin/test_output_writer_lifetime
	./bin/test_core_checkpoint_wire && ./bin/test_core_field && ./bin/test_core_parallel_copy && ./bin/test_core_output && ./bin/test_core_durable_file && ./bin/test_core_hardware && ./bin/test_core_runtime_cadence && ./bin/test_core_scheme_catalog && ./bin/test_core_simulation_checkpoint && ./bin/test_core_simd_aliasing && ./bin/test_core_eos_pressure && ./bin/test_core_npy && ./bin/test_core_output_writer && ./bin/test_core_netcdf_history_writer && ./bin/test_core_shm && ./bin/test_core_coordinate_system && ./bin/test_core_domain_motion && ./bin/test_core_domain_recycling && ./bin/test_core_domain_recycling_swath && ./bin/test_core_stagger_interpolation && ./bin/test_core_rayleigh_damping_yaml && ./bin/test_core_nested_grid && ./bin/test_core_nested_terrain_geometry && ./bin/test_core_nested_level_runtime && ./bin/test_core_nested_production_step && ./bin/test_core_nested_boundary_forcing $(CORE_ZFP_RUN)

test-diagnostics: bin/test_diagnostics_contract bin/test_diagnostics_validation bin/test_diagnostics_logging_ratchet bin/test_diagnostics_conservation_budget bin/test_diagnostics_momentum_budget bin/test_diagnostics_energy_budget bin/test_diagnostics_moisture_budget bin/test_diagnostics_trajectories bin/test_diagnostics_cartesian_derivatives bin/test_diagnostics_column_water_paths bin/test_diagnostics_potential_vorticity bin/test_diagnostics_storm_tracker
	./bin/test_diagnostics_contract && ./bin/test_diagnostics_validation && ./bin/test_diagnostics_logging_ratchet && ./bin/test_diagnostics_conservation_budget && ./bin/test_diagnostics_momentum_budget && ./bin/test_diagnostics_energy_budget && ./bin/test_diagnostics_moisture_budget && ./bin/test_diagnostics_trajectories && ./bin/test_diagnostics_cartesian_derivatives && ./bin/test_diagnostics_column_water_paths && ./bin/test_diagnostics_potential_vorticity && ./bin/test_diagnostics_storm_tracker

test-dynamics: bin/test_dynamics_cartesian bin/test_dynamics_cartesian_bcs bin/test_dynamics_cartesian_lateral_bc_yaml bin/test_dynamics_cartesian_vertical_bc_yaml bin/test_dynamics_coriolis_yaml bin/test_dynamics_factory_v1 bin/test_dynamics_cartesian_ic bin/test_dynamics_acoustic_cartesian_cgrid_gpu_parity bin/test_dynamics_cartesian_cgrid_slow_tendencies_gpu_parity
	./bin/test_dynamics_cartesian && ./bin/test_dynamics_cartesian_bcs && ./bin/test_dynamics_cartesian_lateral_bc_yaml && ./bin/test_dynamics_cartesian_vertical_bc_yaml && ./bin/test_dynamics_coriolis_yaml && ./bin/test_dynamics_factory_v1 && ./bin/test_dynamics_cartesian_ic && ./bin/test_dynamics_acoustic_cartesian_cgrid_gpu_parity && ./bin/test_dynamics_cartesian_cgrid_slow_tendencies_gpu_parity

test-numerics: bin/test_numerics_advection bin/test_numerics_tvd_monotonicity bin/test_numerics_advection_cartesian bin/test_numerics_weno5_reconstruction bin/test_numerics_advect_conservative_cartesian_cgrid_gpu_parity bin/test_numerics_diffusion bin/test_numerics_timestepping bin/test_numerics_runtime_timestep bin/test_numerics_staggered_derivatives
	./bin/test_numerics_advection && ./bin/test_numerics_tvd_monotonicity && ./bin/test_numerics_advection_cartesian && ./bin/test_numerics_weno5_reconstruction && ./bin/test_numerics_advect_conservative_cartesian_cgrid_gpu_parity && ./bin/test_numerics_diffusion && ./bin/test_numerics_timestepping && ./bin/test_numerics_runtime_timestep && ./bin/test_numerics_staggered_derivatives

test-physics: bin/test_physics_microphysics bin/test_qualification_kessler_reference_column bin/test_qualification_lin_reference_processes bin/test_physics_microphysics_runtime_state bin/test_physics_radiation bin/test_physics_radiation_runtime_state bin/test_physics_boundary_layer bin/test_physics_boundary_layer_runtime_state bin/test_physics_chaos_runtime_state bin/test_physics_turbulence_runtime_state bin/test_physics_tke_scheme bin/test_physics_sgs_apply_runtime_state bin/test_physics_reflectivity bin/test_physics_radar_velocity bin/test_physics_radar_zdr bin/test_physics_radar_runtime_state bin/test_physics_terrain bin/test_physics_turbulence_topology bin/test_physics_turbulence_sgs
	./bin/test_physics_microphysics && ./bin/test_qualification_kessler_reference_column && ./bin/test_qualification_lin_reference_processes && ./bin/test_physics_microphysics_runtime_state && ./bin/test_physics_radiation && ./bin/test_physics_radiation_runtime_state && ./bin/test_physics_boundary_layer && ./bin/test_physics_boundary_layer_runtime_state && ./bin/test_physics_chaos_runtime_state && ./bin/test_physics_turbulence_runtime_state && ./bin/test_physics_tke_scheme && ./bin/test_physics_sgs_apply_runtime_state && ./bin/test_physics_reflectivity && ./bin/test_physics_radar_velocity && ./bin/test_physics_radar_zdr && ./bin/test_physics_radar_runtime_state && ./bin/test_physics_terrain && ./bin/test_physics_turbulence_topology && ./bin/test_physics_turbulence_sgs

.PHONY: test-generated-metal-typed-controls
test-generated-metal-typed-controls: bin/test_core_generated_metal_unserialized_controls
	./bin/test_core_generated_metal_unserialized_controls

test-core: test-generated-metal-typed-controls

test-data: bin/test_data_soundings
	./bin/test_data_soundings

test-init: bin/test_init_parametric_cape_sounding bin/test_init_parametric_targets_sounding bin/test_init_file_sounding bin/test_init_hodograph_sources bin/test_init_vortex_seed_trigger bin/test_init_b3_scheme_trigger_decoupling bin/test_init_scheme_profile_validator bin/test_init_sounding_diagnostics bin/test_init_config_v1 bin/test_init_config_migrator
	./bin/test_init_parametric_cape_sounding && ./bin/test_init_parametric_targets_sounding && ./bin/test_init_file_sounding && ./bin/test_init_hodograph_sources && ./bin/test_init_vortex_seed_trigger && ./bin/test_init_b3_scheme_trigger_decoupling && ./bin/test_init_scheme_profile_validator && ./bin/test_init_sounding_diagnostics && ./bin/test_init_config_v1 && ./bin/test_init_config_migrator

ifeq ($(CPU_ONLY),1)
TEST_VULKAN_BINS := bin/test_vulkan_gpu_parity bin/test_vulkan_colormap_lut \
	bin/test_vulkan_grid_geometry bin/test_marching_cubes_cpu
TEST_VULKAN_RUN := ./bin/test_vulkan_gpu_parity && ./bin/test_vulkan_colormap_lut \
	&& ./bin/test_vulkan_grid_geometry && ./bin/test_marching_cubes_cpu
else
TEST_VULKAN_BINS := bin/test_vulkan_backend bin/test_vulkan_gpu_parity \
	bin/test_vulkan_colormap_lut bin/test_vulkan_capture_format \
	bin/test_vulkan_grid_geometry bin/test_marching_cubes_cpu
TEST_VULKAN_RUN := ./bin/test_vulkan_backend && ./bin/test_vulkan_gpu_parity \
	&& ./bin/test_vulkan_colormap_lut && ./bin/test_vulkan_capture_format \
	&& ./bin/test_vulkan_grid_geometry && ./bin/test_marching_cubes_cpu
endif

TEST_VULKAN_BINS += bin/test_vulkan_memory_window_telemetry
TEST_VULKAN_RUN += && ./bin/test_vulkan_memory_window_telemetry

test-vulkan: $(TEST_VULKAN_BINS)
	$(TEST_VULKAN_RUN)
	@if [ "$(CPU_ONLY)" = "1" ]; then \
		echo "[test-vulkan] SKIP: SDK-dependent backend/capture tests omitted from CPU-only profile"; \
	fi

# Fail-closed accelerator smoke. Do not add this to the default CPU test target:
# it is intentionally selected only by lanes that promise a Vulkan device.
test-vulkan-required: bin/test_vulkan_required_dispatch
	./bin/test_vulkan_required_dispatch

test-benchmark-runner:
	python3 -m unittest tests/integration/test_cpu_benchmark_runner.py
	python3 -m unittest tests/integration/test_cpu_benchmark_policy.py
	python3 -m unittest tests/integration/test_cpu_scaling_benchmark.py
	python3 -m unittest tests/integration/test_active_physics_benchmark.py
	python3 -m unittest tests/integration/test_terrain_shortwave_cpu_benchmark.py
	python3 -m unittest tests/integration/test_vulkan_report_overhead_benchmark.py
	python3 -m unittest tests/integration/test_zfp_history_working_set_benchmark.py
	python3 -m unittest tests/integration/test_zfp_field_parallel_benchmark.py
	python3 -m unittest tests/integration/test_nested_relocation_benchmark.py

test-qualification-tools:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/qualification/test_catalog_inventory.py
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest scripts.tests.catalog.test_retained_categorical_dag scripts.tests.catalog.test_backend_completion_inventory scripts.tests.catalog.test_flat_mixed_metal_rebind_v1 scripts.tests.catalog.test_flat_mixed_components_v1 scripts.tests.catalog.test_flat_mixed_metal_public_v1
	python3 -m unittest tests/integration/test_hydrostatic_qualification.py
	python3 -m unittest tests/integration/test_hydrostatic_pair.py
	python3 -m unittest tests/integration/test_agrid_acoustic_sweep.py
	python3 -m unittest tests/integration/test_agrid_boundary_pulse.py
	python3 -m unittest tests/integration/test_moving_nested_supercell_qualification.py
	python3 -m unittest tests/integration/test_moving_nested_reference_extractor.py
	python3 -m unittest tests/integration/test_moving_nested_reference_frontier.py
	python3 -m unittest tests/integration/test_moving_nested_reference_campaign.py
	python3 -m unittest tests/integration/test_moving_nested_vulkan_continuation.py
	python3 -m unittest tests/integration/test_moving_nested_vulkan_lifetime_chain.py
	python3 -m unittest tests/integration/test_moving_nested_vulkan_reference.py
	python3 -m unittest tests/integration/test_stationary_nested_vulkan_runtime_parser.py
	python3 -m unittest tests/integration/test_volume_raymarch_benchmark.py
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/qualification/test_domain_recycling_sidecar.py
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/qualification/test_release_manifest.py

.PHONY: qualify-catalog-reachable-evidence inspect-release-evidence \
		qualify-release-receipts qualify-release-seal \
		reports/pre_beta/gates/test-catalog-authority.json \
		reports/pre_beta/gates/test-catalog-reachable-matrix.json \
		reports/pre_beta/gates/test-catalog-studio-native-parity.json \
		reports/pre_beta/gates/qualify-catalog-bounded-runs.json \
		reports/pre_beta/gates/qualify-catalog-restart-threading.json \
		reports/pre_beta/gates/qualify-catalog-physical-backends.json \
		reports/pre_beta/gates/qualify-io.json \
		reports/pre_beta/gates/qualify-release-final-surface.json

test-release-provenance:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
	  tests/qualification/test_release_manifest.py \
	  tests/qualification/test_reachable_set_evidence.py \
	  tests/qualification/test_record_gate_result.py \
	  tests/qualification/test_inspect_release_evidence.py \
	  tests/qualification/test_catalog_gate_pinning.py \
	  tests/qualification/test_kernel_dispatch_coverage_seal.py \
	  tests/qualification/test_domain_recycling_owner_contracts.py \
	  tests/qualification/test_composite_runtime_ownership.py \
	  tests/qualification/test_physical_topology_ownership.py \
	  tests/qualification/test_backend_io_ownership.py \
	  tests/qualification/test_backend_execution_surface.py \
	  tests/qualification/test_generated_backend_tuples.py \
	  tests/qualification/test_native_admission_batch.py \
	  tests/qualification/test_native_full_preflight.py \
	  tests/qualification/test_native_yaml_reload.py \
	  tests/qualification/test_retained_structural_provenance.py

inspect-release-evidence:
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/qualification/provenance/inspect_release_evidence.py

test-release-docs-ui: test-catalog-studio-native-parity
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/architecture/test_studio_distribution.py
	cd ui && npm run typecheck
	cd ui && npm run typecheck:browser
	cd ui && npm run test:unit
	cd ui/src-tauri && cargo test --locked --lib

qualify-release-final-surface: test test-release-docs-ui
	@echo "[release-final-surface] PASS: full repository, documentation, UI, and Rust library gates"

qualify-catalog-reachable-evidence: bin/nimbus_sim bin/test_core_scheme_catalog
	mkdir -p reports/pre_beta
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/qualification/provenance/generate_reachable_set_evidence.py \
	  --root . --binary bin/nimbus_sim \
	  --cross-product-binary bin/test_core_scheme_catalog \
	  --authority tests/fixtures/reference/catalog/authority_v1.json \
	  --output reports/pre_beta/reachable_set_v1.json

test-catalog-backend-ownership: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_compute_ownership_matrix.py \
	  --binary bin/nimbus_sim

qualify-catalog-backend-ownership: bin/nimbus_sim
	mkdir -p reports/pre_beta
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_compute_ownership_matrix.py \
	  --binary bin/nimbus_sim \
	  --output reports/pre_beta/backend_ownership_v1.json

test-catalog-backend-process-ownership: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_compute_process_ownership_matrix.py \
	  --binary bin/nimbus_sim

qualify-catalog-backend-process-ownership: bin/nimbus_sim
	mkdir -p reports/pre_beta
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_compute_process_ownership_matrix.py \
	  --binary bin/nimbus_sim \
	  --output reports/pre_beta/backend_process_ownership_v1.json

test-catalog-backend-kernel-ownership: bin/nimbus_sim bin/nimbus_kernel_ownership
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_kernel_ownership_catalog.py \
	  --binary bin/nimbus_sim --tool bin/nimbus_kernel_ownership

qualify-catalog-backend-kernel-ownership: bin/nimbus_sim bin/nimbus_kernel_ownership
	mkdir -p reports/pre_beta
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_kernel_ownership_catalog.py \
	  --binary bin/nimbus_sim --tool bin/nimbus_kernel_ownership \
	  --output reports/pre_beta/backend_kernel_ownership_v1.json

test-catalog-domain-recycling-ownership: bin/nimbus_sim bin/test_core_domain_recycling
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/tests/catalog/test_domain_recycling_owner_contracts.py \
	  --binary bin/nimbus_sim --test-binary bin/test_core_domain_recycling

qualify-catalog-domain-recycling-ownership: bin/nimbus_sim bin/test_core_domain_recycling
	mkdir -p reports/pre_beta
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/tests/catalog/test_domain_recycling_owner_contracts.py \
	  --binary bin/nimbus_sim --test-binary bin/test_core_domain_recycling \
	  --output reports/pre_beta/domain_recycling_owner_evidence_v1.json

qualify-catalog-composite-runtime-ownership: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/qualification/provenance/qualify_composite_runtime_ownership.py \
	  --binary bin/nimbus_sim

test-catalog-composite-runtime-ownership: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
	  tests/qualification/test_composite_runtime_ownership.py
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/qualification/provenance/qualify_composite_runtime_ownership.py \
	  --binary bin/nimbus_sim --verify

qualify-catalog-physical-topology-ownership: bin/nimbus_sim bin/nimbus_sim_physical_qualification bin/test_core_nested_production_step
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/qualification/provenance/qualify_physical_topology_ownership.py \
	  --binary bin/nimbus_sim --snapshot bin/nimbus_sim_physical_qualification

test-catalog-physical-topology-ownership: bin/nimbus_sim bin/nimbus_sim_physical_qualification
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
	  tests/qualification/test_physical_topology_ownership.py
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/qualification/provenance/qualify_physical_topology_ownership.py \
	  --binary bin/nimbus_sim --snapshot bin/nimbus_sim_physical_qualification --verify

qualify-catalog-backend-io-ownership: bin/nimbus_sim bin/test_core_output_writer \
		bin/test_core_netcdf_history_writer bin/test_core_zfp_roundtrip bin/test_core_shm \
		bin/test_core_durable_file bin/test_core_simulation_checkpoint bin/test_diagnostics_trajectories
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/qualification/provenance/qualify_backend_io_ownership.py \
	  --binary bin/nimbus_sim

test-catalog-backend-io-ownership: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
	  tests/qualification/test_backend_io_ownership.py
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/qualification/provenance/qualify_backend_io_ownership.py \
	  --binary bin/nimbus_sim --verify

seal-catalog-backend-execution-surface: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/qualification/provenance/seal_backend_execution_surface.py \
	  --binary bin/nimbus_sim

test-catalog-backend-execution-surface: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
	  tests/qualification/test_backend_execution_surface.py
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/qualification/provenance/seal_backend_execution_surface.py \
	  --binary bin/nimbus_sim --verify

qualify-catalog-generated-backend-tuples: bin/nimbus_sim bin/nimbus_generated_backend_tuples
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/tests/catalog/test_generated_backend_tuples.py \
	  --binary bin/nimbus_sim --tool bin/nimbus_generated_backend_tuples \
	  --output reports/pre_beta/generated_backend_tuple_evidence_v1.json

test-catalog-generated-backend-tuples: bin/nimbus_sim bin/nimbus_generated_backend_tuples
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
	  tests/qualification/test_generated_backend_tuples.py
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/tests/catalog/test_generated_backend_tuples.py \
	  --binary bin/nimbus_sim --tool bin/nimbus_generated_backend_tuples \
	  --output reports/pre_beta/generated_backend_tuple_evidence_v1.json --verify

qualify-catalog-native-admission: bin/nimbus_sim bin/nimbus_generated_backend_tuples bin/nimbus_native_admission_batch
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/tests/catalog/test_native_admission_batch.py \
	  --binary bin/nimbus_sim --generated-tool bin/nimbus_generated_backend_tuples \
	  --native-tool bin/nimbus_native_admission_batch \
	  --output reports/pre_beta/native_admission_evidence_v1.json

test-catalog-native-admission: bin/nimbus_sim bin/nimbus_generated_backend_tuples bin/nimbus_native_admission_batch
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
	  tests/qualification/test_native_admission_batch.py
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/tests/catalog/test_native_admission_batch.py \
	  --binary bin/nimbus_sim --generated-tool bin/nimbus_generated_backend_tuples \
	  --native-tool bin/nimbus_native_admission_batch \
	  --output reports/pre_beta/native_admission_evidence_v1.json --verify

qualify-catalog-native-full-preflight: bin/nimbus_sim bin/nimbus_generated_backend_tuples bin/nimbus_native_admission_batch
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/tests/catalog/test_native_full_preflight.py \
	  --binary bin/nimbus_sim --generated-tool bin/nimbus_generated_backend_tuples \
	  --native-tool bin/nimbus_native_admission_batch \
	  --output reports/pre_beta/native_full_preflight_evidence_v1.json

test-catalog-native-full-preflight: bin/nimbus_sim bin/nimbus_generated_backend_tuples bin/nimbus_native_admission_batch
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
	  tests/qualification/test_native_full_preflight.py
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/tests/catalog/test_native_full_preflight.py \
	  --binary bin/nimbus_sim --generated-tool bin/nimbus_generated_backend_tuples \
	  --native-tool bin/nimbus_native_admission_batch \
	  --output reports/pre_beta/native_full_preflight_evidence_v1.json --verify

qualify-catalog-native-yaml-reload: bin/nimbus_sim bin/nimbus_generated_backend_tuples bin/nimbus_native_admission_batch
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/tests/catalog/qualify_native_yaml_reload.py \
	  --binary bin/nimbus_sim --generated-tool bin/nimbus_generated_backend_tuples \
	  --native-tool bin/nimbus_native_admission_batch \
	  --output reports/pre_beta/native_yaml_reload_v1.json

test-catalog-native-yaml-reload: bin/nimbus_sim bin/nimbus_generated_backend_tuples bin/nimbus_native_admission_batch
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
	  tests/qualification/test_native_yaml_reload.py
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/tests/catalog/qualify_native_yaml_reload.py \
	  --binary bin/nimbus_sim --generated-tool bin/nimbus_generated_backend_tuples \
	  --native-tool bin/nimbus_native_admission_batch \
	  --output reports/pre_beta/native_yaml_reload_v1.json --verify

qualify-catalog-backend-operation-observations: bin/nimbus_sim bin/nimbus_kernel_ownership
	mkdir -p reports/pre_beta
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/qualify_catalog_bounded_runs.py \
	  --binary $(PHYSICAL_QUALIFICATION_BINARY) --case-kind accelerator --operation-only \
	  --kernel-ownership-tool bin/nimbus_kernel_ownership \
	  --operation-requirements tests/fixtures/reference/catalog/backend_tuple_operation_requirements_v1.json \
	  --operation-output reports/pre_beta/backend_tuple_operation_observation_v1.json

seal-catalog-backend-kernel-dispatch-coverage: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/qualification/accelerators/seal_kernel_dispatch_coverage.py \
	  --binary bin/nimbus_sim \
	  --output reports/pre_beta/backend_kernel_dispatch_coverage_v1.json

qualify-catalog-backend-kernel-dispatch-coverage:
	$(MAKE) qualify-vulkan-evidence
	$(MAKE) -C metal qualify
	$(MAKE) seal-catalog-backend-kernel-dispatch-coverage

RELEASE_SOURCE_ROOTS := \
	include src mk Makefile .gitignore README.md LICENSE \
	scripts tests/science tests/test_main.cpp tests/test_harness.cpp tests/catch2 \
	tests/compute tests/core tests/data tests/diagnostics tests/dynamics \
	tests/fixtures tests/init tests/integration tests/metal tests/numerics \
	tests/physics tests/qualification tests/vulkan tests/workflows configs \
	docs .github AGENTS.md CONTRIBUTING.md \
	metal/include metal/src metal/Makefile vulkan/include vulkan/src \
	vulkan/shaders vulkan/Makefile ui/src ui/scripts \
	ui/index.html ui/eslint.config.js ui/README.md \
	ui/package.json ui/package-lock.json ui/tsconfig.json ui/tsconfig.app.json \
	ui/tsconfig.node.json ui/tsconfig.integration.json ui/vite.config.ts \
	ui/vite.browser.config.ts ui/playwright.config.ts ui/src-tauri/src \
	ui/src-tauri/capabilities ui/src-tauri/icons ui/src-tauri/.cargo \
	ui/src-tauri/Cargo.toml ui/src-tauri/Cargo.lock ui/src-tauri/.taurignore \
	ui/src-tauri/build.rs ui/src-tauri/tauri.conf.json \
	ui/src-tauri/tauri.macos.conf.json ui/src-tauri/tauri.windows.conf.json \
	ui/src-tauri/tauri.linux.conf.json \
	tests/ui/unit tests/ui/browser tests/ui/data tests/ui/analysis tests/ui/release tests/ui/config \
	tests/ui/package.json tests/ui/tsconfig.json \
	$(wildcard tests/ui/*.mjs tests/ui/*.py tests/ui/*.sh tests/ui/*.ps1 tests/ui/*.md)
RELEASE_RUNTIME_INPUT_ROOTS := data/rrtmg data/thompson
RELEASE_SOURCE_ARGS = $(foreach path,$(RELEASE_SOURCE_ROOTS) $(RELEASE_RUNTIME_INPUT_ROOTS),--source $(path))

define record_release_gate
	mkdir -p reports/pre_beta/gates
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/qualification/provenance/record_gate_result.py \
	  --root . --target $(1) --binary bin/nimbus_sim $(2) \
	  --log reports/pre_beta/gates/$(1).log \
	  --receipt reports/pre_beta/gates/$(1).json $(3) \
	  $(RELEASE_SOURCE_ARGS) \
	  -- $(MAKE) $(1)
endef

reports/pre_beta/gates/test-catalog-authority.json: bin/nimbus_sim
	$(call record_release_gate,test-catalog-authority,,)

reports/pre_beta/gates/test-catalog-reachable-matrix.json: qualify-catalog-reachable-evidence
	$(call record_release_gate,test-catalog-reachable-matrix,,--artifact reports/pre_beta/reachable_set_v1.json)

reports/pre_beta/gates/test-catalog-studio-native-parity.json: bin/nimbus_sim
	$(call record_release_gate,test-catalog-studio-native-parity,,)

reports/pre_beta/gates/qualify-catalog-bounded-runs.json: bin/nimbus_sim
	$(call record_release_gate,qualify-catalog-bounded-runs,--required-hardware,)

reports/pre_beta/gates/qualify-catalog-restart-threading.json: bin/nimbus_sim
	$(call record_release_gate,qualify-catalog-restart-threading,,)

reports/pre_beta/gates/qualify-catalog-physical-backends.json: bin/nimbus_sim
	$(call record_release_gate,qualify-catalog-physical-backends,--required-hardware,\
	  --artifact reports/pre_beta/physical/moving_nested_public_vulkan_admission_v1_pass.json \
	  --artifact reports/pre_beta/physical/moving_nested_vulkan_public_runtime_v1_pass.json \
	  --artifact reports/pre_beta/physical/moving_nested_metal_public_runtime_smoke_v1_pass.json \
	  --artifact reports/pre_beta/backend_tuple_operation_observation_v1.json \
	  --artifact tests/fixtures/reference/catalog/backend_tuple_operation_requirements_v1.json \
	  --artifact bin/nimbus_sim_physical_qualification \
	  --artifact bin/nimbus_sim_internal_qualification \
	  --artifact tests/artifacts/moving_nested_reference_frontier_v12_20260818/moving_child/checkpoints/checkpoint_step_000002700.nimbus \
	  --artifact tests/artifacts/moving_nested_reference_frontier_v12_20260818/moving_child/checkpoints/checkpoint_step_000002700.nimbus.json)

reports/pre_beta/gates/qualify-io.json: bin/nimbus_sim
	$(call record_release_gate,qualify-io,,)

reports/pre_beta/gates/qualify-release-final-surface.json: bin/nimbus_sim
	$(call record_release_gate,qualify-release-final-surface,,)

qualify-release-receipts: \
		reports/pre_beta/gates/test-catalog-authority.json \
		reports/pre_beta/gates/test-catalog-reachable-matrix.json \
		reports/pre_beta/gates/test-catalog-studio-native-parity.json \
		reports/pre_beta/gates/qualify-catalog-bounded-runs.json \
		reports/pre_beta/gates/qualify-catalog-restart-threading.json \
		reports/pre_beta/gates/qualify-catalog-physical-backends.json \
		reports/pre_beta/gates/qualify-io.json \
		reports/pre_beta/gates/qualify-release-final-surface.json

qualify-release-seal: qualify-release-receipts test-release-provenance
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/qualification/provenance/release_manifest.py --root . create \
	  --binary bin/nimbus_sim \
	  --config tests/fixtures/configs/qualification/agrid_flat_cpu_core_ready.yaml \
	  --backend cpu --campaign-id pre-beta-local-candidate-v2 \
	  --device-identity "local candidate; exact devices are bound by gate receipts" \
	  --toolchain-identity "local candidate; compiler identity recorded in manifest" \
	  $(RELEASE_SOURCE_ARGS) \
	  --evidence reports/pre_beta/reachable_set_v1.json \
	  --catalog-authority tests/fixtures/reference/catalog/authority_v1.json \
	  --reachable-set reports/pre_beta/reachable_set_v1.json \
	  --gate reports/pre_beta/gates/test-catalog-authority.json \
	  --gate reports/pre_beta/gates/test-catalog-reachable-matrix.json \
	  --gate reports/pre_beta/gates/test-catalog-studio-native-parity.json \
	  --gate reports/pre_beta/gates/qualify-catalog-bounded-runs.json \
	  --gate reports/pre_beta/gates/qualify-catalog-restart-threading.json \
	  --gate reports/pre_beta/gates/qualify-catalog-physical-backends.json \
	  --gate reports/pre_beta/gates/qualify-io.json \
	  --gate reports/pre_beta/gates/qualify-release-final-surface.json \
	  --package bin/nimbus_sim --package-kind raw-simulator \
	  --package-platform local --package-architecture "$$(uname -m)" \
	  --artifact reports/pre_beta/reachable_set_v1.cross_product.log \
	  --output reports/pre_beta/local_release_candidate_v2.json
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/qualification/provenance/release_manifest.py --root . verify \
	  --manifest reports/pre_beta/local_release_candidate_v2.json

test-studio-native-launch: bin/nimbus_sim
	cd ui/src-tauri && cargo test exact_binary_evidence_reaches_initialized_microphysics_identity --lib

test-mapped-terrain-ready: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_mapped_terrain_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_mapped_rrtmg_shortwave_ready.py --binary ./bin/nimbus_sim

test-collocated-mapped-terrain-ready: bin/nimbus_sim bin/test_dynamics_collocated_mapped_timestep bin/test_physics_microphysics bin/test_physics_chaos_runtime_state
	./bin/test_dynamics_collocated_mapped_timestep "[dynamics][terrain][collocated]"
	./bin/test_physics_microphysics
	./bin/test_physics_chaos_runtime_state
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_collocated_mapped_terrain_ready.py --binary ./bin/nimbus_sim

test-earth-fixed-terrain-startup: bin/nimbus_sim bin/test_init_config_v1 bin/test_physics_terrain bin/test_core_simulation_checkpoint
	./bin/test_init_config_v1 "[terrain]"
	./bin/test_physics_terrain "[physics][terrain][earth_fixed]"
	./bin/test_core_simulation_checkpoint "[checkpoint][terrain]"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_earth_fixed_terrain_startup.py --binary ./bin/nimbus_sim

.PHONY: test-richardson-smagorinsky-ready
test-richardson-smagorinsky-ready: bin/nimbus_sim bin/test_physics_richardson_smagorinsky bin/test_core_simulation_checkpoint bin/test_init_config_v1 bin/test_init_config_migrator
	./bin/test_physics_richardson_smagorinsky
	./bin/test_core_simulation_checkpoint "[richardson]"
	./bin/test_init_config_v1 "[richardson]"
	./bin/test_init_config_migrator "[richardson]"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_richardson_smagorinsky_ready.py --binary ./bin/nimbus_sim

.PHONY: test-earth-fixed-terrain-motion-ready
test-earth-fixed-terrain-motion-ready: bin/nimbus_sim bin/test_init_config_v1 bin/test_core_simulation_checkpoint bin/test_core_earth_fixed_step_transaction
	./bin/test_init_config_v1 "[terrain]"
	./bin/test_core_earth_fixed_step_transaction
	./bin/test_core_simulation_checkpoint
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_earth_fixed_terrain_motion_ready.py --binary ./bin/nimbus_sim

test-earth-fixed-terrain-recycling-ready: bin/nimbus_sim bin/test_core_domain_recycling bin/test_core_simulation_checkpoint bin/test_core_earth_fixed_step_transaction
	./bin/test_core_earth_fixed_step_transaction
	./bin/test_core_domain_recycling "[core][domain_recycling][terrain]"
	./bin/test_core_simulation_checkpoint "[checkpoint][terrain]"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_earth_fixed_terrain_recycling_ready.py --binary ./bin/nimbus_sim

test-nested-grid-recycling-ready: bin/nimbus_sim bin/test_core_domain_recycling bin/test_core_simulation_checkpoint
	./bin/test_core_domain_recycling "[domain_recycling][nested]"
	./bin/test_core_simulation_checkpoint "[checkpoint][nesting]"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_nested_grid_recycling_ready.py --binary ./bin/nimbus_sim

test-two-way-nested-ready: bin/nimbus_sim bin/test_core_nested_grid bin/test_core_nested_boundary_forcing bin/test_core_simulation_checkpoint bin/test_numerics_advection_cartesian bin/test_dynamics_cartesian_cgrid_slow_tendencies_gpu_parity bin/test_core_nested_production_step
	./bin/test_core_nested_grid "[core][nested_grid][two_way]"
	./bin/test_core_nested_boundary_forcing "[core][nested_grid][boundary][relaxation][momentum][conservation]"
	./bin/test_core_simulation_checkpoint "[checkpoint][nesting]"
	./bin/test_numerics_advection_cartesian "[interface-flux]"
	./bin/test_dynamics_cartesian_cgrid_slow_tendencies_gpu_parity "[momentum-interface-flux]"
	./bin/test_core_nested_production_step "[core][nesting][production][dynamics]"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_two_way_nested_ready.py --binary ./bin/nimbus_sim

test-active-terrain-nested-engineering: bin/test_core_nested_terrain_geometry
	OMP_NUM_THREADS=1 ./bin/test_core_nested_terrain_geometry "[core][nested_grid][active_terrain]"
	OMP_NUM_THREADS=4 ./bin/test_core_nested_terrain_geometry "[core][nested_grid][active_terrain]"

test-active-terrain-nested-ready: bin/nimbus_sim bin/test_core_nested_terrain_geometry bin/test_core_nested_grid bin/test_core_nested_level_runtime bin/test_core_simulation_checkpoint
	OMP_NUM_THREADS=1 ./bin/test_core_nested_terrain_geometry "[core][nested_grid][active_terrain]"
	OMP_NUM_THREADS=4 ./bin/test_core_nested_terrain_geometry "[core][nested_grid][active_terrain]"
	./bin/test_core_simulation_checkpoint "[checkpoint][nesting][active_terrain]"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_active_terrain_nested_ready.py --binary ./bin/nimbus_sim

test-native-mapped-terrain-ready: bin/nimbus_sim bin/test_vulkan_mapped_mass_continuity
	./scripts/run_with_vulkan_env.sh ./bin/test_vulkan_mapped_mass_continuity
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_native_mapped_terrain_ready.py --backend vulkan --binary $(PHYSICAL_QUALIFICATION_BINARY)
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_native_mapped_terrain_ready.py --backend vulkan --terrain schar --binary $(PHYSICAL_QUALIFICATION_BINARY)
	$(MAKE) -C metal all
	./bin/test_metal_compute_backend "[metal][terrain]"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_native_mapped_terrain_ready.py --backend metal --binary $(PHYSICAL_QUALIFICATION_BINARY)
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_native_mapped_terrain_ready.py --backend metal --terrain schar --binary $(PHYSICAL_QUALIFICATION_BINARY)

test-tke-rrtmg-ready: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_tke_rrtmg_ready.py --binary ./bin/nimbus_sim

.PHONY: test-native-tke-ready
test-native-tke-ready: bin/nimbus_sim bin/test_physics_native_tke bin/test_physics_tke_scheme
	python3 scripts/dev/shaders/generate_tke_msl.py --check
	./bin/test_physics_tke_scheme
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_tke "[native-tke][vulkan]"
	./bin/test_physics_native_tke "[native-tke][metal]"
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_native_tke_ready.py --backend vulkan --binary $(PHYSICAL_QUALIFICATION_BINARY)
	python3 scripts/tests/science/test_native_tke_ready.py --backend metal --binary $(PHYSICAL_QUALIFICATION_BINARY)
	python3 scripts/tests/science/test_tke_rrtmg_ready.py --binary $(PHYSICAL_QUALIFICATION_BINARY)

test-turbulence-closure-ready: bin/nimbus_sim bin/test_physics_turbulence_sgs bin/test_physics_tke_scheme bin/test_physics_turbulence_runtime_state bin/test_init_config_v1 bin/test_core_simulation_checkpoint
	./bin/test_physics_turbulence_sgs
	./bin/test_physics_tke_scheme
	./bin/test_physics_turbulence_runtime_state
	./bin/test_init_config_v1
	./bin/test_core_simulation_checkpoint
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_dynamic_smagorinsky_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_tke_rrtmg_ready.py --binary ./bin/nimbus_sim

test-user-filter-width-ready: bin/nimbus_sim bin/test_physics_turbulence_sgs bin/test_physics_turbulence_runtime_state bin/test_init_config_v1 bin/test_core_simulation_checkpoint
	./bin/test_physics_turbulence_sgs
	./bin/test_physics_turbulence_runtime_state
	./bin/test_init_config_v1
	./bin/test_core_simulation_checkpoint
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_user_filter_width_ready.py --binary ./bin/nimbus_sim

test-coupled-rk-ready: bin/nimbus_sim bin/test_numerics_timestepping bin/test_init_config_v1
	./bin/test_numerics_timestepping
	./bin/test_init_config_v1
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_coupled_rk_ready.py --binary ./bin/nimbus_sim

test-public-config-admission: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/config/test_public_preset_admission.py --binary ./bin/nimbus_sim

CATALOG_RUST_TARGET_DIR ?= $(CURDIR)/build/catalog-rust-target

test-catalog-authority: bin/nimbus_sim bin/test_core_scheme_catalog
	./bin/test_core_scheme_catalog
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_catalog_authority.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest scripts.tests.catalog.test_catalog_authority_contract
	cd ui && npm run test:scheme-catalog
	cd ui/src-tauri && CARGO_TARGET_DIR=$(CATALOG_RUST_TARGET_DIR) cargo test config_preflight --lib

test-catalog-reachable-matrix: test-catalog-authority
	./bin/test_core_scheme_catalog "[cross-product]"
	cd ui && npm run test:coupling

test-catalog-browser-snapshot: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_browser_catalog_snapshot.py --binary ./bin/nimbus_sim

test-catalog-studio-native-parity: test-catalog-authority test-catalog-browser-snapshot
	CARGO_TARGET_DIR=$(CATALOG_RUST_TARGET_DIR) PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/test_studio_native_parity.py --binary ./bin/nimbus_sim

qualify-catalog-bounded-runs: test-catalog-studio-native-parity
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/qualify_catalog_bounded_runs.py --binary ./bin/nimbus_sim

qualify-catalog-restart-threading: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/catalog/qualify_catalog_restart_threading.py \
	  --binary ./bin/nimbus_sim

qualify-catalog-physical-backends: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 \
	  scripts/tests/catalog/qualify_catalog_physical_backends.py \
	  --binary ./bin/nimbus_sim

test-ready-admission-matrix: bin/nimbus_sim bin/test_core_scheme_catalog test-public-config-admission
	./bin/test_core_scheme_catalog
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_ready_admission_matrix.py --binary ./bin/nimbus_sim

test-weno5-rrtmg-ready: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_cgrid_numerics_ready.py --profile weno5 --binary ./bin/nimbus_sim

test-implicit-rrtmg-ready: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_cgrid_numerics_ready.py --profile implicit --binary ./bin/nimbus_sim

.PHONY: test-native-implicit-ready
test-native-implicit-ready: bin/nimbus_sim bin/test_numerics_native_implicit bin/test_numerics_diffusion
	python3 scripts/dev/shaders/generate_implicit_msl.py --check
	./bin/test_numerics_diffusion
	./bin/test_numerics_native_implicit "[implicit][scaling]"
	./bin/test_numerics_native_implicit "[native-implicit][metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_implicit "[native-implicit][vulkan]"
	python3 scripts/tests/science/test_native_implicit_ready.py --backend metal --binary $(PHYSICAL_QUALIFICATION_BINARY)
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_native_implicit_ready.py --backend vulkan --binary $(PHYSICAL_QUALIFICATION_BINARY)
	python3 scripts/tests/science/test_cgrid_numerics_ready.py --profile implicit --binary $(PHYSICAL_QUALIFICATION_BINARY)

test-pbl-rrtmg-ready: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_pbl_rrtmg_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_diagnostic_science_ready.py --binary ./bin/nimbus_sim

test-cloud-top-radiative-ysu-ready: bin/nimbus_sim bin/test_physics_boundary_layer bin/test_physics_boundary_layer_runtime_state bin/test_physics_microphysics_runtime_state bin/test_init_config_v1 bin/test_init_config_migrator bin/test_core_simulation_checkpoint
	./bin/test_physics_boundary_layer
	./bin/test_physics_boundary_layer_runtime_state
	./bin/test_physics_microphysics_runtime_state
	./bin/test_init_config_v1
	./bin/test_init_config_migrator
	./bin/test_core_simulation_checkpoint
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_cloud_top_radiative_ysu_ready.py --binary ./bin/nimbus_sim

test-lin-rrtmg-ready: bin/nimbus_sim bin/test_qualification_lin_reference_processes
	./bin/test_qualification_lin_reference_processes
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_lin_rrtmg_ready.py --binary ./bin/nimbus_sim

test-thompson-cpu-ready: bin/nimbus_sim bin/test_physics_thompson_cold_processes
	OMP_NUM_THREADS=1 ./bin/test_physics_thompson_cold_processes
	OMP_NUM_THREADS=4 ./bin/test_physics_thompson_cold_processes
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_thompson_cpu_ready.py --binary ./bin/nimbus_sim

test-lin-rrtmg-vulkan-ready: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_lin_rrtmg_accelerator_ready.py --backend vulkan --binary $(PHYSICAL_QUALIFICATION_BINARY)

test-lin-rrtmg-metal-ready: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_lin_rrtmg_accelerator_ready.py --backend metal --binary $(PHYSICAL_QUALIFICATION_BINARY)

test-native-chaos-vulkan-ready: bin/nimbus_sim bin/test_vulkan_gpu_parity
	./scripts/run_with_vulkan_env.sh ./bin/test_vulkan_gpu_parity "Vulkan Earth-control evolution preserves Gaussian and AR reconstruction contracts atomically"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_native_chaos_accelerator_ready.py --backend vulkan --binary $(PHYSICAL_QUALIFICATION_BINARY)
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_native_chaos_restart.py --backend vulkan --binary $(PHYSICAL_QUALIFICATION_BINARY)

test-native-chaos-metal-ready: bin/nimbus_sim
	$(MAKE) -C metal all
	./bin/test_metal_compute_backend "Metal Earth-control evolution preserves Gaussian and AR reconstruction contracts atomically"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_native_chaos_accelerator_ready.py --backend metal --binary $(PHYSICAL_QUALIFICATION_BINARY)
	python3 scripts/tests/science/test_native_chaos_restart.py --backend metal --binary $(PHYSICAL_QUALIFICATION_BINARY)

test-variable-diffusion-ready: bin/nimbus_sim bin/test_vulkan_gpu_parity
	./scripts/run_with_vulkan_env.sh ./bin/test_vulkan_gpu_parity "[vulkan][gpu_parity][diffusion]"
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_variable_diffusion_ready.py --backend vulkan --binary $(PHYSICAL_QUALIFICATION_BINARY)
	$(MAKE) -C metal all
	./bin/test_metal_compute_backend "[metal][diffusion]"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_variable_diffusion_ready.py --backend metal --binary $(PHYSICAL_QUALIFICATION_BINARY)

test-simple-grey-absorbing-ready: bin/nimbus_sim bin/test_physics_radiation bin/test_physics_radiation_runtime_state bin/test_init_config_v1 bin/test_core_simulation_checkpoint
	./bin/test_physics_radiation "Absorbing water-cloud simple grey*"
	./bin/test_physics_radiation_runtime_state "absorbing water-cloud simple-grey*"
	./bin/test_init_config_v1 "simple-grey radiation requires explicit physical solar inputs"
	./bin/test_core_simulation_checkpoint "simple-grey checkpoint*"
	./bin/test_core_simulation_checkpoint "simulation checkpoint is deterministic and restores an independent run"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_simple_grey_absorbing_water_cloud_ready.py --binary ./bin/nimbus_sim

test-radar-ready: bin/nimbus_sim bin/test_physics_reflectivity bin/test_physics_radar_velocity bin/test_physics_radar_zdr bin/test_physics_radar_runtime_state bin/test_init_config_v1 bin/test_core_output bin/test_core_simulation_checkpoint
	./bin/test_physics_reflectivity
	./bin/test_physics_radar_velocity
	./bin/test_physics_radar_zdr
	./bin/test_physics_radar_runtime_state
	./bin/test_init_config_v1
	./bin/test_core_output
	./bin/test_core_simulation_checkpoint
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_radar_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_radar_polarimetric_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/qualification/output/qualify_radar_product_sets.py --binary ./bin/nimbus_sim

test-domain-recycling-ready: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_domain_recycling_ready.py --binary ./bin/nimbus_sim

test-tracked-window-ready: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_tracked_window_ready.py --binary ./bin/nimbus_sim

test-pbl-top-ready: bin/nimbus_sim bin/test_physics_boundary_layer bin/test_physics_boundary_layer_runtime_state bin/test_init_config_v1 bin/test_init_config_migrator
	./bin/test_physics_boundary_layer "PBL schemes publish the selected theta and TKE top diagnostics"
	./bin/test_physics_boundary_layer "potential-temperature PBL height interpolates the 1.5 K crossing"
	./bin/test_physics_boundary_layer "TKE PBL height interpolates the turbulent-energy threshold"
	./bin/test_physics_boundary_layer_runtime_state "boundary-layer runtime initialization rejects invalid configuration"
	./bin/test_init_config_v1 "load_v1_config: surface and boundary-layer ownership is explicit"
	./bin/test_init_config_migrator "migrate: PBL-top diagnosis survives v0 and v1 projection"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_pbl_top_ready.py --binary ./bin/nimbus_sim

test-agrid-core-ready: bin/nimbus_sim bin/test_dynamics_cartesian bin/test_physics_turbulence_sgs bin/test_physics_tke_scheme bin/test_physics_boundary_layer bin/test_physics_radiation bin/test_physics_chaos_runtime_state
	./bin/test_dynamics_cartesian "[coriolis]"
	./bin/test_physics_turbulence_sgs
	./bin/test_physics_tke_scheme
	./bin/test_physics_boundary_layer
	./bin/test_physics_radiation
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_agrid_core_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_agrid_turbulence_family_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_agrid_column_family_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_agrid_milbrandt_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_agrid_lin_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_agrid_thompson_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_agrid_implicit_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_agrid_weno_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_agrid_rk_ready.py --binary ./bin/nimbus_sim
	./bin/test_physics_chaos_runtime_state
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_agrid_chaos_ready.py --binary ./bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/qualification/domain_motion/qualify_constant_velocity_domain.py --binary ./bin/nimbus_sim

test-agrid-native-ready: bin/nimbus_sim bin/test_vulkan_gpu_parity
	./scripts/run_with_vulkan_env.sh ./bin/test_vulkan_gpu_parity "Vulkan native A-grid completion stages match the CPU operators atomically"
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_agrid_native_complete_ready.py --backend vulkan --binary $(PHYSICAL_QUALIFICATION_BINARY)
	$(MAKE) -C metal all
	./bin/test_metal_compute_backend "Metal native A-grid completion stages match the CPU operators atomically"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_agrid_native_complete_ready.py --backend metal --binary $(PHYSICAL_QUALIFICATION_BINARY)

# Release-facing Studio workflow gate. This generates a synthetic netCDF4
# CM1-style control from maintained CDL, advances the exact 30 s guided Nimbus
# config, and requires native field reads plus immutable launch/runtime and Run
# Library evidence. It intentionally remains outside the fast aggregate suite.
qualify-cm1-guided-smoke: bin/nimbus_sim
	command -v ncgen >/dev/null
	cd ui/src-tauri && cargo test maintained_cm1_guided_smoke_qualifies_native_studio_evidence --lib -- --ignored --nocapture

test-integration: bin/test_integration test-studio-native-launch test-benchmark-runner test-qualification-tools
	./bin/test_integration

test-shm-e2e: bin/test_shm_e2e
	./bin/test_shm_e2e

# Parity-harness smoke (GPU.md Phase 1): runs a small Cartesian
# C-grid fixture under backend: gpu with the harness on; skips cleanly when
# no GPU is available.
test-parity: $(BIN)
	bash ./scripts/tests/gpu/test_parity_smoke.sh

test: test-script-suite test-compute test-core test-catalog-backend-ownership test-catalog-backend-process-ownership test-catalog-backend-kernel-ownership test-diagnostics test-dynamics test-numerics test-physics test-data test-init test-vulkan test-integration test-shm-e2e test-parity
	@echo "=== All tests passed ==="

smoke-test: $(BIN)
	./$(BIN) --headless --config=configs/1000meterResolution/kessler/baseline.yaml --duration=5 2>/dev/null && echo "[smoke-test] PASS"

smoke-test-e2e: $(BIN) vulkan
	bash ./tests/science/smoke/smoke_test_e2e.sh

# Deliberately separate from the fast aggregate suite: this advances a 32^3
# production C-grid for one simulated hour under strict validation (~14 s on
# the audited M4 Max) and checks quantitative balance/EOS/inventory thresholds.
qualify-science-hydrostatic: $(BIN)
	bash ./tests/science/hydrostatic/qualify_hydrostatic_hold.sh

qualify-science-core: $(BIN) bin/test_numerics_staggered_derivatives bin/test_numerics_advection_cartesian bin/test_numerics_tvd_monotonicity bin/test_dynamics_cartesian_lateral_bc_yaml
	./bin/test_numerics_staggered_derivatives "[convergence]"
	./bin/test_numerics_advection_cartesian "[manufactured]"
	./bin/test_numerics_tvd_monotonicity "[convergence]"
	./bin/test_dynamics_cartesian_lateral_bc_yaml "[SCI-017][acoustic]"
	bash ./tests/science/hydrostatic/qualify_hydrostatic_pair.sh

qualify-science-diffusion: $(BIN)
	python3 ./tests/science/cgrid/physics/qualify_active_diffusion_storm.py

# Fail-closed native-device qualification. This intentionally fails rather
# than skips on a lane that does not provide a usable Vulkan device.
qualify-science-diffusion-gpu: bin/test_vulkan_required_dispatch bin/test_vulkan_gpu_parity
	./scripts/run_with_vulkan_env.sh ./bin/test_vulkan_required_dispatch
	./scripts/run_with_vulkan_env.sh ./bin/test_vulkan_gpu_parity "[diffusion]"

qualify-gpu-residency-equivalence: $(BIN)
	./scripts/run_with_vulkan_env.sh python3 ./scripts/qualification/accelerators/qualify_gpu_residency_equivalence.py --binary $(BIN)

qualify-vulkan-strict: $(BIN)
	NIMBUS_BIN=$(BIN) bash ./scripts/qualification/accelerators/qualify_vulkan_strict.sh

qualify-vulkan-report-bearing: $(BIN)
	NIMBUS_BIN=$(BIN) bash ./scripts/qualification/accelerators/qualify_vulkan_report_bearing.sh

qualify-vulkan-restart: $(BIN)
	./scripts/run_with_vulkan_env.sh python3 ./scripts/qualification/checkpointing/qualify_vulkan_restart.py --binary $(BIN)

qualify-vulkan-scalar-owner-parity: bin/test_numerics_advect_conservative_cartesian_cgrid_gpu_parity
	NIMBUS_GPU_REQUIRED=1 ./scripts/run_with_vulkan_env.sh ./bin/test_numerics_advect_conservative_cartesian_cgrid_gpu_parity

qualify-vulkan-acoustic-owner-parity: bin/test_dynamics_acoustic_cartesian_cgrid_gpu_parity
	NIMBUS_GPU_REQUIRED=1 ./scripts/run_with_vulkan_env.sh ./bin/test_dynamics_acoustic_cartesian_cgrid_gpu_parity

qualify-vulkan-slow-path-parity: bin/test_dynamics_cartesian_cgrid_slow_tendencies_gpu_parity
	NIMBUS_GPU_REQUIRED=1 ./scripts/run_with_vulkan_env.sh ./bin/test_dynamics_cartesian_cgrid_slow_tendencies_gpu_parity "~[SCI-170]"

qualify-vulkan-slow-owner-parity: bin/test_dynamics_cartesian_cgrid_slow_tendencies_gpu_parity
	NIMBUS_GPU_REQUIRED=1 ./scripts/run_with_vulkan_env.sh ./bin/test_dynamics_cartesian_cgrid_slow_tendencies_gpu_parity "[SCI-170]"

qualify-vulkan-sponge-owner-parity: bin/test_core_rayleigh_damping_yaml
	NIMBUS_GPU_REQUIRED=1 ./scripts/run_with_vulkan_env.sh ./bin/test_core_rayleigh_damping_yaml "[SCI-171]"

VULKAN_MIXED_DURATION ?= 140
VULKAN_MIXED_CHECKPOINT ?= 70

qualify-vulkan-mixed-physics: $(BIN) bin/test_numerics_advect_conservative_cartesian_cgrid_gpu_parity
	NIMBUS_GPU_REQUIRED=1 ./scripts/run_with_vulkan_env.sh ./bin/test_numerics_advect_conservative_cartesian_cgrid_gpu_parity
	./scripts/run_with_vulkan_env.sh python3 ./scripts/qualification/checkpointing/qualify_vulkan_restart.py --binary $(BIN) --config tests/fixtures/configs/qualification/cgrid_coupled_physics.yaml --duration $(VULKAN_MIXED_DURATION) --checkpoint $(VULKAN_MIXED_CHECKPOINT) --require-mixed-physics

VULKAN_COMBINATION_DURATION ?= 4

qualify-vulkan-combinations: $(BIN)
	./scripts/run_with_vulkan_env.sh python3 ./scripts/qualification/accelerators/qualify_vulkan_combinations.py \
	  --binary $(BIN) --duration $(VULKAN_COMBINATION_DURATION) \
	  --report reports/qualification/vulkan-combination-matrix-current.json

benchmark-vulkan-report-overhead: $(BIN)
	./scripts/run_with_vulkan_env.sh python3 ./scripts/benchmarks/gpu/benchmark_vulkan_report_overhead.py --binary $(BIN)

VULKAN_MINIMUM_ACOUSTIC_REPORT_DURATION ?= 120

benchmark-vulkan-minimum-acoustic-report-overhead: $(BIN)
	./scripts/run_with_vulkan_env.sh python3 ./scripts/benchmarks/gpu/benchmark_vulkan_report_overhead.py \
	  --binary $(BIN) --duration $(VULKAN_MINIMUM_ACOUSTIC_REPORT_DURATION) \
	  --acoustic-substeps 2 --dt-large 0.25 \
	  --report reports/benchmarks/vulkan-minimum-acoustic-report-overhead-current.json

benchmark-vulkan-scalar-attribution: $(BENCHMARK_BIN) \
		$(VULKAN_SCALAR_ATTRIBUTION_COMPARATOR_BIN)
	./scripts/run_with_vulkan_env.sh python3 \
	  ./scripts/benchmarks/gpu/benchmark_vulkan_scalar_attribution.py \
	  --optimized $(BENCHMARK_BIN) \
	  --comparator $(VULKAN_SCALAR_ATTRIBUTION_COMPARATOR_BIN)

benchmark-vulkan-scalar-regions: $(BENCHMARK_BIN) \
		$(VULKAN_SCALAR_REGIONS_EXPERIMENT_BIN)
	./scripts/run_with_vulkan_env.sh python3 \
	  ./scripts/benchmarks/gpu/benchmark_vulkan_scalar_attribution.py \
	  --optimized $(VULKAN_SCALAR_REGIONS_EXPERIMENT_BIN) \
	  --comparator $(BENCHMARK_BIN) \
	  --label vulkan-scalar-regions \
	  --schema nimbus.vulkan-scalar-regions.v2 \
	  --comparison-contract "same equations, FP32 device commit, binary64 fixed-slab reports, cadence, ledgers, and endpoint; experiment only combines face/limiter OpenMP teams and omits the unused theta limiter traversal" \
	  --output reports/benchmarks/vulkan-scalar-regions-current.json

VULKAN_SGS_APPLY_TRIALS ?= 7
VULKAN_SGS_APPLY_DURATION ?= 120

benchmark-vulkan-sgs-apply: $(BIN)
	./scripts/run_with_vulkan_env.sh python3 \
	  ./scripts/benchmarks/gpu/benchmark_vulkan_sgs_apply.py \
	  --binary $(BIN) \
	  --duration $(VULKAN_SGS_APPLY_DURATION) \
	  --trials $(VULKAN_SGS_APPLY_TRIALS)

VULKAN_NOOP_BUDGET_TRIALS ?= 15

benchmark-vulkan-noop-budget-scans: $(BENCHMARK_BIN) \
		$(VULKAN_NOOP_BUDGET_COMPARATOR_BIN)
	./scripts/run_with_vulkan_env.sh python3 \
	  ./scripts/benchmarks/gpu/benchmark_vulkan_scalar_attribution.py \
	  --optimized $(BENCHMARK_BIN) \
	  --comparator $(VULKAN_NOOP_BUDGET_COMPARATOR_BIN) \
	  --label vulkan-noop-budget-scans \
	  --schema nimbus.vulkan-noop-budget-scans.v2 \
	  --comparison-contract "same equations, cadence, report values, ledgers, and endpoint; comparator restores only disabled-diffusion primary-state verification and residency invalidation plus the provably redundant post-disabled-diffusion and post-boundary conservation scans" \
	  --gate-policy whole-step \
	  --trials $(VULKAN_NOOP_BUDGET_TRIALS) \
	  --output reports/benchmarks/vulkan-noop-budget-scans-current.json

benchmark-vulkan-endpoint-moisture: $(BENCHMARK_BIN) \
		$(VULKAN_ENDPOINT_MOISTURE_COMPARATOR_BIN)
	./scripts/run_with_vulkan_env.sh python3 \
	  ./scripts/benchmarks/gpu/benchmark_vulkan_scalar_attribution.py \
	  --optimized $(BENCHMARK_BIN) \
	  --comparator $(VULKAN_ENDPOINT_MOISTURE_COMPARATOR_BIN) \
	  --label vulkan-endpoint-moisture \
	  --schema nimbus.vulkan-endpoint-moisture.v1 \
	  --comparison-contract "same equations, diagnostic arithmetic, fixed-slab merge order, cadence, fields, ledgers, and endpoint; comparator restores only the two standalone endpoint moisture traversals" \
	  --output reports/benchmarks/vulkan-endpoint-moisture-current.json

benchmark-vulkan-microphysics-gate: $(BENCHMARK_BIN) \
		$(VULKAN_MICROPHYSICS_GATE_COMPARATOR_BIN)
	./scripts/run_with_vulkan_env.sh python3 \
	  ./scripts/benchmarks/gpu/benchmark_vulkan_scalar_attribution.py \
	  --optimized $(BENCHMARK_BIN) \
	  --comparator $(VULKAN_MICROPHYSICS_GATE_COMPARATOR_BIN) \
	  --label vulkan-microphysics-gate \
	  --schema nimbus.vulkan-microphysics-gate.v1 \
	  --comparison-contract "same microphysics tendencies, conservation expressions and parenthesization, fixed-x-slab association, compensated merge order, guards, cadence, fields, ledgers, and endpoint; comparator restores only the standalone conservation-gate traversal" \
	  --output reports/benchmarks/vulkan-microphysics-gate-current.json

benchmark-vulkan-microphysics-validation: $(BENCHMARK_BIN) \
		$(VULKAN_MICROPHYSICS_VALIDATION_COMPARATOR_BIN)
	./scripts/run_with_vulkan_env.sh python3 \
	  ./scripts/benchmarks/gpu/benchmark_vulkan_scalar_attribution.py \
	  --optimized $(BENCHMARK_BIN) \
	  --comparator $(VULKAN_MICROPHYSICS_VALIDATION_COMPARATOR_BIN) \
	  --label vulkan-microphysics-validation \
	  --schema nimbus.vulkan-microphysics-validation.v1 \
	  --comparison-contract "same tendency fields, physical domain, nonfinite predicates, exact integer theta/total counts, guards, cadence, fields, ledgers, and endpoint; comparator restores only the eight separate tendency-validation scans" \
	  --output reports/benchmarks/vulkan-microphysics-validation-current.json

benchmark-vulkan-microphysics-input-validation: $(BENCHMARK_BIN) \
		$(VULKAN_MICROPHYSICS_INPUT_VALIDATION_COMPARATOR_BIN)
	./scripts/run_with_vulkan_env.sh python3 \
	  ./scripts/benchmarks/gpu/benchmark_vulkan_scalar_attribution.py \
	  --optimized $(BENCHMARK_BIN) \
	  --comparator $(VULKAN_MICROPHYSICS_INPUT_VALIDATION_COMPARATOR_BIN) \
	  --label vulkan-microphysics-input-validation \
	  --schema nimbus.vulkan-microphysics-input-validation.v1 \
	  --comparison-contract "same physical domain, pressure/density/theta/water/coupled-tendency predicates, serial fail-first cell and category, guards, fields, ledgers, and endpoint; comparator restores only the branching scalar input-validation loop" \
	  --gate-policy whole-step \
	  --output reports/benchmarks/vulkan-microphysics-input-validation-current.json

qualify-vulkan:
	$(MAKE) verify-shaders
	$(MAKE) qualify-science-diffusion-gpu
	$(MAKE) qualify-gpu-residency-equivalence
	$(MAKE) qualify-vulkan-strict
	$(MAKE) qualify-vulkan-report-bearing
	$(MAKE) qualify-vulkan-restart
	$(MAKE) qualify-vulkan-scalar-owner-parity
	$(MAKE) qualify-vulkan-acoustic-owner-parity
	$(MAKE) qualify-vulkan-slow-path-parity
	$(MAKE) qualify-vulkan-slow-owner-parity
	$(MAKE) qualify-vulkan-sponge-owner-parity
	$(MAKE) qualify-vulkan-mixed-physics
	$(MAKE) qualify-vulkan-combinations

.PHONY: qualify-vulkan-evidence
qualify-vulkan-evidence: $(BIN)
	$(MAKE) build-fingerprint
	./scripts/run_with_vulkan_env.sh python3 ./scripts/qualification/accelerators/qualify_vulkan_evidence.py \
	  --binary $(BIN) $(if $(strip $(VULKAN_HEADERS_PREFIX)),--make-arg=VULKAN_HEADERS_PREFIX=$(VULKAN_HEADERS_PREFIX),)

test-all: test smoke-test smoke-test-e2e

bin/test_physics_native_thompson_completion: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_thompson_completion.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-thompson-completion test-thompson-native-ready
test-native-thompson-completion: bin/test_physics_native_thompson_completion bin/test_physics_native_thompson_sedimentation
	./bin/test_physics_native_thompson_completion "[native-thompson-completion][cpu]"
	./bin/test_physics_native_thompson_completion "[native-thompson-completion][metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_completion "[native-thompson-completion][vulkan]"
	./scripts/run_with_vulkan_env.sh env NIMBUS_VULKAN_MEMORY_POLICY=staged NIMBUS_GPU_RESIDENT=0 ./bin/test_physics_native_thompson_completion "[native-thompson-completion][vulkan]"
	./bin/test_physics_native_thompson_sedimentation "[native-thompson][metal][coupling]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_sedimentation "[native-thompson][vulkan][coupling]"

test-thompson-native-ready: bin/nimbus_sim test-native-thompson-completion bin/test_physics_native_thompson_phase bin/test_physics_thompson_nucleation bin/test_physics_thompson_cold_processes
	./bin/test_physics_thompson_nucleation
	./bin/test_physics_thompson_cold_processes
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_phase
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_thompson_accelerator_ready.py --binary ./bin/nimbus_sim

.PHONY: test-native-tke-implicit-ready
test-native-tke-implicit-ready: bin/nimbus_sim bin/test_numerics_diffusion
	./bin/test_numerics_diffusion "[implicit]"
	python3 scripts/tests/science/test_native_tke_ready.py --implicit --backend metal --binary ./bin/nimbus_sim
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_native_tke_ready.py --implicit --backend vulkan --binary ./bin/nimbus_sim

bin/test_physics_native_thompson_implicit: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_thompson_implicit.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-thompson-implicit-ready
test-native-thompson-implicit-ready: bin/nimbus_sim bin/test_physics_native_thompson_implicit
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_thompson_implicit
	python3 scripts/tests/science/test_thompson_implicit_ready.py --binary ./bin/nimbus_sim

.PHONY: test-native-thompson-tke-ready
test-native-thompson-tke-ready: bin/nimbus_sim bin/test_core_scheme_catalog
	./bin/test_core_scheme_catalog
	python3 scripts/tests/science/test_thompson_tke_ready.py --binary ./bin/nimbus_sim

.PHONY: test-native-weno-ready
test-native-weno-ready: bin/nimbus_sim bin/test_numerics_advection_cartesian bin/test_numerics_weno5_reconstruction
	./bin/test_numerics_advection_cartesian
	./bin/test_numerics_weno5_reconstruction
	python3 scripts/tests/science/test_weno_native_family_ready.py --binary ./bin/nimbus_sim

.PHONY: test-native-thompson-tke-weno-ready
test-native-thompson-tke-weno-ready: bin/nimbus_sim bin/test_core_scheme_catalog bin/test_numerics_advection_cartesian
	./bin/test_core_scheme_catalog
	./bin/test_numerics_advection_cartesian
	python3 scripts/tests/science/test_thompson_tke_weno_ready.py --binary ./bin/nimbus_sim

.PHONY: test-native-tke-weno-ready
test-native-tke-weno-ready: bin/nimbus_sim bin/test_core_scheme_catalog bin/test_numerics_advection_cartesian bin/test_physics_native_tke
	./bin/test_core_scheme_catalog
	./bin/test_numerics_advection_cartesian
	./bin/test_physics_native_tke "[native-tke][metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_tke "[native-tke][vulkan]"
	python3 scripts/tests/science/test_tke_weno_ready.py --binary ./bin/nimbus_sim

.PHONY: test-native-thompson-weno-ready
test-native-thompson-weno-ready: bin/nimbus_sim bin/test_core_scheme_catalog bin/test_numerics_advection_cartesian
	./bin/test_core_scheme_catalog
	./bin/test_numerics_advection_cartesian
	python3 scripts/tests/science/test_thompson_weno_ready.py --binary ./bin/nimbus_sim

bin/test_physics_native_milbrandt_warm: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_milbrandt_warm.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-milbrandt-warm
test-native-milbrandt-warm: bin/test_physics_native_milbrandt_warm
	python3 scripts/dev/shaders/generate_milbrandt_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./bin/test_physics_native_milbrandt_warm "[metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_milbrandt_warm "[vulkan]"

bin/test_physics_native_milbrandt_cloud: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_milbrandt_cloud.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-milbrandt-cloud
test-native-milbrandt-cloud: bin/test_physics_native_milbrandt_cloud
	python3 scripts/dev/shaders/generate_milbrandt_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./bin/test_physics_native_milbrandt_cloud "[metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_milbrandt_cloud "[vulkan]"

bin/test_physics_native_milbrandt_melting: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_milbrandt_melting.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-milbrandt-melting
test-native-milbrandt-melting: bin/test_physics_native_milbrandt_melting
	python3 scripts/dev/shaders/generate_milbrandt_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./bin/test_physics_native_milbrandt_melting "[metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_milbrandt_melting "[vulkan]"

bin/test_physics_native_milbrandt_cold: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_milbrandt_cold.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-milbrandt-cold
test-native-milbrandt-cold: bin/test_physics_native_milbrandt_cold
	python3 scripts/dev/shaders/generate_milbrandt_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./bin/test_physics_native_milbrandt_cold "[metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_milbrandt_cold "[vulkan]"

bin/test_physics_native_milbrandt_sedimentation: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_milbrandt_sedimentation.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-milbrandt-sedimentation
test-native-milbrandt-sedimentation: bin/test_physics_native_milbrandt_sedimentation
	python3 scripts/dev/shaders/generate_milbrandt_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./bin/test_physics_native_milbrandt_sedimentation "[metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_milbrandt_sedimentation "[vulkan]"

bin/test_physics_native_milbrandt_scheme: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_milbrandt_scheme.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-milbrandt-scheme
test-native-milbrandt-scheme: bin/test_physics_native_milbrandt_scheme
	python3 scripts/dev/shaders/generate_milbrandt_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./bin/test_physics_native_milbrandt_scheme "[metal]"
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_milbrandt_scheme "[vulkan]"

.PHONY: test-native-milbrandt qualify-science-milbrandt-native
test-native-milbrandt: bin/test_physics_native_milbrandt_cloud bin/test_physics_native_milbrandt_warm bin/test_physics_native_milbrandt_cold bin/test_physics_native_milbrandt_melting bin/test_physics_native_milbrandt_sedimentation bin/test_physics_native_milbrandt_scheme
	python3 scripts/dev/shaders/generate_milbrandt_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_milbrandt_cloud
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_milbrandt_warm
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_milbrandt_cold
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_milbrandt_melting
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_milbrandt_sedimentation
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_milbrandt_scheme

qualify-science-milbrandt-native: bin/nimbus_sim
	python3 scripts/tests/science/test_milbrandt_accelerator_ready.py --binary ./bin/nimbus_sim

bin/test_numerics_native_rk_combination: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/numerics/test_native_rk_combination.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-rk-combination
test-native-rk-combination: bin/test_numerics_native_rk_combination
	python3 scripts/dev/shaders/generate_rk_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_rk_combination

bin/test_numerics_native_rk_tendency: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/numerics/test_native_rk_tendency.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-rk
test-native-rk: bin/nimbus_sim bin/test_numerics_native_rk_combination bin/test_numerics_native_rk_tendency bin/test_physics_native_collocated_sgs
	python3 scripts/dev/shaders/generate_tke_msl.py --kernel collocated_sgs --check
	python3 scripts/dev/shaders/generate_rk_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_rk_combination
	./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_rk_tendency
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_collocated_sgs
	python3 scripts/tests/science/test_coupled_rk_accelerator_ready.py --binary ./bin/nimbus_sim

bin/test_physics_native_collocated_sgs: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_collocated_sgs.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@


bin/test_core_native_conservative_remap: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/core/test_native_conservative_remap.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-conservative-remap
test-native-conservative-remap: bin/test_core_native_conservative_remap
	python3 scripts/dev/shaders/generate_remap_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./scripts/run_with_vulkan_env.sh ./bin/test_core_native_conservative_remap

bin/test_core_native_earth_geometry: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/core/test_native_earth_geometry.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-earth-geometry
test-native-earth-geometry: bin/test_core_native_earth_geometry
	python3 scripts/dev/shaders/generate_earth_geometry_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./scripts/run_with_vulkan_env.sh ./bin/test_core_native_earth_geometry

bin/test_dynamics_native_moving_momentum: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/dynamics/test_native_moving_momentum.cpp tests/dynamics/test_native_moving_boundary.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-moving-momentum
test-native-moving-momentum: bin/test_dynamics_native_moving_momentum
	python3 scripts/dev/shaders/generate_moving_momentum_msl.py --check
	python3 scripts/dev/shaders/generate_remap_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./scripts/run_with_vulkan_env.sh ./bin/test_dynamics_native_moving_momentum

bin/test_dynamics_native_moving_boundary: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/dynamics/test_native_moving_boundary.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-moving-boundary
test-native-moving-boundary: bin/test_dynamics_native_moving_boundary
	python3 scripts/dev/shaders/generate_moving_momentum_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./scripts/run_with_vulkan_env.sh ./bin/test_dynamics_native_moving_boundary

bin/test_dynamics_native_moving_energy: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/dynamics/test_native_moving_energy.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-moving-energy
test-native-moving-energy: bin/test_dynamics_native_moving_energy
	python3 scripts/dev/shaders/generate_moving_energy_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./scripts/run_with_vulkan_env.sh ./bin/test_dynamics_native_moving_energy

bin/test_dynamics_moving_environment: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/dynamics/test_moving_environment.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-moving-environment
test-moving-environment: bin/test_dynamics_moving_environment
	python3 scripts/dev/shaders/generate_reference_atmosphere_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./scripts/run_with_vulkan_env.sh ./bin/test_dynamics_moving_environment

bin/test_dynamics_native_cgrid_boundary: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/dynamics/test_native_cgrid_boundary.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-cgrid-boundary
test-native-cgrid-boundary: bin/test_dynamics_native_cgrid_boundary
	python3 scripts/dev/shaders/generate_mapped_boundary_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./scripts/run_with_vulkan_env.sh ./bin/test_dynamics_native_cgrid_boundary

bin/test_core_held_source_remap: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/core/test_held_source_remap.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-held-source-remap
test-held-source-remap: bin/test_core_held_source_remap
	python3 scripts/dev/shaders/generate_column_remap_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./scripts/run_with_vulkan_env.sh ./bin/test_core_held_source_remap
	NIMBUS_VULKAN_MEMORY_POLICY=staged ./scripts/run_with_vulkan_env.sh ./bin/test_core_held_source_remap '[vulkan]'

bin/test_physics_native_simple_grey: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_simple_grey.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_physics_native_slab: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_slab.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_core_environment_column_refresh: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/core/test_environment_column_refresh.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_core_native_held_sources: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/core/test_native_held_sources.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_dynamics_mapped_reconstruction: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/dynamics/test_mapped_reconstruction.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_dynamics_native_acoustic_boundary: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/dynamics/test_native_acoustic_boundary.cpp vulkan/shaders/compute/dynamics/boundary_float32.glsl $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_physics_native_kessler_layers: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_kessler_layers.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_dynamics_native_moist_mapped_acoustic: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/dynamics/test_native_moist_mapped_acoustic.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

NATIVE_EARTH_COLUMN_TEST_SRCS := \
	tests/physics/test_native_simple_grey.cpp \
	tests/physics/test_native_slab.cpp \
	tests/core/test_environment_column_refresh.cpp \
	tests/core/test_native_held_sources.cpp \
	tests/dynamics/test_native_acoustic_boundary.cpp \
	tests/dynamics/test_mapped_reconstruction.cpp \
	tests/dynamics/test_native_moist_mapped_acoustic.cpp \
	tests/physics/test_native_kessler_layers.cpp

bin/test_native_earth_fixed_column_physics: tests/test_main.cpp tests/support/production_wind_profile.cpp $(NATIVE_EARTH_COLUMN_TEST_SRCS) $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-earth-fixed-column-physics
test-native-earth-fixed-column-physics: bin/test_native_earth_fixed_column_physics
	python3 scripts/dev/shaders/generate_simple_grey_msl.py --check
	python3 scripts/dev/shaders/generate_slab_column_msl.py --check
	python3 scripts/dev/shaders/generate_held_sources_msl.py --check
	python3 scripts/dev/shaders/generate_acoustic_bc_face_cartesian_cgrid_msl.py --check
	python3 scripts/dev/shaders/generate_advect_conservative_scalar_flux_cartesian_cgrid_msl.py --check
	python3 scripts/dev/shaders/generate_kessler_sedimentation_msl.py --check
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	./scripts/run_with_vulkan_env.sh ./bin/test_native_earth_fixed_column_physics

bin/test_core_native_earth_fixed_step: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/core/test_native_earth_fixed_step.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

EARTH_NATIVE_EVIDENCE ?= /tmp/nimbus-native-earth-fixed-ready
.PHONY: test-native-earth-fixed-ready
test-native-earth-fixed-ready: bin/nimbus_sim
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_native_earth_fixed_ready.py --binary bin/nimbus_sim --output $(EARTH_NATIVE_EVIDENCE)/kessler
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_native_earth_fixed_ready.py --microphysics lin --binary bin/nimbus_sim --output $(EARTH_NATIVE_EVIDENCE)/lin

bin/test_physics_native_lin_layers: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/physics/test_native_lin_layers.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/test_core_nested_scalar_feedback: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/core/test_nested_scalar_feedback.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-moist-two-way-nested-ready
MOIST_NESTED_EVIDENCE ?= /tmp/nimbus-moist-two-way-evidence

test-moist-two-way-nested-ready: bin/nimbus_sim bin/test_core_nested_scalar_feedback bin/test_core_nested_grid bin/test_core_simulation_checkpoint bin/test_numerics_advection_cartesian
	./bin/test_core_nested_scalar_feedback
	./bin/test_core_nested_grid "[uniform-wind],[two_way]"
	./bin/test_core_simulation_checkpoint "[checkpoint][nesting]"
	./bin/test_numerics_advection_cartesian "[interface][moist]"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_moist_two_way_nested_ready.py --binary ./bin/nimbus_sim --output "$(MOIST_NESTED_EVIDENCE)"

.PHONY: test-two-way-recycling-ready
TWO_WAY_RECYCLING_EVIDENCE ?= /tmp/nimbus-two-way-recycling-evidence

test-two-way-recycling-ready: bin/nimbus_sim bin/test_core_nested_scalar_feedback bin/test_core_nested_grid bin/test_core_domain_recycling bin/test_core_simulation_checkpoint
	./bin/test_core_nested_scalar_feedback
	./bin/test_core_nested_grid "[two_way]"
	./bin/test_core_domain_recycling "[nested]"
	./bin/test_core_simulation_checkpoint "[checkpoint][nesting]"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_nested_grid_recycling_ready.py --binary ./bin/nimbus_sim --config tests/fixtures/configs/qualification/nested_grid_recycling_two_way_cpu_ready.yaml --output "$(TWO_WAY_RECYCLING_EVIDENCE)"

.PHONY: test-moving-two-way-nested-ready
MOVING_TWO_WAY_EVIDENCE ?= /tmp/nimbus-moving-two-way-evidence

test-moving-two-way-nested-ready: bin/nimbus_sim bin/test_core_nested_production_step bin/test_diagnostics_energy_budget bin/test_core_simulation_checkpoint
	./bin/test_core_nested_production_step "[movement]"
	./bin/test_diagnostics_energy_budget
	./bin/test_core_simulation_checkpoint "[checkpoint][nesting]"
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/science/test_moving_two_way_nested_ready.py --binary ./bin/nimbus_sim --output "$(MOVING_TWO_WAY_EVIDENCE)"


bin/test_numerics_native_scalar_capture: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/numerics/test_native_scalar_capture.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-scalar-capture
test-native-scalar-capture: bin/test_numerics_native_scalar_capture
	./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_scalar_capture
	NIMBUS_VULKAN_MEMORY_POLICY=staged ./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_scalar_capture '[vulkan]'

bin/test_numerics_native_conservative_momentum: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/numerics/test_native_conservative_momentum.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

NATIVE_TWO_WAY_EVIDENCE ?= /tmp/nimbus-native-two-way-ready
.PHONY: test-native-two-way-ready
test-native-two-way-ready: bin/test_core_nested_grid bin/test_core_simulation_checkpoint bin/nimbus_sim bin/test_numerics_native_conservative_momentum bin/test_numerics_native_scalar_capture bin/test_core_nested_scalar_feedback bin/test_numerics_native_phase_feedback bin/test_physics_boundary_layer bin/test_physics_native_slab
	./bin/test_core_nested_grid "[uniform-wind],[two_way]"
	./bin/test_core_simulation_checkpoint "[checkpoint][nesting]"
	./bin/test_core_nested_scalar_feedback
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	python3 scripts/dev/shaders/generate_conservative_momentum_msl.py --check
	python3 scripts/dev/shaders/generate_conservative_reduction_msl.py --check
	python3 scripts/dev/shaders/generate_slab_column_msl.py --check
	./bin/test_physics_boundary_layer
	./bin/test_numerics_native_phase_feedback
	./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_slab
	./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_conservative_momentum
	./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_scalar_capture
	NIMBUS_SCALAR_FEEDBACK_BACKEND=metal ./bin/test_core_nested_scalar_feedback
	NIMBUS_SCALAR_FEEDBACK_BACKEND=vulkan ./scripts/run_with_vulkan_env.sh ./bin/test_core_nested_scalar_feedback
	NIMBUS_NESTED_FEEDBACK_BACKEND=metal ./bin/test_numerics_native_phase_feedback
	NIMBUS_NESTED_FEEDBACK_BACKEND=vulkan ./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_phase_feedback
	NIMBUS_VULKAN_MEMORY_POLICY=staged ./scripts/run_with_vulkan_env.sh ./bin/test_physics_native_slab '[vulkan]'
	NIMBUS_VULKAN_MEMORY_POLICY=staged ./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_conservative_momentum '[vulkan]'
	NIMBUS_VULKAN_MEMORY_POLICY=staged ./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_scalar_capture '[vulkan]'
	NIMBUS_SCALAR_FEEDBACK_BACKEND=vulkan NIMBUS_VULKAN_MEMORY_POLICY=staged ./scripts/run_with_vulkan_env.sh ./bin/test_core_nested_scalar_feedback
	NIMBUS_NESTED_FEEDBACK_BACKEND=vulkan NIMBUS_VULKAN_MEMORY_POLICY=staged ./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_phase_feedback
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_native_two_way_nested_ready.py --advection weno --binary bin/nimbus_sim --output $(NATIVE_TWO_WAY_EVIDENCE)/weno-stationary
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_native_two_way_nested_ready.py --advection tvd --binary bin/nimbus_sim --output $(NATIVE_TWO_WAY_EVIDENCE)/tvd-stationary
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_native_nested_motion_ready.py --advection weno --binary bin/nimbus_sim --output $(NATIVE_TWO_WAY_EVIDENCE)/weno-motion
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_native_nested_motion_ready.py --advection tvd --binary bin/nimbus_sim --output $(NATIVE_TWO_WAY_EVIDENCE)/tvd-motion


bin/test_numerics_native_phase_feedback: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/numerics/test_native_phase_feedback.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

.PHONY: test-native-mapped-two-way-ready
NATIVE_MAPPED_TWO_WAY_EVIDENCE ?= /tmp/nimbus-native-mapped-two-way-ready

test-native-mapped-two-way-ready: bin/nimbus_sim bin/test_init_config_v1 bin/test_core_scheme_catalog bin/test_core_simulation_checkpoint bin/test_numerics_native_conservative_momentum bin/test_dynamics_mapped_reconstruction bin/test_dynamics_native_moist_mapped_acoustic
	python3 scripts/dev/shaders/verify_spirv.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py
	python3 scripts/dev/shaders/generate_conservative_momentum_msl.py --check
	./bin/test_init_config_v1
	./bin/test_core_scheme_catalog
	./bin/test_core_simulation_checkpoint '[checkpoint][nesting]'
	./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_conservative_momentum
	./scripts/run_with_vulkan_env.sh ./bin/test_dynamics_mapped_reconstruction
	./scripts/run_with_vulkan_env.sh ./bin/test_dynamics_native_moist_mapped_acoustic
	NIMBUS_VULKAN_MEMORY_POLICY=staged ./scripts/run_with_vulkan_env.sh ./bin/test_numerics_native_conservative_momentum '[vulkan]'
	NIMBUS_VULKAN_MEMORY_POLICY=staged ./scripts/run_with_vulkan_env.sh ./bin/test_dynamics_mapped_reconstruction '[vulkan]'
	NIMBUS_VULKAN_MEMORY_POLICY=staged ./scripts/run_with_vulkan_env.sh ./bin/test_dynamics_native_moist_mapped_acoustic '[vulkan]'
	@for reconstruction in tvd weno; do \
	  for terrain in bell schar bell3 schar3; do \
	    ./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_native_two_way_nested_ready.py \
	      --binary bin/nimbus_sim --advection "$$reconstruction" --terrain "$$terrain" \
	      --output "$(NATIVE_MAPPED_TWO_WAY_EVIDENCE)/$$reconstruction-$$terrain" || exit $$?; \
	  done; \
	done

.PHONY: test-thompson-two-way-ready
.PHONY: test-unseeded-private-two-way-ready

test-unseeded-private-two-way-ready: bin/nimbus_sim bin/nimbus_private_checkpoint_probe
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_unseeded_private_two_way_ready.py --binary ./bin/nimbus_sim

THOMPSON_TWO_WAY_EVIDENCE ?= /tmp/nimbus-thompson-two-way-ready

test-thompson-two-way-ready: bin/nimbus_sim bin/nimbus_private_checkpoint_probe
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/science/test_thompson_two_way_ready.py \
	  --binary bin/nimbus_sim --output $(THOMPSON_TWO_WAY_EVIDENCE)

.PHONY: test-moving-mapped-two-way-ready
test-moving-mapped-two-way-ready: bin/nimbus_sim bin/test_init_config_v1 bin/test_core_nested_terrain_geometry bin/test_core_nested_grid bin/test_core_nested_production_step bin/test_core_simulation_checkpoint bin/test_core_output bin/test_diagnostics_pressure_projection bin/test_diagnostics_resolved bin/test_dynamics_cartesian bin/test_diagnostics_trajectories bin/test_diagnostics_potential_vorticity
	python3 scripts/qualification/nested_domains/moving_mapped_two_way/run_operators.py --output /tmp/nimbus-moving-mapped-operators
	./scripts/run_with_vulkan_env.sh python3 scripts/qualification/nested_domains/moving_mapped_two_way/run_bounded_evidence.py --output /tmp/nimbus-moving-mapped-families
	./scripts/run_with_vulkan_env.sh python3 scripts/qualification/nested_domains/moving_mapped_two_way/run_regressions.py --output /tmp/nimbus-moving-mapped-regressions

bin/test_diagnostics_pressure_projection: tests/test_main.cpp tests/diagnostics/test_pressure_projection.cpp src/diagnostics/derived/pressure_projection.cpp $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(LDLIBS) -o $@

bin/test_diagnostics_resolved: tests/test_main.cpp tests/support/production_wind_profile.cpp tests/diagnostics/test_resolved_diagnostics.cpp $(NESTED_PRODUCTION_TEST_OBJS) $(TEST_FLAGS_STAMP) | bin
	$(CXX_COMPILE) $(TEST_CXXFLAGS) $(CPPFLAGS) $(filter %.cpp,$^) $(filter %.o,$^) $(LDLIBS) $(METAL_LDLIBS) -o $@

$(filter bin/test_physics_milbrandt_%,$(CATCH2_BINS)) bin/test_physics_native_milbrandt_cold: src/microphysics/schemes/milbrandt/cold.hpp src/microphysics/schemes/milbrandt/cold_snow_riming_admission.hpp include/microphysics/milbrandt_cold_parameters.hpp


bin/test_physics_milbrandt_triple_frozen_collection: src/microphysics/schemes/milbrandt/triple_rain_collection.hpp

bin/test_physics_milbrandt_triple_frozen_collection: src/microphysics/schemes/milbrandt/triple_rain_collision.hpp

bin/test_physics_milbrandt_triple_frozen_collection: src/microphysics/schemes/milbrandt/triple_hail_wet_growth.hpp src/microphysics/schemes/milbrandt/triple_cold_state.hpp
