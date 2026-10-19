# ---------------------------------------------------------------------------
# Main binary (incremental object-file build)
# ---------------------------------------------------------------------------
ifeq ($(CHECKED),1)
  CHECKED_SUFFIX := _checked
  MAIN_ASSERT_FLAGS :=
else
  CHECKED_SUFFIX :=
  MAIN_ASSERT_FLAGS := -DNDEBUG
endif
ifeq ($(LTO),1)
  PROFILE_SUFFIX := $(CHECKED_SUFFIX)_lto
else
  PROFILE_SUFFIX := $(CHECKED_SUFFIX)
endif
ifeq ($(PORTABLE),1)
  PROFILE_SUFFIX := $(PROFILE_SUFFIX)_portable
endif
ifneq ($(SANITIZE),none)
  PROFILE_SUFFIX := $(PROFILE_SUFFIX)_$(SANITIZE)
endif
ifeq ($(CPU_ONLY),1)
  BIN := bin/nimbus_sim_cpu$(PROFILE_SUFFIX)$(EXEEXT)
  BUILDDIR := build/cpu_only$(PROFILE_SUFFIX)
  ifeq ($(PORTABLE),1)
    BENCHMARK_BIN := bin/nimbus_sim_cpu_only_portable_benchmark
    BENCHMARK_BUILDDIR := build/benchmark_cpu_only_portable
  else
    BENCHMARK_BIN := bin/nimbus_sim_cpu_only_benchmark
    BENCHMARK_BUILDDIR := build/benchmark_cpu_only
  endif
else
  BIN := bin/nimbus_sim$(PROFILE_SUFFIX)$(EXEEXT)
  ifeq ($(PROFILE_SUFFIX),)
    BUILDDIR := build/release
  else
    BUILDDIR := build/$(patsubst _%,%,$(PROFILE_SUFFIX))
  endif
  ifeq ($(PORTABLE),1)
    BENCHMARK_BIN := bin/nimbus_sim_cpu_portable_benchmark
    BENCHMARK_BUILDDIR := build/benchmark_cpu_portable
  else
    BENCHMARK_BIN := bin/nimbus_sim_cpu_benchmark
    BENCHMARK_BUILDDIR := build/benchmark_cpu
  endif
endif
ifeq ($(PROFILE_SUFFIX),)
  FIELD_VALIDATOR := bin/field_validator$(EXEEXT)
else
  FIELD_VALIDATOR := bin/field_validator$(PROFILE_SUFFIX)$(EXEEXT)
endif
MAIN_FLAGS_STAMP := $(BUILDDIR)/.build-flags
TEST_FLAGS_STAMP := build/tests/.build-flags
# A dedicated release-like artifact prevents benchmark runs from silently
# reusing the ordinary build's objects or assertion policy.
# Fold CPPFLAGS into the stamped compile identity, then place the enforced
# release flags last so command-line CPPFLAGS cannot silently downgrade the
# benchmark to -O0 or re-enable assertions.
BENCHMARK_CXXFLAGS := $(filter-out -O% -DNDEBUG,$(CXXFLAGS) $(CPPFLAGS)) -O3 -DNDEBUG
BENCHMARK_FLAGS_STAMP := $(BENCHMARK_BUILDDIR)/.build-flags

# SCI-173 controlled allocation-lifetime A/B. Both artifacts reuse the same
# release-like object set except for the translation units that own the
# measurement window or repaired scratch. The comparator defines only a
# replaceable-scratch lifetime experiment; it cannot change equations or
# committed/checkpoint state.
ACTIVE_PHYSICS_OPT_BIN := bin/nimbus_sim_active_physics_allocations
NESTED_RELOCATION_BENCHMARK_BIN := bin/benchmark_nested_relocation
ACTIVE_PHYSICS_LEGACY_BIN := bin/nimbus_sim_active_physics_allocations_legacy
SMAG_FILTER_LEGACY_BIN := bin/nimbus_sim_smagorinsky_filter_legacy
SGS_VERTICAL_STRESS_LEGACY_BIN := \
	bin/nimbus_sim_sgs_vertical_stress_legacy
ZFP_HISTORY_COMPARATOR_BIN := bin/nimbus_sim_zfp_history_comparator
ZFP_BRICK_GATHER_COMPARATOR_BIN := bin/nimbus_sim_zfp_brick_gather_comparator
ZFP_TRANSIENT_WORKSPACE_COMPARATOR_BIN := \
	bin/nimbus_sim_zfp_transient_workspace_comparator
ZFP_WORKSPACE_PHASE_BIN := bin/nimbus_sim_zfp_workspace_phase
ZFP_WORKSPACE_CROSS_READER_FIXTURE := \
	build/qualification/zfp-codec-workspace/writer-keyframe.zfp3d
VULKAN_SCALAR_ATTRIBUTION_COMPARATOR_BIN := \
	bin/nimbus_sim_vulkan_scalar_attribution_comparator
VULKAN_SCALAR_REGIONS_EXPERIMENT_BIN := \
	bin/nimbus_sim_vulkan_scalar_regions_experiment
VULKAN_NOOP_BUDGET_COMPARATOR_BIN := \
	bin/nimbus_sim_vulkan_noop_budget_comparator
VULKAN_ENDPOINT_MOISTURE_COMPARATOR_BIN := \
	bin/nimbus_sim_vulkan_endpoint_moisture_comparator
VULKAN_MICROPHYSICS_GATE_COMPARATOR_BIN := \
	bin/nimbus_sim_vulkan_microphysics_gate_comparator
VULKAN_MICROPHYSICS_VALIDATION_COMPARATOR_BIN := \
	bin/nimbus_sim_vulkan_microphysics_validation_comparator
VULKAN_MICROPHYSICS_INPUT_VALIDATION_COMPARATOR_BIN := \
	bin/nimbus_sim_vulkan_microphysics_input_validation_comparator
ACTIVE_PHYSICS_OPT_BUILDDIR := build/benchmark_active_physics
NESTED_RELOCATION_BENCHMARK_BUILDDIR := \
	build/benchmark_nested_relocation
ACTIVE_PHYSICS_LEGACY_BUILDDIR := build/benchmark_active_physics_legacy
SMAG_FILTER_LEGACY_BUILDDIR := build/benchmark_smagorinsky_filter_legacy
SGS_VERTICAL_STRESS_LEGACY_BUILDDIR := \
	build/benchmark_sgs_vertical_stress_legacy
ZFP_HISTORY_COMPARATOR_BUILDDIR := build/benchmark_zfp_history_comparator
ZFP_BRICK_GATHER_COMPARATOR_BUILDDIR := \
	build/benchmark_zfp_brick_gather_comparator
ZFP_TRANSIENT_WORKSPACE_COMPARATOR_BUILDDIR := \
	build/benchmark_zfp_transient_workspace_phase_comparator
ZFP_WORKSPACE_PHASE_BUILDDIR := build/benchmark_zfp_workspace_phase
VULKAN_SCALAR_ATTRIBUTION_COMPARATOR_BUILDDIR := \
	build/benchmark_vulkan_scalar_attribution_comparator
VULKAN_SCALAR_REGIONS_EXPERIMENT_BUILDDIR := \
	build/benchmark_vulkan_scalar_regions_experiment
VULKAN_NOOP_BUDGET_COMPARATOR_BUILDDIR := \
	build/benchmark_vulkan_noop_budget_comparator
VULKAN_ENDPOINT_MOISTURE_COMPARATOR_BUILDDIR := \
	build/benchmark_vulkan_endpoint_moisture_comparator
VULKAN_MICROPHYSICS_GATE_COMPARATOR_BUILDDIR := \
	build/benchmark_vulkan_microphysics_gate_comparator
VULKAN_MICROPHYSICS_VALIDATION_COMPARATOR_BUILDDIR := \
	build/benchmark_vulkan_microphysics_validation_comparator
VULKAN_MICROPHYSICS_INPUT_VALIDATION_COMPARATOR_BUILDDIR := \
	build/benchmark_vulkan_microphysics_input_validation_comparator
SMAG_FILTER_SRC := src/turbulence/schemes/smagorinsky/smagorinsky.cpp
SGS_VERTICAL_STRESS_SRC := src/turbulence/base/eddy_viscosity.cpp
ZFP_HISTORY_SRC := src/core/output/output_writer.cpp
VULKAN_SCALAR_ATTRIBUTION_SRC := \
	src/numerics/advection/advection_cartesian.cpp
VULKAN_NOOP_BUDGET_SRC := src/core/orchestration/dynamics/dynamics.cpp
VULKAN_MICROPHYSICS_GATE_SRC := \
	src/core/orchestration/physics/microphysics_step.cpp
ACTIVE_PHYSICS_SPECIAL_SRCS := \
	src/core/runtime/headless_runtime.cpp \
	src/core/orchestration/dynamics/dynamics.cpp \
	src/core/orchestration/physics/microphysics_step.cpp \
	src/radiation/runtime.cpp \
	src/boundary_layer/runtime.cpp \
	src/microphysics/schemes/milbrandt/milbrandt.cpp

# Object groups mirror the dependency layers declared by the source manifest.
CORE_LAYER_OBJS := $(patsubst %.cpp,$(BUILDDIR)/%.o,$(CORE_INFRA_SRCS) \
                   $(CORE_OUTPUT_SRCS) $(CORE_RUNTIME_SRCS) \
                   $(CORE_ORCHESTRATION_SRCS))
COMPUTE_OBJS := $(patsubst %.cpp,$(BUILDDIR)/%.o,$(COMPUTE_SRCS))
MODULE_OBJS := $(patsubst %.cpp,$(BUILDDIR)/%.o,$(BOUNDARY_CONDITION_SRCS) \
               $(DYNAMICS_SCHEME_SRCS) $(NUMERICS_SRCS) \
               $(MICROPHYSICS_SRCS) $(RADIATION_SRCS) \
               $(BOUNDARY_LAYER_SRCS) $(TURBULENCE_SRCS) $(CHAOS_SRCS) \
               $(TERRAIN_SRCS) $(TERRAIN_NATIVE_SRCS) $(TERRAIN_COLLOCATED_SRCS) $(RADAR_SRCS) $(SOUNDING_READER_SRCS) \
               $(INIT_SRCS) $(DIAGNOSTICS_SRCS))
VULKAN_BACKEND_OBJS := $(patsubst %.cpp,$(BUILDDIR)/%.o,$(VULKAN_BACKEND_SRCS))
METAL_BACKEND_OBJS :=
ifneq ($(METAL_MAIN_FLAGS),)
METAL_BACKEND_OBJS := $(BUILDDIR)/metal/src/compute/metal_compute_backend.o
endif
OBJS := $(CORE_LAYER_OBJS) $(COMPUTE_OBJS) $(MODULE_OBJS) \
        $(VULKAN_BACKEND_OBJS) $(METAL_BACKEND_OBJS)
FIELD_VALIDATOR_OBJS := $(patsubst %.cpp,$(BUILDDIR)/%.o,$(FIELD_VALIDATOR_SRCS))
BENCHMARK_OBJS := $(patsubst %.cpp,$(BENCHMARK_BUILDDIR)/%.o,$(SRCS))
ACTIVE_PHYSICS_REPLACED_OBJS := \
	$(patsubst %.cpp,$(BENCHMARK_BUILDDIR)/%.o,$(ACTIVE_PHYSICS_SPECIAL_SRCS))
ACTIVE_PHYSICS_BASE_OBJS := \
	$(filter-out $(ACTIVE_PHYSICS_REPLACED_OBJS),$(BENCHMARK_OBJS))
ACTIVE_PHYSICS_OPT_SPECIAL_OBJS := \
	$(patsubst %.cpp,$(ACTIVE_PHYSICS_OPT_BUILDDIR)/%.o,$(ACTIVE_PHYSICS_SPECIAL_SRCS))
ACTIVE_PHYSICS_LEGACY_SPECIAL_OBJS := \
	$(patsubst %.cpp,$(ACTIVE_PHYSICS_LEGACY_BUILDDIR)/%.o,$(ACTIVE_PHYSICS_SPECIAL_SRCS))
ACTIVE_PHYSICS_TELEMETRY_OBJ := \
	$(ACTIVE_PHYSICS_OPT_BUILDDIR)/src/core/infra/allocation_telemetry.o
NESTED_RELOCATION_BENCHMARK_TEST_OBJS := \
	$(NESTED_RELOCATION_BENCHMARK_BUILDDIR)/tests/test_main.o \
	$(NESTED_RELOCATION_BENCHMARK_BUILDDIR)/tests/core/test_nested_production_step.o
NESTED_RELOCATION_BENCHMARK_BASE_OBJS := \
	$(filter-out $(BENCHMARK_BUILDDIR)/src/core/runtime/nimbus_sim.o,\
		$(BENCHMARK_OBJS))
SMAG_FILTER_BASE_OBJS := \
	$(filter-out $(BENCHMARK_BUILDDIR)/$(SMAG_FILTER_SRC:.cpp=.o),$(ACTIVE_PHYSICS_BASE_OBJS))
SMAG_FILTER_LEGACY_OBJ := \
	$(SMAG_FILTER_LEGACY_BUILDDIR)/$(SMAG_FILTER_SRC:.cpp=.o)
SGS_VERTICAL_STRESS_BASE_OBJS := \
	$(filter-out \
	  $(BENCHMARK_BUILDDIR)/$(SGS_VERTICAL_STRESS_SRC:.cpp=.o),\
	  $(ACTIVE_PHYSICS_BASE_OBJS))
SGS_VERTICAL_STRESS_LEGACY_OBJ := \
	$(SGS_VERTICAL_STRESS_LEGACY_BUILDDIR)/$(SGS_VERTICAL_STRESS_SRC:.cpp=.o)
ZFP_HISTORY_BASE_OBJS := \
	$(filter-out \
	  $(BENCHMARK_BUILDDIR)/$(ZFP_HISTORY_SRC:.cpp=.o),\
	  $(BENCHMARK_OBJS))
ZFP_HISTORY_COMPARATOR_OBJ := \
	$(ZFP_HISTORY_COMPARATOR_BUILDDIR)/$(ZFP_HISTORY_SRC:.cpp=.o)
ZFP_BRICK_GATHER_COMPARATOR_OBJ := \
	$(ZFP_BRICK_GATHER_COMPARATOR_BUILDDIR)/$(ZFP_HISTORY_SRC:.cpp=.o)
ZFP_TRANSIENT_WORKSPACE_COMPARATOR_OBJ := \
	$(ZFP_TRANSIENT_WORKSPACE_COMPARATOR_BUILDDIR)/$(ZFP_HISTORY_SRC:.cpp=.o)
ZFP_WORKSPACE_PHASE_OBJ := \
	$(ZFP_WORKSPACE_PHASE_BUILDDIR)/$(ZFP_HISTORY_SRC:.cpp=.o)
VULKAN_SCALAR_ATTRIBUTION_BASE_OBJS := \
	$(filter-out \
	  $(BENCHMARK_BUILDDIR)/$(VULKAN_SCALAR_ATTRIBUTION_SRC:.cpp=.o),\
	  $(BENCHMARK_OBJS))
VULKAN_SCALAR_ATTRIBUTION_COMPARATOR_OBJ := \
	$(VULKAN_SCALAR_ATTRIBUTION_COMPARATOR_BUILDDIR)/$(VULKAN_SCALAR_ATTRIBUTION_SRC:.cpp=.o)
VULKAN_SCALAR_REGIONS_EXPERIMENT_OBJ := \
	$(VULKAN_SCALAR_REGIONS_EXPERIMENT_BUILDDIR)/$(VULKAN_SCALAR_ATTRIBUTION_SRC:.cpp=.o)
VULKAN_NOOP_BUDGET_BASE_OBJS := \
	$(filter-out \
	  $(BENCHMARK_BUILDDIR)/$(VULKAN_NOOP_BUDGET_SRC:.cpp=.o),\
	  $(BENCHMARK_OBJS))
VULKAN_NOOP_BUDGET_COMPARATOR_OBJ := \
	$(VULKAN_NOOP_BUDGET_COMPARATOR_BUILDDIR)/$(VULKAN_NOOP_BUDGET_SRC:.cpp=.o)
VULKAN_ENDPOINT_MOISTURE_COMPARATOR_OBJ := \
	$(VULKAN_ENDPOINT_MOISTURE_COMPARATOR_BUILDDIR)/$(VULKAN_NOOP_BUDGET_SRC:.cpp=.o)
VULKAN_MICROPHYSICS_GATE_BASE_OBJS := \
	$(filter-out \
	  $(BENCHMARK_BUILDDIR)/$(VULKAN_MICROPHYSICS_GATE_SRC:.cpp=.o),\
	  $(BENCHMARK_OBJS))
VULKAN_MICROPHYSICS_GATE_COMPARATOR_OBJ := \
	$(VULKAN_MICROPHYSICS_GATE_COMPARATOR_BUILDDIR)/$(VULKAN_MICROPHYSICS_GATE_SRC:.cpp=.o)
VULKAN_MICROPHYSICS_VALIDATION_COMPARATOR_OBJ := \
	$(VULKAN_MICROPHYSICS_VALIDATION_COMPARATOR_BUILDDIR)/$(VULKAN_MICROPHYSICS_GATE_SRC:.cpp=.o)
VULKAN_MICROPHYSICS_INPUT_VALIDATION_COMPARATOR_OBJ := \
	$(VULKAN_MICROPHYSICS_INPUT_VALIDATION_COMPARATOR_BUILDDIR)/$(VULKAN_MICROPHYSICS_GATE_SRC:.cpp=.o)

# Auto-dependency tracking (-MMD -MP generates .d files alongside .o files)
DEPFLAGS = -MMD -MP
DEPS := $(OBJS:.o=.d) $(FIELD_VALIDATOR_OBJS:.o=.d) $(BENCHMARK_OBJS:.o=.d) \
	$(ACTIVE_PHYSICS_OPT_SPECIAL_OBJS:.o=.d) \
	$(ACTIVE_PHYSICS_LEGACY_SPECIAL_OBJS:.o=.d) \
	$(SMAG_FILTER_LEGACY_OBJ:.o=.d) \
	$(SGS_VERTICAL_STRESS_LEGACY_OBJ:.o=.d) \
	$(ZFP_HISTORY_COMPARATOR_OBJ:.o=.d) \
	$(ZFP_BRICK_GATHER_COMPARATOR_OBJ:.o=.d) \
	$(ZFP_TRANSIENT_WORKSPACE_COMPARATOR_OBJ:.o=.d) \
	$(ZFP_WORKSPACE_PHASE_OBJ:.o=.d) \
	$(VULKAN_SCALAR_ATTRIBUTION_COMPARATOR_OBJ:.o=.d) \
	$(VULKAN_SCALAR_REGIONS_EXPERIMENT_OBJ:.o=.d) \
	$(VULKAN_NOOP_BUDGET_COMPARATOR_OBJ:.o=.d) \
	$(VULKAN_ENDPOINT_MOISTURE_COMPARATOR_OBJ:.o=.d) \
	$(VULKAN_MICROPHYSICS_GATE_COMPARATOR_OBJ:.o=.d) \
	$(VULKAN_MICROPHYSICS_VALIDATION_COMPARATOR_OBJ:.o=.d) \
	$(VULKAN_MICROPHYSICS_INPUT_VALIDATION_COMPARATOR_OBJ:.o=.d) \
	$(NESTED_RELOCATION_BENCHMARK_TEST_OBJS:.o=.d)

# Ordinary builds preserve the reviewed shader bytes and compiled identities.
.DEFAULT_GOAL := all
DEFAULT_ARTIFACTS := $(BIN) $(FIELD_VALIDATOR)
ifeq ($(CPU_ONLY),1)
all: $(DEFAULT_ARTIFACTS)
else
all: check-vulkan-assets
	$(MAKE) $(DEFAULT_ARTIFACTS)
endif
-include $(DEPS)

ifeq ($(WINDOWS_HOST),1)
ifneq ($(strip $(NETCDF_LDLIBS)),)
.PHONY: check-windows-netcdf-runtime
check-windows-netcdf-runtime:
	python3 scripts/windows/ensure_netcdf_runtime.py --check
all: check-windows-netcdf-runtime
$(BIN) $(FIELD_VALIDATOR): | check-windows-netcdf-runtime
endif
endif

# Compile each .cpp to a .o (incremental, parallelizable with -j)
$(BUILDDIR)/%.o: %.cpp $(MAIN_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(CXXFLAGS) $(METAL_MAIN_FLAGS) $(MAIN_ASSERT_FLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILDDIR)/%.o: %.mm $(MAIN_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(METAL_CXX) $(CXXFLAGS) $(METAL_MAIN_FLAGS) $(MAIN_ASSERT_FLAGS) $(CPPFLAGS) \
		$(DEPFLAGS) -fobjc-arc -c $< -o $@

# Link
$(BIN): $(OBJS) $(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(OBJS) $(LDFLAGS) $(LDLIBS) $(METAL_LDLIBS) -o $(BIN)

# Internal A-grid qualification artifact. It reuses every production object
# except the CLI composition root, whose separately compiled internal view can
# seed qualification-only state. Runtime configuration and science preflight
# are byte-identical to the ordinary simulator.
AGRID_QUAL_MAIN_OBJ := build/qualification/src/core/runtime/nimbus_sim.o
AGRID_QUAL_OBJS := $(filter-out $(BUILDDIR)/src/core/runtime/nimbus_sim.o,$(OBJS)) \
	$(AGRID_QUAL_MAIN_OBJ)
-include $(AGRID_QUAL_MAIN_OBJ:.o=.d)

$(AGRID_QUAL_MAIN_OBJ): src/core/runtime/nimbus_sim.cpp $(MAIN_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(CXXFLAGS) $(MAIN_ASSERT_FLAGS) $(CPPFLAGS) \
		-DNIMBUS_INTERNAL_AGRID_QUALIFICATION $(DEPFLAGS) -c $< -o $@

bin/nimbus_sim_agrid_qualification: $(AGRID_QUAL_OBJS) $(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(AGRID_QUAL_OBJS) $(LDFLAGS) $(LDLIBS) $(METAL_LDLIBS) -o $@

bin/nimbus_sim_internal_qualification: $(AGRID_QUAL_OBJS) $(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(AGRID_QUAL_OBJS) $(LDFLAGS) $(LDLIBS) $(METAL_LDLIBS) -o $@

MOVING_NESTED_METAL_REFERENCE_SMOKE_OUTPUT ?= \
	/private/tmp/nimbus-moving-nested-metal-reference-smoke-v1

.PHONY: qualify-moving-nested-supercell-pilot qualify-moving-nested-supercell-live-pilot qualify-moving-nested-supercell-restart qualify-moving-nested-supercell-changed-thread qualify-moving-nested-supercell-lifetime qualify-moving-nested-reference-frontier-validate qualify-moving-nested-reference-frontier test-moving-nested-vulkan-admission qualify-moving-nested-vulkan-public-admission qualify-moving-nested-metal-public-admission qualify-moving-nested-vulkan-runtime inspect-moving-nested-vulkan-runtime qualify-moving-nested-metal-runtime-smoke qualify-moving-nested-metal-runtime inspect-moving-nested-metal-runtime qualify-moving-nested-metal-lifetime inspect-moving-nested-metal-lifetime inspect-moving-nested-metal-lifetime-chain qualify-moving-nested-metal-reference-smoke qualify-moving-nested-metal-reference inspect-moving-nested-metal-reference seal-moving-nested-metal-reference-evidence ratify-moving-nested-metal-reference-policy qualify-moving-nested-metal-reference-frontier seal-moving-nested-metal-reference-frontier-evidence test-moving-nested-vulkan-continuation diagnose-moving-nested-vulkan-continuation test-moving-nested-vulkan-lifetime-chain inspect-moving-nested-vulkan-lifetime-chain diagnose-moving-nested-vulkan-lifetime qualify-moving-nested-vulkan-reference inspect-moving-nested-vulkan-reference qualify-moving-nested-vulkan-reference-frontier qualify-moving-nested-vulkan-relocation-cost
qualify-moving-nested-supercell-pilot: bin/nimbus_sim_internal_qualification bin/nimbus_sim
	python3 scripts/qualification/nested_domains/qualify_moving_nested_supercell.py --duration 30

qualify-moving-nested-supercell-live-pilot: bin/nimbus_sim_internal_qualification bin/nimbus_sim
	python3 scripts/qualification/nested_domains/qualify_moving_nested_supercell.py --duration 1350 --write-every 450 --require-movement

qualify-moving-nested-supercell-restart: bin/nimbus_sim_internal_qualification bin/nimbus_sim
	python3 scripts/qualification/nested_domains/qualify_moving_nested_supercell_restart.py --final-time 1350

qualify-moving-nested-supercell-changed-thread: bin/nimbus_sim_internal_qualification bin/nimbus_sim
	python3 scripts/qualification/nested_domains/qualify_moving_nested_supercell_restart.py --final-time 1350 --continuous-threads 12 --resume-threads 4

qualify-moving-nested-supercell-lifetime: bin/nimbus_sim_internal_qualification bin/nimbus_sim
	python3 scripts/qualification/nested_domains/qualify_moving_nested_supercell.py --duration 4500 --require-movement

qualify-moving-nested-reference-frontier-validate: bin/nimbus_sim_internal_qualification
	python3 scripts/qualification/nested_domains/run_moving_nested_reference_frontier.py --validate-only

qualify-moving-nested-reference-frontier: bin/nimbus_sim_internal_qualification
	@test -n "$(MOVING_REFERENCE_FRONTIER_OUTPUT)" || \
	  { echo "Set MOVING_REFERENCE_FRONTIER_OUTPUT to a fresh campaign directory"; exit 2; }
	python3 scripts/qualification/nested_domains/run_moving_nested_reference_frontier.py \
	  --output "$(MOVING_REFERENCE_FRONTIER_OUTPUT)"

test-moving-nested-vulkan-admission: bin/nimbus_sim_internal_qualification bin/nimbus_sim
	python3 scripts/tests/gpu/test_moving_nested_vulkan_admission.py

qualify-moving-nested-vulkan-public-admission: bin/nimbus_sim_internal_qualification bin/nimbus_sim
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/gpu/test_moving_nested_vulkan_admission.py \
	  --public-binary $(PHYSICAL_QUALIFICATION_BINARY) \
	  --require-vulkan-positive \
	  --report reports/pre_beta/physical/moving_nested_public_vulkan_admission_v1_pass.json

qualify-moving-nested-metal-public-admission: bin/nimbus_sim_internal_qualification bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/gpu/test_moving_nested_vulkan_admission.py \
	  --public-binary $(PHYSICAL_QUALIFICATION_BINARY) \
	  --require-metal-positive

qualify-moving-nested-vulkan-runtime: bin/nimbus_sim_internal_qualification $(FIELD_VALIDATOR)
	runtime_root=$$(mktemp -d /private/tmp/nimbus-moving-nested-vulkan.XXXXXX); \
	  ./scripts/run_with_vulkan_env.sh python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_runtime.py \
	    --output "$$runtime_root/run"

qualify-moving-nested-vulkan-public-runtime: bin/nimbus_sim $(FIELD_VALIDATOR)
	runtime_root=$$(mktemp -d /private/tmp/nimbus-moving-nested-vulkan-public.XXXXXX); \
	  ./scripts/run_with_vulkan_env.sh python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_runtime.py \
	    --binary $(PHYSICAL_QUALIFICATION_BINARY) --public-runtime \
	    --public-source-binary bin/nimbus_sim \
	    --config tests/fixtures/configs/qualification/moving_nested_supercell_vulkan_v1.yaml \
	    --output "$$runtime_root/run" \
	    --report reports/pre_beta/physical/moving_nested_vulkan_public_runtime_v1_pass.json

inspect-moving-nested-vulkan-runtime: bin/nimbus_sim_internal_qualification $(FIELD_VALIDATOR)
	python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_runtime.py --inspect-only

qualify-moving-nested-metal-runtime-smoke: bin/nimbus_sim_internal_qualification $(FIELD_VALIDATOR)
	runtime_root=$$(mktemp -d /private/tmp/nimbus-moving-nested-metal-smoke.XXXXXX); \
	  python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_runtime.py \
	    --compute-backend metal --fresh-start --duration 30 --allow-no-relocation \
	    --output "$$runtime_root/run"

qualify-moving-nested-metal-public-runtime-smoke: bin/nimbus_sim $(FIELD_VALIDATOR)
	runtime_root=$$(mktemp -d /private/tmp/nimbus-moving-nested-metal-public-smoke.XXXXXX); \
	  python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_runtime.py \
	    --binary $(PHYSICAL_QUALIFICATION_BINARY) --public-runtime \
	    --public-source-binary bin/nimbus_sim --compute-backend metal \
	    --config tests/fixtures/configs/qualification/moving_nested_supercell_metal_v1.yaml \
	    --fresh-start --duration 30 --allow-no-relocation \
	    --output "$$runtime_root/run" \
	    --report reports/pre_beta/physical/moving_nested_metal_public_runtime_smoke_v1_pass.json

qualify-moving-nested-metal-runtime: bin/nimbus_sim_internal_qualification $(FIELD_VALIDATOR)
	python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_runtime.py \
	  --compute-backend metal --duration 4500

inspect-moving-nested-metal-runtime: bin/nimbus_sim_internal_qualification $(FIELD_VALIDATOR)
	python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_runtime.py \
	  --compute-backend metal --duration 4500 --inspect-only

qualify-moving-nested-metal-lifetime: bin/nimbus_sim_internal_qualification $(FIELD_VALIDATOR)
	python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_runtime.py \
	  --compute-backend metal --fresh-start --duration 4500 \
	  --output tests/artifacts/moving_nested_metal_lifetime_v1_segment_0000_4500
	python3 scripts/qualification/nested_domains/inspect_moving_nested_vulkan_lifetime_chain.py \
	  --compute-backend metal

inspect-moving-nested-metal-lifetime: bin/nimbus_sim_internal_qualification $(FIELD_VALIDATOR)
	python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_runtime.py \
	  --compute-backend metal --fresh-start --duration 4500 --inspect-only \
	  --output tests/artifacts/moving_nested_metal_lifetime_v1_segment_0000_4500
	python3 scripts/qualification/nested_domains/inspect_moving_nested_vulkan_lifetime_chain.py \
	  --compute-backend metal

inspect-moving-nested-metal-lifetime-chain:
	python3 scripts/qualification/nested_domains/inspect_moving_nested_vulkan_lifetime_chain.py \
	  --compute-backend metal

qualify-moving-nested-metal-reference-smoke: bin/nimbus_sim_internal_qualification $(FIELD_VALIDATOR)
	python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_reference.py \
	  --compute-backend metal --duration 30 --allow-short-smoke \
	  --output "$(MOVING_NESTED_METAL_REFERENCE_SMOKE_OUTPUT)"

qualify-moving-nested-metal-reference: bin/nimbus_sim_internal_qualification $(FIELD_VALIDATOR)
	python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_reference.py \
	  --compute-backend metal
	python3 scripts/qualification/nested_domains/seal_moving_nested_accelerator_reference_evidence.py \
	  --compute-backend metal

inspect-moving-nested-metal-reference: bin/nimbus_sim_internal_qualification $(FIELD_VALIDATOR)
	python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_reference.py \
	  --compute-backend metal --inspect-only

seal-moving-nested-metal-reference-evidence: bin/nimbus_sim_internal_qualification
	python3 scripts/qualification/nested_domains/seal_moving_nested_accelerator_reference_evidence.py \
	  --compute-backend metal

ratify-moving-nested-metal-reference-policy:
	python3 scripts/qualification/nested_domains/ratify_moving_nested_metal_reference_policy.py $(METAL_REFERENCE_POLICY_ARGS)

qualify-moving-nested-metal-reference-frontier: bin/nimbus_sim_internal_qualification
	python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_reference_frontier.py \
	  --compute-backend metal
	python3 scripts/qualification/nested_domains/seal_moving_nested_accelerator_frontier_evidence.py \
	  --compute-backend metal

seal-moving-nested-metal-reference-frontier-evidence:
	python3 scripts/qualification/nested_domains/seal_moving_nested_accelerator_frontier_evidence.py \
	  --compute-backend metal

test-moving-nested-vulkan-continuation:
	python3 -m unittest tests/integration/test_moving_nested_vulkan_continuation.py

diagnose-moving-nested-vulkan-continuation:
	python3 scripts/qualification/nested_domains/compare_moving_nested_vulkan_continuation.py

test-moving-nested-vulkan-lifetime-chain:
	python3 -m unittest tests/integration/test_moving_nested_vulkan_lifetime_chain.py

inspect-moving-nested-vulkan-lifetime-chain:
	python3 scripts/qualification/nested_domains/inspect_moving_nested_vulkan_lifetime_chain.py

diagnose-moving-nested-vulkan-lifetime:
	python3 scripts/qualification/nested_domains/compare_moving_nested_vulkan_lifetime.py

qualify-moving-nested-vulkan-reference: bin/nimbus_sim_internal_qualification $(FIELD_VALIDATOR)
	./scripts/run_with_vulkan_env.sh python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_reference.py

inspect-moving-nested-vulkan-reference: bin/nimbus_sim_internal_qualification $(FIELD_VALIDATOR)
	python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_reference.py --inspect-only

qualify-moving-nested-vulkan-reference-frontier: bin/nimbus_sim_internal_qualification
	python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_reference_frontier.py

qualify-moving-nested-vulkan-relocation-cost:
	python3 scripts/qualification/nested_domains/qualify_moving_nested_vulkan_relocation_cost.py

$(FIELD_VALIDATOR): $(FIELD_VALIDATOR_OBJS) $(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(FIELD_VALIDATOR_OBJS) $(LDFLAGS) $(LDLIBS) -o $(FIELD_VALIDATOR)

.PHONY: FORCE
FORCE:

$(MAIN_FLAGS_STAMP): FORCE
	@mkdir -p $(dir $@)
	@tmp="$@.tmp"; \
	  compiler_path="$$(command -v $(firstword $(CXX)) 2>/dev/null || printf '%s' '$(firstword $(CXX))')"; \
	  compiler_version="$$($(CXX) --version 2>/dev/null | sed -n '1p')"; \
	  { printf '%s\n' 'compiler=$(CXX)'; \
	    printf '%s\n' "compiler_path=$$compiler_path"; \
	    printf '%s\n' "compiler_version=$$compiler_version"; \
	    printf '%s\n' 'cxxflags=$(CXXFLAGS) $(METAL_MAIN_FLAGS) $(MAIN_ASSERT_FLAGS) $(CPPFLAGS)'; \
	    printf '%s\n' 'ldflags=$(LDFLAGS) $(LDLIBS) $(METAL_LDLIBS)'; } > "$$tmp"; \
	  if ! cmp -s "$$tmp" "$@"; then mv "$$tmp" "$@"; else rm -f "$$tmp"; fi

# The stamp's contents, not merely Makefile timestamps, track the exact
# compiler and flags. It changes only when the build identity changes, which
# invalidates every dedicated benchmark object without defeating incrementality.
$(BENCHMARK_FLAGS_STAMP): FORCE
	@mkdir -p $(dir $@)
	@tmp="$@.tmp"; \
	  compiler_path="$$(command -v $(firstword $(CXX)) 2>/dev/null || printf '%s' '$(firstword $(CXX))')"; \
	  compiler_version="$$($(CXX) --version 2>/dev/null | sed -n '1p')"; \
	  { printf '%s\n' 'compiler=$(CXX)'; \
	    printf '%s\n' "compiler_path=$$compiler_path"; \
	    printf '%s\n' "compiler_version=$$compiler_version"; \
	    printf '%s\n' 'cxxflags=$(BENCHMARK_CXXFLAGS)'; \
	    printf '%s\n' 'ldflags=$(LDFLAGS) $(LDLIBS)'; } > "$$tmp"; \
	  if ! cmp -s "$$tmp" "$@"; then mv "$$tmp" "$@"; else rm -f "$$tmp"; fi

$(BENCHMARK_BUILDDIR)/%.o: %.cpp $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(BENCHMARK_BIN): $(BENCHMARK_OBJS) $(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(BENCHMARK_OBJS) $(LDFLAGS) $(LDLIBS) -o $(BENCHMARK_BIN)

$(ACTIVE_PHYSICS_OPT_BUILDDIR)/%.o: %.cpp $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) -DNIMBUS_ALLOCATION_TELEMETRY \
		$(DEPFLAGS) -c $< -o $@

$(NESTED_RELOCATION_BENCHMARK_BUILDDIR)/%.o: %.cpp \
		$(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) -I tests -I vulkan/include \
		-DNIMBUS_ALLOCATION_TELEMETRY \
		-DNIMBUS_NESTED_RELOCATION_BENCHMARK \
		$(DEPFLAGS) -c $< -o $@

$(ACTIVE_PHYSICS_LEGACY_BUILDDIR)/%.o: %.cpp $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) -DNIMBUS_ALLOCATION_TELEMETRY \
		-DNIMBUS_PHYSICS_LEGACY_ALLOCATION_EXPERIMENT \
		$(DEPFLAGS) -c $< -o $@

$(SMAG_FILTER_LEGACY_BUILDDIR)/%.o: %.cpp $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) -DNIMBUS_ALLOCATION_TELEMETRY \
		-DNIMBUS_SMAGORINSKY_LEGACY_FILTER_WIDTH_EXPERIMENT \
		$(DEPFLAGS) -c $< -o $@

$(SGS_VERTICAL_STRESS_LEGACY_OBJ): \
		$(SGS_VERTICAL_STRESS_SRC) $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) -DNIMBUS_ALLOCATION_TELEMETRY \
		-DNIMBUS_RECOMPUTE_VERTICAL_SGS_STRESSES \
		$(DEPFLAGS) -c $< -o $@

$(ZFP_HISTORY_COMPARATOR_OBJ): \
		$(ZFP_HISTORY_SRC) $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) \
		-DNIMBUS_ZFP_RECOMPRESS_HISTORY \
		$(DEPFLAGS) -c $< -o $@

$(ZFP_BRICK_GATHER_COMPARATOR_OBJ): \
		$(ZFP_HISTORY_SRC) $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) \
		-DNIMBUS_ZFP_GATHER_BRICKS \
		$(DEPFLAGS) -c $< -o $@

$(ZFP_TRANSIENT_WORKSPACE_COMPARATOR_OBJ): \
		$(ZFP_HISTORY_SRC) $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) \
		-DNIMBUS_ZFP_TRANSIENT_CODEC_WORKSPACE -DNIMBUS_ZFP_PHASE_TIMING \
		$(DEPFLAGS) -c $< -o $@

$(ZFP_WORKSPACE_PHASE_OBJ): \
		$(ZFP_HISTORY_SRC) $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) -DNIMBUS_ZFP_PHASE_TIMING \
		$(DEPFLAGS) -c $< -o $@

$(VULKAN_SCALAR_ATTRIBUTION_COMPARATOR_OBJ): \
		$(VULKAN_SCALAR_ATTRIBUTION_SRC) $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) \
		-DNIMBUS_SCALAR_ATTRIBUTION_REDUNDANT_COEFFICIENT_PREPASS \
		$(DEPFLAGS) -c $< -o $@

$(VULKAN_SCALAR_REGIONS_EXPERIMENT_OBJ): \
		$(VULKAN_SCALAR_ATTRIBUTION_SRC) $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) \
		-DNIMBUS_SCALAR_ATTRIBUTION_COMBINED_OPENMP_EXPERIMENT \
		$(DEPFLAGS) -c $< -o $@

$(VULKAN_NOOP_BUDGET_COMPARATOR_OBJ): \
		$(VULKAN_NOOP_BUDGET_SRC) $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) \
		-DNIMBUS_REDUNDANT_NOOP_BUDGET_SCANS \
		$(DEPFLAGS) -c $< -o $@

$(VULKAN_ENDPOINT_MOISTURE_COMPARATOR_OBJ): \
		$(VULKAN_NOOP_BUDGET_SRC) $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) \
		-DNIMBUS_SEPARATE_ENDPOINT_MOISTURE_SCANS \
		$(DEPFLAGS) -c $< -o $@

$(VULKAN_MICROPHYSICS_GATE_COMPARATOR_OBJ): \
		$(VULKAN_MICROPHYSICS_GATE_SRC) $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) \
		-DNIMBUS_SEPARATE_MICROPHYSICS_CONSERVATION_SCAN \
		$(DEPFLAGS) -c $< -o $@

$(VULKAN_MICROPHYSICS_VALIDATION_COMPARATOR_OBJ): \
		$(VULKAN_MICROPHYSICS_GATE_SRC) $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) \
		-DNIMBUS_SEPARATE_MICROPHYSICS_TENDENCY_VALIDATION_SCANS \
		$(DEPFLAGS) -c $< -o $@

$(VULKAN_MICROPHYSICS_INPUT_VALIDATION_COMPARATOR_OBJ): \
		$(VULKAN_MICROPHYSICS_GATE_SRC) $(BENCHMARK_FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX_COMPILE) $(BENCHMARK_CXXFLAGS) \
		-DNIMBUS_BRANCHING_MICROPHYSICS_INPUT_VALIDATION \
		$(DEPFLAGS) -c $< -o $@

$(ACTIVE_PHYSICS_OPT_BIN): $(ACTIVE_PHYSICS_BASE_OBJS) \
		$(ACTIVE_PHYSICS_OPT_SPECIAL_OBJS) $(ACTIVE_PHYSICS_TELEMETRY_OBJ) \
		$(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(ACTIVE_PHYSICS_BASE_OBJS) \
		$(ACTIVE_PHYSICS_OPT_SPECIAL_OBJS) $(ACTIVE_PHYSICS_TELEMETRY_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(NESTED_RELOCATION_BENCHMARK_BIN): \
		$(NESTED_RELOCATION_BENCHMARK_BASE_OBJS) \
		$(NESTED_RELOCATION_BENCHMARK_TEST_OBJS) \
		$(ACTIVE_PHYSICS_TELEMETRY_OBJ) $(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(NESTED_RELOCATION_BENCHMARK_BASE_OBJS) \
		$(NESTED_RELOCATION_BENCHMARK_TEST_OBJS) \
		$(ACTIVE_PHYSICS_TELEMETRY_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(ACTIVE_PHYSICS_LEGACY_BIN): $(ACTIVE_PHYSICS_BASE_OBJS) \
		$(ACTIVE_PHYSICS_LEGACY_SPECIAL_OBJS) $(ACTIVE_PHYSICS_TELEMETRY_OBJ) \
		$(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(ACTIVE_PHYSICS_BASE_OBJS) \
		$(ACTIVE_PHYSICS_LEGACY_SPECIAL_OBJS) $(ACTIVE_PHYSICS_TELEMETRY_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(SMAG_FILTER_LEGACY_BIN): $(SMAG_FILTER_BASE_OBJS) \
		$(SMAG_FILTER_LEGACY_OBJ) $(ACTIVE_PHYSICS_OPT_SPECIAL_OBJS) \
		$(ACTIVE_PHYSICS_TELEMETRY_OBJ) $(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(SMAG_FILTER_BASE_OBJS) $(SMAG_FILTER_LEGACY_OBJ) \
		$(ACTIVE_PHYSICS_OPT_SPECIAL_OBJS) $(ACTIVE_PHYSICS_TELEMETRY_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(SGS_VERTICAL_STRESS_LEGACY_BIN): $(SGS_VERTICAL_STRESS_BASE_OBJS) \
		$(SGS_VERTICAL_STRESS_LEGACY_OBJ) \
		$(ACTIVE_PHYSICS_OPT_SPECIAL_OBJS) \
		$(ACTIVE_PHYSICS_TELEMETRY_OBJ) $(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(SGS_VERTICAL_STRESS_BASE_OBJS) \
		$(SGS_VERTICAL_STRESS_LEGACY_OBJ) \
		$(ACTIVE_PHYSICS_OPT_SPECIAL_OBJS) $(ACTIVE_PHYSICS_TELEMETRY_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(ZFP_HISTORY_COMPARATOR_BIN): $(ZFP_HISTORY_BASE_OBJS) \
		$(ZFP_HISTORY_COMPARATOR_OBJ) $(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(ZFP_HISTORY_BASE_OBJS) $(ZFP_HISTORY_COMPARATOR_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(ZFP_BRICK_GATHER_COMPARATOR_BIN): $(ZFP_HISTORY_BASE_OBJS) \
		$(ZFP_BRICK_GATHER_COMPARATOR_OBJ) $(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(ZFP_HISTORY_BASE_OBJS) $(ZFP_BRICK_GATHER_COMPARATOR_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(ZFP_TRANSIENT_WORKSPACE_COMPARATOR_BIN): $(ZFP_HISTORY_BASE_OBJS) \
		$(ZFP_TRANSIENT_WORKSPACE_COMPARATOR_OBJ) $(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(ZFP_HISTORY_BASE_OBJS) \
		$(ZFP_TRANSIENT_WORKSPACE_COMPARATOR_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(ZFP_WORKSPACE_PHASE_BIN): $(ZFP_HISTORY_BASE_OBJS) \
		$(ZFP_WORKSPACE_PHASE_OBJ) $(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(ZFP_HISTORY_BASE_OBJS) $(ZFP_WORKSPACE_PHASE_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(VULKAN_SCALAR_ATTRIBUTION_COMPARATOR_BIN): \
		$(VULKAN_SCALAR_ATTRIBUTION_BASE_OBJS) \
		$(VULKAN_SCALAR_ATTRIBUTION_COMPARATOR_OBJ) \
		$(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(VULKAN_SCALAR_ATTRIBUTION_BASE_OBJS) \
		$(VULKAN_SCALAR_ATTRIBUTION_COMPARATOR_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(VULKAN_SCALAR_REGIONS_EXPERIMENT_BIN): \
		$(VULKAN_SCALAR_ATTRIBUTION_BASE_OBJS) \
		$(VULKAN_SCALAR_REGIONS_EXPERIMENT_OBJ) \
		$(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(VULKAN_SCALAR_ATTRIBUTION_BASE_OBJS) \
		$(VULKAN_SCALAR_REGIONS_EXPERIMENT_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(VULKAN_NOOP_BUDGET_COMPARATOR_BIN): \
		$(VULKAN_NOOP_BUDGET_BASE_OBJS) \
		$(VULKAN_NOOP_BUDGET_COMPARATOR_OBJ) \
		$(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(VULKAN_NOOP_BUDGET_BASE_OBJS) \
		$(VULKAN_NOOP_BUDGET_COMPARATOR_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(VULKAN_ENDPOINT_MOISTURE_COMPARATOR_BIN): \
		$(VULKAN_NOOP_BUDGET_BASE_OBJS) \
		$(VULKAN_ENDPOINT_MOISTURE_COMPARATOR_OBJ) \
		$(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(VULKAN_NOOP_BUDGET_BASE_OBJS) \
		$(VULKAN_ENDPOINT_MOISTURE_COMPARATOR_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(VULKAN_MICROPHYSICS_GATE_COMPARATOR_BIN): \
		$(VULKAN_MICROPHYSICS_GATE_BASE_OBJS) \
		$(VULKAN_MICROPHYSICS_GATE_COMPARATOR_OBJ) \
		$(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(VULKAN_MICROPHYSICS_GATE_BASE_OBJS) \
		$(VULKAN_MICROPHYSICS_GATE_COMPARATOR_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(VULKAN_MICROPHYSICS_VALIDATION_COMPARATOR_BIN): \
		$(VULKAN_MICROPHYSICS_GATE_BASE_OBJS) \
		$(VULKAN_MICROPHYSICS_VALIDATION_COMPARATOR_OBJ) \
		$(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(VULKAN_MICROPHYSICS_GATE_BASE_OBJS) \
		$(VULKAN_MICROPHYSICS_VALIDATION_COMPARATOR_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(VULKAN_MICROPHYSICS_INPUT_VALIDATION_COMPARATOR_BIN): \
		$(VULKAN_MICROPHYSICS_GATE_BASE_OBJS) \
		$(VULKAN_MICROPHYSICS_INPUT_VALIDATION_COMPARATOR_OBJ) \
		$(SOURCE_MANIFEST) | bin
	$(CXX_COMPILE) $(VULKAN_MICROPHYSICS_GATE_BASE_OBJS) \
		$(VULKAN_MICROPHYSICS_INPUT_VALIDATION_COMPARATOR_OBJ) \
		$(LDFLAGS) $(LDLIBS) -o $@

bin:
	mkdir -p bin

run: $(BIN)
	./$(BIN)

metal/include/compute/metal/shaders/acoustics/acoustic_eos_msl.hpp: vulkan/shaders/compute/acoustic/pressure_eos.glsl vulkan/shaders/compute/acoustic/momentum_precision.glsl vulkan/shaders/compute/microphysics/compensated_fma.glsl scripts/dev/shaders/generate_acoustic_eos_msl.py
	python3 scripts/dev/shaders/generate_acoustic_eos_msl.py

metal/include/compute/metal/shaders/advection/positive_scalar_budget_msl.hpp: vulkan/shaders/compute/advection/positive_scalar_budget.glsl scripts/dev/shaders/generate_positive_scalar_budget_msl.py
	python3 scripts/dev/shaders/generate_positive_scalar_budget_msl.py

metal/include/compute/metal/shaders/chaos/earth_control_evolution_msl.hpp: vulkan/shaders/compute/chaos/earth_control_evolution.comp vulkan/shaders/compute/microphysics/compensated_fma.glsl scripts/dev/shaders/generate_earth_control_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_earth_control_msl.py

TKE_MSL_HEADERS := metal/include/compute/metal/shaders/turbulence/tke_closure_msl.hpp \
                   metal/include/compute/metal/shaders/turbulence/tke_tendency_msl.hpp
$(TKE_MSL_HEADERS): metal/include/compute/metal/shaders/turbulence/%_msl.hpp: vulkan/shaders/compute/turbulence/%.comp scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_tke_msl.py --kernel $*

IMPLICIT_MSL_HEADERS := metal/include/compute/metal/shaders/diffusion/implicit_diffusion_stencil_msl.hpp \
                       metal/include/compute/metal/shaders/diffusion/implicit_diffusion_vector_msl.hpp
$(IMPLICIT_MSL_HEADERS): metal/include/compute/metal/shaders/diffusion/%_msl.hpp: vulkan/shaders/compute/diffusion/%.comp scripts/dev/shaders/generate_implicit_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_implicit_msl.py --kernel $*

MILBRANDT_MSL_HEADERS := metal/include/compute/metal/shaders/microphysics/milbrandt_sedimentation_msl.hpp \
                        metal/include/compute/metal/shaders/microphysics/milbrandt_cold_msl.hpp \
                        metal/include/compute/metal/shaders/microphysics/milbrandt_melting_msl.hpp \
                        metal/include/compute/metal/shaders/microphysics/milbrandt_warm_rain_msl.hpp \
                         metal/include/compute/metal/shaders/microphysics/milbrandt_cloud_adjustment_msl.hpp
$(MILBRANDT_MSL_HEADERS): metal/include/compute/metal/shaders/microphysics/%_msl.hpp: \
        vulkan/shaders/compute/microphysics/compensated_fma.glsl \
        vulkan/shaders/compute/microphysics/%.comp \
        vulkan/shaders/compute/microphysics/milbrandt_thermodynamics.glsl \
        vulkan/shaders/compute/microphysics/liquid_adjustment.glsl \
        vulkan/shaders/compute/microphysics/thompson_compensated.glsl \
        scripts/dev/shaders/generate_milbrandt_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_milbrandt_msl.py --kernel $*

RK_MSL_HEADERS := metal/include/compute/metal/shaders/time_stepping/rk_combination_msl.hpp \
                  metal/include/compute/metal/shaders/time_stepping/rk_tendency_msl.hpp
$(RK_MSL_HEADERS): metal/include/compute/metal/shaders/time_stepping/%_msl.hpp: vulkan/shaders/compute/time_stepping/%.comp vulkan/shaders/compute/microphysics/compensated_fma.glsl scripts/dev/shaders/generate_rk_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_rk_msl.py --kernel $*

metal/include/compute/metal/shaders/turbulence/collocated_sgs_msl.hpp: vulkan/shaders/compute/turbulence/collocated_sgs.comp vulkan/shaders/compute/microphysics/compensated_fma.glsl scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_tke_msl.py --kernel collocated_sgs

metal/include/compute/metal/shaders/terrain/conservative_remap_msl.hpp: vulkan/shaders/compute/terrain/compensated_storage.glsl vulkan/shaders/compute/terrain/conservative_remap.comp vulkan/shaders/compute/microphysics/compensated_fma.glsl scripts/dev/shaders/generate_remap_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_remap_msl.py

metal/include/compute/metal/shaders/terrain/earth_geometry_msl.hpp: vulkan/shaders/compute/terrain/earth_geometry.comp vulkan/shaders/compute/microphysics/compensated_fma.glsl scripts/dev/shaders/generate_earth_geometry_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_earth_geometry_msl.py

metal/include/compute/metal/shaders/terrain/moving_momentum_msl.hpp: vulkan/shaders/compute/terrain/moving_momentum.comp vulkan/shaders/compute/terrain/compensated_storage.glsl vulkan/shaders/compute/microphysics/compensated_fma.glsl scripts/dev/shaders/generate_moving_momentum_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_moving_momentum_msl.py

metal/include/compute/metal/shaders/terrain/mapped_boundary_msl.hpp: vulkan/shaders/compute/terrain/mapped_boundary.comp scripts/dev/shaders/generate_mapped_boundary_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_mapped_boundary_msl.py

metal/include/compute/metal/shaders/terrain/mapped_scalar_math_msl.hpp: vulkan/shaders/compute/terrain/signed_scalar_update.glsl vulkan/shaders/compute/microphysics/compensated_fma.glsl scripts/dev/shaders/generate_mapped_scalar_math_msl.py
	python3 scripts/dev/shaders/generate_mapped_scalar_math_msl.py

metal/include/compute/metal/shaders/boundary_layer/slab_column_msl.hpp: vulkan/shaders/compute/boundary_layer/slab_column.comp vulkan/shaders/compute/microphysics/compensated_fma.glsl scripts/dev/shaders/generate_slab_column_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_slab_column_msl.py

metal/include/compute/metal/shaders/radiation/simple_grey_msl.hpp: vulkan/shaders/compute/radiation/simple_grey.comp vulkan/shaders/compute/microphysics/compensated_fma.glsl scripts/dev/shaders/generate_simple_grey_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_simple_grey_msl.py

metal/include/compute/metal/shaders/terrain/held_sources_msl.hpp: vulkan/shaders/compute/terrain/held_sources.comp vulkan/shaders/compute/microphysics/compensated_fma.glsl vulkan/shaders/compute/terrain/compensated_storage.glsl scripts/dev/shaders/generate_held_sources_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_held_sources_msl.py

metal/include/compute/metal/shaders/terrain/column_remap_msl.hpp: vulkan/shaders/compute/terrain/column_remap.comp vulkan/shaders/compute/microphysics/compensated_fma.glsl vulkan/shaders/compute/terrain/compensated_storage.glsl scripts/dev/shaders/generate_column_remap_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_column_remap_msl.py

metal/include/compute/metal/shaders/terrain/reference_atmosphere_msl.hpp: vulkan/shaders/compute/terrain/reference_atmosphere.comp vulkan/shaders/compute/microphysics/compensated_fma.glsl vulkan/shaders/compute/terrain/compensated_storage.glsl scripts/dev/shaders/generate_reference_atmosphere_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_reference_atmosphere_msl.py

metal/include/compute/metal/shaders/terrain/moving_energy_msl.hpp: vulkan/shaders/compute/terrain/moving_energy.comp vulkan/shaders/compute/microphysics/compensated_fma.glsl scripts/dev/shaders/generate_moving_energy_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_moving_energy_msl.py

metal/include/compute/metal/shaders/advection/advect_conservative_scalar_flux_cartesian_cgrid_msl.hpp: vulkan/shaders/compute/advection/advect_conservative_scalar_flux_cartesian_cgrid.comp vulkan/shaders/compute/advection/positive_scalar_budget.glsl scripts/dev/shaders/generate_advect_conservative_scalar_flux_cartesian_cgrid_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_advect_conservative_scalar_flux_cartesian_cgrid_msl.py

metal/include/compute/metal/shaders/dynamics/acoustic_bc_face_cartesian_cgrid_msl.hpp: vulkan/shaders/compute/dynamics/acoustic_bc_face_cartesian_cgrid.comp vulkan/shaders/compute/dynamics/boundary_float32.glsl scripts/dev/shaders/generate_acoustic_bc_face_cartesian_cgrid_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_acoustic_bc_face_cartesian_cgrid_msl.py

metal/include/compute/metal/shaders/microphysics/kessler_sedimentation_msl.hpp: vulkan/shaders/compute/microphysics/kessler_sedimentation.comp scripts/dev/shaders/generate_kessler_sedimentation_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_kessler_sedimentation_msl.py

metal/include/compute/metal/shaders/microphysics/lin_pointwise_msl.hpp: vulkan/shaders/compute/microphysics/lin_pointwise.comp scripts/dev/shaders/generate_lin_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_lin_msl.py --shader lin_pointwise

metal/include/compute/metal/shaders/microphysics/lin_sedimentation_msl.hpp: vulkan/shaders/compute/microphysics/lin_sedimentation.comp scripts/dev/shaders/generate_lin_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_lin_msl.py --shader lin_sedimentation

metal/include/compute/metal/shaders/dynamics/conservative_momentum_msl.hpp: vulkan/shaders/compute/dynamics/conservative_momentum.comp vulkan/shaders/compute/microphysics/compensated_fma.glsl scripts/dev/shaders/generate_conservative_momentum_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_conservative_momentum_msl.py

metal/include/compute/metal/shaders/dynamics/conservative_reduction_msl.hpp: vulkan/shaders/compute/dynamics/conservative_reduction.comp vulkan/shaders/compute/microphysics/compensated_fma.glsl vulkan/shaders/compute/terrain/compensated_storage.glsl scripts/dev/shaders/generate_conservative_reduction_msl.py scripts/dev/shaders/generate_tke_msl.py
	python3 scripts/dev/shaders/generate_conservative_reduction_msl.py

THOMPSON_MSL_KERNELS := thompson_moment_sedimentation thompson_cloud_adjustment \
                       thompson_warm_rain thompson_cloud_phase thompson_freezing \
                       thompson_deposition thompson_completion
THOMPSON_MSL_HEADERS := $(addprefix metal/include/compute/metal/shaders/microphysics/,$(addsuffix _msl.hpp,$(THOMPSON_MSL_KERNELS)))

$(THOMPSON_MSL_HEADERS): metal/include/compute/metal/shaders/microphysics/%_msl.hpp: vulkan/shaders/compute/microphysics/%.comp $(wildcard vulkan/shaders/compute/microphysics/*.glsl) scripts/dev/shaders/generate_tke_msl.py scripts/dev/shaders/generate_thompson_msl.py
	python3 scripts/dev/shaders/generate_thompson_msl.py --kernel $*

metal/include/compute/metal/shaders/advection/weno_common_msl.hpp: vulkan/shaders/compute/advection/advect_conservative_scalar_flux_cartesian_cgrid.comp scripts/dev/shaders/generate_weno_common_msl.py
	python3 scripts/dev/shaders/generate_weno_common_msl.py
