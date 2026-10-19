.PHONY: cpu-only
cpu-only:
	$(MAKE) CPU_ONLY=1 all

.PHONY: cpu-only-portable
cpu-only-portable:
	$(MAKE) CPU_ONLY=1 PORTABLE=1 all

.PHONY: qualify-storm-tracking
qualify-storm-tracking: bin/test_diagnostics_storm_tracker bin/test_core_simulation_checkpoint bin/test_init_config_v1 bin/test_init_config_migrator bin/test_qualification_storm_history_replay
	$(MAKE) CPU_ONLY=1 bin/nimbus_sim_cpu
	./bin/test_diagnostics_storm_tracker
	./bin/test_core_simulation_checkpoint
	./bin/test_init_config_v1
	./bin/test_init_config_migrator
	./bin/test_qualification_storm_history_replay
	python3 scripts/qualification/domain_motion/qualify_storm_tracking.py --binary bin/nimbus_sim_cpu

.PHONY: qualify-storm-tracking-vulkan
qualify-storm-tracking-vulkan: bin/nimbus_sim bin/test_diagnostics_storm_tracker bin/test_core_simulation_checkpoint bin/test_init_config_v1 bin/test_init_config_migrator
	./bin/test_diagnostics_storm_tracker
	./bin/test_core_simulation_checkpoint
	./bin/test_init_config_v1
	./bin/test_init_config_migrator
	python3 scripts/qualification/domain_motion/qualify_storm_tracking.py --binary bin/nimbus_sim \
		--compute-backend vulkan --require-vulkan --grid-scope cgrid

.PHONY: qualify-checkpoint-resume qualify-signal-checkpoint qualify-hard-crash-checkpoint qualify-trajectory-checkpoint qualify-stationary-nested-output
qualify-checkpoint-resume: bin/nimbus_sim
	python3 scripts/qualification/checkpointing/qualify_checkpoint_resume.py

qualify-signal-checkpoint: bin/nimbus_sim
	python3 scripts/qualification/checkpointing/qualify_signal_checkpoint.py

qualify-hard-crash-checkpoint: bin/nimbus_sim
	python3 scripts/qualification/checkpointing/qualify_hard_crash_checkpoint.py

qualify-trajectory-checkpoint: bin/nimbus_sim
	python3 scripts/qualification/checkpointing/qualify_trajectory_checkpoint.py

qualify-stationary-nested-output: bin/nimbus_sim $(FIELD_VALIDATOR)
	python3 scripts/qualification/nested_domains/qualify_stationary_nested_output.py

.PHONY: qualify-domain-recycling
qualify-domain-recycling:
	$(MAKE) CPU_ONLY=1 bin/nimbus_sim_cpu
	python3 scripts/qualification/domain_motion/qualify_domain_recycling.py --binary bin/nimbus_sim_cpu
	python3 scripts/qualification/domain_motion/qualify_domain_recycling_sustained.py --binary bin/nimbus_sim_cpu

.PHONY: qualify-domain-recycling-vulkan-field
qualify-domain-recycling-vulkan-field: verify-shaders bin/test_parity_check bin/test_vulkan_required_dispatch bin/test_vulkan_gpu_parity bin/test_vulkan_backend
	./bin/test_parity_check
	./scripts/run_with_vulkan_env.sh ./bin/test_vulkan_required_dispatch
	./scripts/run_with_vulkan_env.sh ./bin/test_vulkan_gpu_parity "Vulkan homogeneous D5 field shift is bitwise equal to the canonical map"
	./bin/test_vulkan_backend "Vulkan backend repeat initialization clears every stale capability"

.PHONY: qualify-domain-recycling-vulkan-batch
qualify-domain-recycling-vulkan-batch: qualify-domain-recycling-vulkan-field
	./scripts/run_with_vulkan_env.sh ./bin/test_vulkan_gpu_parity "[domain_recycling][D5][batch]"
	./scripts/run_with_vulkan_env.sh ./bin/test_vulkan_gpu_parity "[diffusion]"

.PHONY: qualify-domain-recycling-vulkan-live
qualify-domain-recycling-vulkan-live: qualify-domain-recycling-vulkan-batch bin/nimbus_sim
	python3 scripts/qualification/domain_motion/qualify_domain_recycling.py --binary bin/nimbus_sim \
		--compute-backend vulkan --require-vulkan
	python3 scripts/qualification/domain_motion/qualify_domain_recycling_sustained.py --binary bin/nimbus_sim \
		--compute-backend vulkan --require-vulkan

.PHONY: qualify-domain-recycling-metal-live
qualify-domain-recycling-metal-live: bin/nimbus_sim
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/qualification/domain_motion/qualify_domain_recycling.py \
		--binary bin/nimbus_sim --compute-backend metal --require-metal
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/qualification/domain_motion/qualify_domain_recycling_sustained.py \
		--binary bin/nimbus_sim --compute-backend metal --require-metal \
		--continuous-duration 170 --resumed-duration 180

.PHONY: qualify-open-sgs
qualify-open-sgs: bin/test_physics_sgs_apply_runtime_state bin/test_core_simulation_checkpoint bin/test_init_config_v1 bin/test_init_config_migrator bin/test_core_scheme_catalog
	$(MAKE) CPU_ONLY=1 bin/nimbus_sim_cpu
	./bin/test_physics_sgs_apply_runtime_state
	./bin/test_core_simulation_checkpoint
	./bin/test_init_config_v1
	./bin/test_init_config_migrator
	./bin/test_core_scheme_catalog
	python3 tests/science/cgrid/physics/qualify_cgrid_open_sgs.py --binary bin/nimbus_sim_cpu

.PHONY: qualify-open-sgs-vulkan
qualify-open-sgs-vulkan: $(BIN) bin/test_core_scheme_catalog bin/test_vulkan_gpu_parity
	./scripts/run_with_vulkan_env.sh ./bin/test_vulkan_gpu_parity "GPU Smagorinsky to SGS apply preserves resident tendencies"
	./scripts/run_with_vulkan_env.sh python3 tests/science/cgrid/physics/qualify_cgrid_open_sgs.py --binary $(BIN) --backend vulkan

.PHONY: qualify-stationary-storm-lifetime
qualify-stationary-storm-lifetime:
	$(MAKE) CPU_ONLY=1 bin/nimbus_sim_cpu
	python3 scripts/qualification/domain_motion/qualify_stationary_storm_lifetime.py --binary bin/nimbus_sim_cpu

.PHONY: qualify-constant-velocity-domain
qualify-constant-velocity-domain:
	$(MAKE) CPU_ONLY=1 bin/nimbus_sim_cpu
	python3 scripts/qualification/domain_motion/qualify_constant_velocity_domain.py --binary bin/nimbus_sim_cpu

.PHONY: qualify-constant-velocity-vulkan qualify-vulkan-stationary-nesting \
	qualify-vulkan-nested-relocation-substrate \
	qualify-nested-detector-relocation-substrate \
	test-stationary-nested-vulkan-admission \
	qualify-stationary-nested-vulkan-runtime \
	test-stationary-nested-vulkan-smagorinsky-admission \
	qualify-stationary-nested-vulkan-smagorinsky-runtime \
	test-stationary-nested-vulkan-kessler-smagorinsky-admission \
	qualify-stationary-nested-vulkan-kessler-smagorinsky-runtime \
	test-stationary-nested-radiation-admission \
	qualify-stationary-nested-radiation-runtime
qualify-constant-velocity-vulkan: $(BIN)
	./scripts/run_with_vulkan_env.sh python3 scripts/qualification/domain_motion/qualify_constant_velocity_vulkan.py --binary $(PHYSICAL_QUALIFICATION_BINARY)

test-stationary-nested-vulkan-admission: bin/nimbus_sim
	python3 scripts/tests/gpu/test_stationary_nested_vulkan_admission.py --binary bin/nimbus_sim

qualify-stationary-nested-vulkan-runtime: bin/nimbus_sim $(FIELD_VALIDATOR)
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/gpu/test_stationary_nested_vulkan_admission.py \
	  --binary $(PHYSICAL_QUALIFICATION_BINARY) --require-vulkan-positive
	./scripts/run_with_vulkan_env.sh python3 scripts/qualification/nested_domains/qualify_stationary_nested_vulkan_runtime.py \
	  --binary $(PHYSICAL_QUALIFICATION_BINARY) --validator $(FIELD_VALIDATOR)

test-stationary-nested-vulkan-smagorinsky-admission: bin/nimbus_sim
	python3 scripts/tests/gpu/test_stationary_nested_vulkan_smagorinsky_admission.py \
	  --binary bin/nimbus_sim

qualify-stationary-nested-vulkan-smagorinsky-runtime: bin/nimbus_sim $(FIELD_VALIDATOR)
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/gpu/test_stationary_nested_vulkan_smagorinsky_admission.py \
	  --binary $(PHYSICAL_QUALIFICATION_BINARY) --require-vulkan-positive
	./scripts/run_with_vulkan_env.sh python3 scripts/qualification/nested_domains/qualify_stationary_nested_vulkan_runtime.py \
	  --binary $(PHYSICAL_QUALIFICATION_BINARY) --validator $(FIELD_VALIDATOR) \
	  --config tests/fixtures/configs/qualification/stationary_nested_vulkan_smagorinsky.yaml \
	  --require-smagorinsky

test-stationary-nested-vulkan-kessler-smagorinsky-admission: bin/nimbus_sim
	python3 scripts/tests/gpu/test_stationary_nested_vulkan_kessler_smagorinsky_admission.py \
	  --binary bin/nimbus_sim

qualify-stationary-nested-vulkan-kessler-smagorinsky-runtime: bin/nimbus_sim $(FIELD_VALIDATOR)
	./scripts/run_with_vulkan_env.sh python3 scripts/tests/gpu/test_stationary_nested_vulkan_kessler_smagorinsky_admission.py \
	  --binary $(PHYSICAL_QUALIFICATION_BINARY) --require-vulkan-positive
	./scripts/run_with_vulkan_env.sh python3 scripts/qualification/nested_domains/qualify_stationary_nested_vulkan_runtime.py \
	  --binary $(PHYSICAL_QUALIFICATION_BINARY) --validator $(FIELD_VALIDATOR) \
	  --config tests/fixtures/configs/qualification/stationary_nested_vulkan_kessler_smagorinsky.yaml \
	  --duration-s 4 --require-smagorinsky --require-kessler

test-stationary-nested-radiation-admission: bin/nimbus_sim
	python3 scripts/tests/gpu/test_stationary_nested_radiation_admission.py \
	  --binary bin/nimbus_sim

qualify-stationary-nested-radiation-runtime: bin/nimbus_sim $(FIELD_VALIDATOR)
	./scripts/run_with_vulkan_env.sh \
	  python3 scripts/tests/gpu/test_stationary_nested_radiation_admission.py \
	  --binary $(PHYSICAL_QUALIFICATION_BINARY) --require-vulkan-positive
	./scripts/run_with_vulkan_env.sh \
	  python3 scripts/qualification/nested_domains/qualify_stationary_nested_vulkan_runtime.py \
	  --binary $(PHYSICAL_QUALIFICATION_BINARY) --validator $(FIELD_VALIDATOR) \
	  --config tests/fixtures/configs/qualification/stationary_nested_vulkan_radiation.yaml \
	  --duration-s 6 --require-radiation

qualify-vulkan-stationary-nesting: bin/test_core_nested_production_step
	NIMBUS_GPU_REQUIRED=1 ./scripts/run_with_vulkan_env.sh \
	  ./bin/test_core_nested_production_step \
	  "stationary one-way child executes counted physical Vulkan work"
	NIMBUS_GPU_REQUIRED=1 ./scripts/run_with_vulkan_env.sh \
	  ./bin/test_core_nested_production_step \
	  "stationary nested Kessler and Smagorinsky physics qualify physical Vulkan parity"
	NIMBUS_GPU_REQUIRED=1 ./scripts/run_with_vulkan_env.sh \
	  ./bin/test_core_nested_production_step \
	  "stationary child transmits a physical Vulkan acoustic wave"

qualify-vulkan-nested-relocation-substrate: bin/test_core_nested_production_step
	NIMBUS_GPU_REQUIRED=1 ./scripts/run_with_vulkan_env.sh \
	  ./bin/test_core_nested_production_step \
	  "synchronized child relocation resumes exact physical Vulkan work"

qualify-nested-detector-relocation-substrate: bin/test_core_nested_production_step
	./bin/test_core_nested_production_step \
	  "production detector drives repeated synchronized child movement"

.PHONY: qualify-constant-velocity-storm-lifetime
qualify-constant-velocity-storm-lifetime:
	$(MAKE) CPU_ONLY=1 bin/nimbus_sim_cpu
	python3 scripts/qualification/domain_motion/qualify_constant_velocity_storm_lifetime.py --binary bin/nimbus_sim_cpu

.PHONY: test-cpu-only-profile
test-cpu-only-profile: cpu-only
	./bin/nimbus_sim_cpu --version
	./bin/nimbus_sim_cpu --headless --config=tests/fixtures/configs/benchmark/benchmark.yaml --duration=0 --compute-backend cpu --log-profile quiet --no-timing

.PHONY: test-cpu-only-portable-profile
test-cpu-only-portable-profile: cpu-only-portable
	./bin/nimbus_sim_cpu_portable --version
	./bin/nimbus_sim_cpu_portable --headless --config=tests/fixtures/configs/benchmark/benchmark.yaml --duration=0 --compute-backend cpu --log-profile quiet --no-timing

# ---------------------------------------------------------------------------
# Vulkan viewer and compute shaders
# ---------------------------------------------------------------------------
.PHONY: check-vulkan-assets
check-vulkan-assets:
	python3 scripts/dev/shaders/check_artifacts.py
	python3 scripts/dev/shaders/verify_kernel_manifest.py

vulkan-compute-shaders:
	python3 scripts/dev/shaders/verify_spirv.py --refresh
	python3 scripts/dev/shaders/verify_kernel_manifest.py --refresh

.PHONY: qualify-agrid-native-continuity-vulkan
qualify-agrid-native-continuity-vulkan: bin/nimbus_sim vulkan-compute-shaders
	./scripts/qualification/accelerators/qualify_agrid_native_continuity.sh vulkan

.PHONY: qualify-agrid-native-continuity-metal
qualify-agrid-native-continuity-metal: bin/nimbus_sim
	./scripts/qualification/accelerators/qualify_agrid_native_continuity.sh metal

.PHONY: qualify-agrid-native-continuity
qualify-agrid-native-continuity: qualify-agrid-native-continuity-vulkan qualify-agrid-native-continuity-metal

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
	@printf "Compiler cache: "; \
	  if [ -n "$(CACHE_PROGRAM)" ]; then \
	    $(CACHE_PROGRAM) --version 2>&1 | head -1; \
	  else \
	    echo "not installed (optional: ccache or sccache)"; \
	  fi
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
