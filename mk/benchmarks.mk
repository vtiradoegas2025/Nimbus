BENCH_CONFIG ?= tests/fixtures/configs/benchmark/benchmark.yaml
BENCH_TRIALS ?= 7
BENCH_THREADS ?= $(shell python3 -c "import os; print(os.cpu_count() or 1)")
BENCH_TIMEOUT ?= 0
BENCH_OUTPUT ?=
BENCH_BASELINE ?=
BENCH_POLICY ?=
BENCH_REPORT ?= reports/benchmarks/cpu-current.json
BENCH_COMPARISON ?=
BENCH_SCALE_CONFIG ?= tests/fixtures/configs/benchmark/cpu_scaling_student.yaml
BENCH_SCALE_THREADS ?=
BENCH_SCALE_OUTPUT ?= reports/benchmarks/cpu-scaling-current.json
BENCH_SCALE_ENDPOINT_DURATION ?= 30
benchmark-cpu: $(BENCHMARK_BIN)
	python3 scripts/benchmarks/cpu/run_cpu_benchmark.py \
	  --binary "$(BENCHMARK_BIN)" --config "$(BENCH_CONFIG)" \
	  --trials "$(BENCH_TRIALS)" --threads "$(BENCH_THREADS)" \
	  --timeout "$(BENCH_TIMEOUT)" $(if $(strip $(BENCH_OUTPUT)),--output "$(BENCH_OUTPUT)",) \
	  --build-profile "release-like -O3 -DNDEBUG" \
	  --compiler "$(CXX)" \
	  --cxxflags "$(BENCHMARK_CXXFLAGS)" \
	  --ldflags "$(LDFLAGS) $(LDLIBS)"

benchmark-cpu-scaling: $(BENCHMARK_BIN)
	python3 scripts/benchmarks/cpu/benchmark_cpu_scaling.py \
	  --binary "$(BENCHMARK_BIN)" --config "$(BENCH_SCALE_CONFIG)" \
	  --trials "$(BENCH_TRIALS)" \
	  $(if $(strip $(BENCH_SCALE_THREADS)),--thread-counts "$(BENCH_SCALE_THREADS)",) \
	  --endpoint-duration "$(BENCH_SCALE_ENDPOINT_DURATION)" \
	  --timeout "$(BENCH_TIMEOUT)" --output "$(BENCH_SCALE_OUTPUT)" \
	  --build-profile "release-like -O3 -DNDEBUG" \
	  --compiler "$(CXX)" \
	  --cxxflags "$(BENCHMARK_CXXFLAGS)" \
	  --ldflags "$(LDFLAGS) $(LDLIBS)"

NESTED_RELOCATION_BENCH_CHILD_NX ?= 128
NESTED_RELOCATION_BENCH_CHILD_NY ?= 128
NESTED_RELOCATION_BENCH_NZ ?= 64
NESTED_RELOCATION_BENCH_WARMUP_MOVES ?= 4
NESTED_RELOCATION_BENCH_MEASURED_MOVES ?= 8
NESTED_RELOCATION_BENCH_TRIALS ?= 5
NESTED_RELOCATION_BENCH_THREADS ?= 1
NESTED_RELOCATION_BENCH_PROFILE ?= baseline
NESTED_RELOCATION_BENCH_REPORT ?= \
	reports/benchmarks/nested-relocation-current.json

benchmark-nested-relocation: $(NESTED_RELOCATION_BENCHMARK_BIN)
	python3 scripts/benchmarks/domain/benchmark_nested_relocation.py \
	  --binary "$(NESTED_RELOCATION_BENCHMARK_BIN)" \
	  --child-nx "$(NESTED_RELOCATION_BENCH_CHILD_NX)" \
	  --child-ny "$(NESTED_RELOCATION_BENCH_CHILD_NY)" \
	  --nz "$(NESTED_RELOCATION_BENCH_NZ)" \
	  --warmup-moves "$(NESTED_RELOCATION_BENCH_WARMUP_MOVES)" \
	  --measured-moves "$(NESTED_RELOCATION_BENCH_MEASURED_MOVES)" \
	  --trials "$(NESTED_RELOCATION_BENCH_TRIALS)" \
	  --threads "$(NESTED_RELOCATION_BENCH_THREADS)" \
	  --profile "$(NESTED_RELOCATION_BENCH_PROFILE)" \
	  --compiler "$(CXX)" --cxxflags "$(BENCHMARK_CXXFLAGS)" \
	  --output "$(NESTED_RELOCATION_BENCH_REPORT)"

test-nested-relocation-benchmark:
	python3 -m unittest \
	  tests/integration/test_nested_relocation_benchmark.py

TERRAIN_SW_BENCH_TRIALS ?= 5
TERRAIN_SW_BENCH_THREADS ?= 1,4
TERRAIN_SW_BENCH_REPORT ?= reports/benchmarks/terrain-shortwave-cpu-current.json

benchmark-terrain-shortwave-cpu: $(ACTIVE_PHYSICS_OPT_BIN)
	python3 scripts/benchmarks/cpu/benchmark_terrain_shortwave_cpu.py \
	  --binary "$(ACTIVE_PHYSICS_OPT_BIN)" \
	  --config "tests/fixtures/configs/benchmark/terrain_shortwave_cpu.yaml" \
	  --trials "$(TERRAIN_SW_BENCH_TRIALS)" \
	  --thread-counts "$(TERRAIN_SW_BENCH_THREADS)" \
	  --output "$(TERRAIN_SW_BENCH_REPORT)"

TERRAIN_SW_SCALE_TRIALS ?= 7
TERRAIN_SW_SCALE_THREADS ?= 1,4
TERRAIN_SW_SCALE_SIZES ?= 16,32,64
TERRAIN_SW_SCALE_REPORT ?= reports/benchmarks/terrain-shortwave-cpu-scaling-current.json

benchmark-terrain-shortwave-cpu-scaling: $(ACTIVE_PHYSICS_OPT_BIN)
	python3 scripts/benchmarks/cpu/benchmark_terrain_shortwave_cpu_scaling.py \
	  --binary "$(ACTIVE_PHYSICS_OPT_BIN)" \
	  --config "tests/fixtures/configs/benchmark/terrain_shortwave_cpu.yaml" \
	  --trials "$(TERRAIN_SW_SCALE_TRIALS)" \
	  --thread-counts "$(TERRAIN_SW_SCALE_THREADS)" \
	  --horizontal-sizes "$(TERRAIN_SW_SCALE_SIZES)" \
	  --output "$(TERRAIN_SW_SCALE_REPORT)"

benchmark-active-physics-allocations: $(ACTIVE_PHYSICS_OPT_BIN) \
		$(ACTIVE_PHYSICS_LEGACY_BIN)
	python3 scripts/benchmarks/cpu/benchmark_active_physics_allocations.py \
	  --optimized "$(ACTIVE_PHYSICS_OPT_BIN)" \
	  --legacy "$(ACTIVE_PHYSICS_LEGACY_BIN)" \
	  --config "tests/fixtures/configs/benchmark/active_physics_allocations.yaml" \
	  --trials "$(BENCH_TRIALS)" --threads "$(BENCH_THREADS)"

benchmark-milbrandt-allocations: $(ACTIVE_PHYSICS_OPT_BIN) \
		$(ACTIVE_PHYSICS_LEGACY_BIN)
	python3 scripts/benchmarks/cpu/benchmark_active_physics_allocations.py \
	  --optimized "$(ACTIVE_PHYSICS_OPT_BIN)" \
	  --legacy "$(ACTIVE_PHYSICS_LEGACY_BIN)" \
	  --config "tests/fixtures/configs/benchmark/milbrandt_allocations.yaml" \
	  --duration 40 --trials "$(BENCH_TRIALS)" --threads "$(BENCH_THREADS)" \
	  --output "reports/benchmarks/milbrandt-allocation-current.json"

benchmark-smagorinsky-filter-width: $(ACTIVE_PHYSICS_OPT_BIN) \
		$(SMAG_FILTER_LEGACY_BIN)
	python3 scripts/benchmarks/cpu/benchmark_active_physics_allocations.py \
	  --optimized "$(ACTIVE_PHYSICS_OPT_BIN)" \
	  --legacy "$(SMAG_FILTER_LEGACY_BIN)" \
	  --config "tests/fixtures/configs/benchmark/active_physics_allocations.yaml" \
	  --trials "$(BENCH_TRIALS)" --threads "$(BENCH_THREADS)" \
	  --comparison-contract "same equations/order/cadence/workers; comparator changes only flat-grid Smagorinsky filter-width evaluation frequency" \
	  --minimum-speedup-percent 2 \
	  --output "reports/benchmarks/smagorinsky-filter-width-current.json"

benchmark-sgs-vertical-stress-reuse: $(ACTIVE_PHYSICS_OPT_BIN) \
		$(SGS_VERTICAL_STRESS_LEGACY_BIN)
	python3 scripts/benchmarks/cpu/benchmark_active_physics_allocations.py \
	  --optimized "$(ACTIVE_PHYSICS_OPT_BIN)" \
	  --legacy "$(SGS_VERTICAL_STRESS_LEGACY_BIN)" \
	  --config "tests/fixtures/configs/benchmark/cpu_scaling_student.yaml" \
	  --duration 40 --trials "$(BENCH_TRIALS)" --threads "$(BENCH_THREADS)" \
	  --comparison-contract "same equations/order/cadence/workers and binary64 face stresses; comparator recomputes shared vertical SGS stresses in each adjacent control volume" \
	  --minimum-speedup-percent 2 \
	  --output "reports/benchmarks/sgs-vertical-stress-reuse-current.json"

benchmark-zfp-history-working-set: $(BENCHMARK_BIN) \
		$(ZFP_HISTORY_COMPARATOR_BIN)
	python3 scripts/benchmarks/storage/benchmark_zfp_history_working_set.py \
	  --optimized "$(BENCHMARK_BIN)" \
	  --comparator "$(ZFP_HISTORY_COMPARATOR_BIN)" \
	  --config "tests/fixtures/configs/benchmark/zfp_history_working_set.yaml" \
	  --duration 20 --trials "$(BENCH_TRIALS)" --threads 1 \
	  --minimum-speedup-percent 2 \
	  --output "reports/benchmarks/zfp-history-working-set-current.json"

benchmark-zfp-brick-strides: $(BENCHMARK_BIN) \
		$(ZFP_BRICK_GATHER_COMPARATOR_BIN) bin/test_core_zfp_roundtrip
	./bin/test_core_zfp_roundtrip \
	  "[core][zfp][bricks][writer][allocation]"
	python3 scripts/benchmarks/storage/benchmark_zfp_history_working_set.py \
	  --optimized "$(BENCHMARK_BIN)" \
	  --comparator "$(ZFP_BRICK_GATHER_COMPARATOR_BIN)" \
	  --config "tests/fixtures/configs/benchmark/zfp_history_working_set.yaml" \
	  --duration 20 --trials "$(BENCH_TRIALS)" --threads 1 \
	  --minimum-speedup-percent 2 \
	  --experiment "strided_brick_views" \
	  --comparison-contract "same source, equations, compression parameters, brick order, codec, I/O, history update, config, and threads; comparator only gathers/scatters each brick through two reusable contiguous float buffers instead of exposing strided source/history views directly to libzfp" \
	  --deterministic-memory-reduction-bytes 1048576 \
	  --accept-memory-only \
	  --output "reports/benchmarks/zfp-brick-strides-current.json"

benchmark-zfp-codec-workspace: $(ZFP_WORKSPACE_PHASE_BIN) \
		$(ZFP_TRANSIENT_WORKSPACE_COMPARATOR_BIN) bin/test_core_zfp_roundtrip
	./bin/test_core_zfp_roundtrip \
	  "[core][zfp][writer][workspace]"
	python3 scripts/benchmarks/storage/benchmark_zfp_history_working_set.py \
	  --optimized "$(ZFP_WORKSPACE_PHASE_BIN)" \
	  --comparator "$(ZFP_TRANSIENT_WORKSPACE_COMPARATOR_BIN)" \
	  --config "tests/fixtures/configs/benchmark/zfp_history_working_set.yaml" \
	  --duration 20 --trials "$(BENCH_TRIALS)" --threads 1 \
	  --minimum-speedup-percent 2 \
	  --experiment "persistent_codec_workspace" \
	  --comparison-contract "same source, equations, compression parameters, strided brick views, brick order, codec, I/O, history update, config, and threads; comparator recreates ZFP stream/field controls per field and reopens the bitstream per brick instead of retaining a writer-lifetime high-water workspace" \
	  --accept-lifecycle-only \
	  --zfp-fixture-output "$(ZFP_WORKSPACE_CROSS_READER_FIXTURE)" \
	  --output "reports/benchmarks/zfp-codec-workspace-current.json"

qualify-zfp-codec-workspace: benchmark-zfp-codec-workspace bin/test_core_zfp_benchmark \
		bin/test_core_output_writer
	./bin/test_core_zfp_benchmark
	./bin/test_core_output_writer
	cd ui/src-tauri && cargo test zfp3d --lib
	cd ui/src-tauri && \
	  NIMBUS_TEST_ZFP3D="$(abspath $(ZFP_WORKSPACE_CROSS_READER_FIXTURE))" \
	  cargo test real_zfp3d_decode --lib -- --ignored
	python3 -m unittest tests.integration.test_zfp_history_working_set_benchmark

check-cpu-benchmark:
	@test -n "$(BENCH_BASELINE)" -a -n "$(BENCH_POLICY)" || { echo "Set BENCH_BASELINE and BENCH_POLICY to reviewed external inputs."; exit 1; }
	python3 scripts/benchmarks/cpu/check_cpu_benchmark.py \
	  --baseline "$(BENCH_BASELINE)" --current "$(BENCH_REPORT)" \
	  --policy "$(BENCH_POLICY)" $(if $(strip $(BENCH_COMPARISON)),--output "$(BENCH_COMPARISON)",)
