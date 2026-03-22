Run performance benchmarks. Steps:
1. Build with full optimizations: `make clean && make -j$(sysctl -n hw.ncpu)`
2. Run a short simulation: `./bin/tornado_sim --headless --config=configs/lp.yaml --duration=60`
3. Report: wall-clock time, memory usage, timesteps completed, timesteps/second
4. If `$ARGUMENTS` specifies a module or kernel, focus the benchmark there
5. Compare against previous results in data/benchmark/ if available