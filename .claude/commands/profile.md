Profile a simulation run to find performance bottlenecks.

1. Build with debug symbols: `make clean && make -j$(sysctl -n hw.ncpu) EXTRA_CXXFLAGS="-g"`
2. Run a short sim: `./bin/tornado_sim --headless --config=configs/lp.yaml --duration=30`
3. On macOS, use `sample ./bin/tornado_sim <pid> 10` or suggest Instruments workflow
4. Analyze the output: report the top 5-10 hottest functions by CPU time
5. Identify optimization targets — which are in physics kernels vs framework overhead
6. Check OpenMP utilization — is parallelism actually engaged?
7. Suggest concrete optimizations for the hottest paths