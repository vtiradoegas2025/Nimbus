#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

int main(int argc, char** argv)
{
    std::string mode;
    std::string counter_path;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument(argv[i]);
        if (argument.rfind("--fixture=", 0) == 0)
            mode = argument.substr(10);
        else if (argument.rfind("--counter=", 0) == 0)
            counter_path = argument.substr(10);
    }

    const char* scope = "test_scope";
    if (mode == "environment")
    {
        const char* threads = std::getenv("OMP_NUM_THREADS");
        const char* binding = std::getenv("OMP_PROC_BIND");
        if (std::getenv("NIMBUS_DEBUG_EXPORTS") || std::getenv("OMP_THREAD_LIMIT") || !threads ||
            std::string(threads) != "3" || !binding || std::string(binding) != "close")
        {
            std::fputs("Inherited environment leaked or controlled tuning is missing\n", stderr);
            return 1;
        }
        std::puts("[MEMORY MILESTONE] phase=startup_ready, peak_rss_bytes=123456");
        std::puts("[MEMORY PLAN] cells=1024, field_bytes=4096, predicted_fields=40000, "
                  "fixed_overhead_bytes=0, predicted_bytes=163840000");
    }
    else if (mode == "heterogeneous" && !counter_path.empty())
    {
        int value = 0;
        {
            std::ifstream input(counter_path);
            if (input.is_open() && !(input >> value))
                return 2;
        }
        std::ofstream output(counter_path);
        output << ++value;
        output.close();
        if (!output)
            return 3;
        scope = value == 1 ? "scope_a" : "scope_b";
    }
    else
    {
        std::fputs("Unknown fixture mode\n", stderr);
        return 4;
    }

    std::printf("[PERF SUMMARY] measured_steps=4, warmup_steps=2, scope=%s, "
                "step_wall_s=2, avg_step_ms=500, sim_s_per_wall_s=3, measured_simulated_s=6\n",
                scope);
    return 0;
}
