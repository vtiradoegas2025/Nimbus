#include <windows.h>

#include <cstdlib>

int main()
{
    if (std::getenv("NIMBUS_TEST_PROCESS_WAIT"))
        Sleep(INFINITE);
    const char* code = std::getenv("NIMBUS_TEST_PROCESS_EXIT");
    return code ? std::atoi(code) : 0;
}
