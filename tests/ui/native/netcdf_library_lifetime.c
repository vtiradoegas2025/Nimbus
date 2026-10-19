#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static void mark(char event)
{
    const char* path = getenv("NIMBUS_NETCDF_LIFETIME_MARKER");
    if (!path)
        abort();
    FILE* file = fopen(path, "a");
    if (!file)
        abort();
    fputc(event, file);
    fclose(file);
}

__attribute__((constructor)) static void loaded(void)
{
    mark('L');
}
__attribute__((destructor)) static void unloaded(void)
{
    mark('D');
}
static void process_cleanup(void)
{
    mark('E');
}

int nc_open(const char* path, int mode, int* id)
{
    static int registered = 0;
    if (!registered)
    {
        if (atexit(process_cleanup) != 0)
            abort();
        registered = 1;
    }
    mark('O');
    *id = 1;
    return 0;
}
int nc_close(int id)
{
    mark('C');
    return 0;
}
int nc_inq(int id, int* dims, int* vars, int* attributes, int* unlimited)
{
    *dims       = 0;
    *vars       = 0;
    *attributes = 0;
    *unlimited  = -1;
    return 0;
}
int nc_inq_format(int id, int* format)
{
    *format = 3;
    return 0;
}
int nc_inq_dim(int id, int dim, char* name, size_t* length)
{
    return -1;
}
int nc_inq_var(int id, int var, char* name, int* type, int* rank, int* dims, int* attributes)
{
    return -1;
}
int nc_inq_att(int id, int var, const char* name, int* type, size_t* length)
{
    return -1;
}
int nc_get_att_text(int id, int var, const char* name, char* values)
{
    return -1;
}
int nc_get_att_double(int id, int var, const char* name, double* values)
{
    return -1;
}
int nc_get_vara_float(int id, int var, const size_t* start, const size_t* count, float* values)
{
    return -1;
}
int nc_get_vars_float(int id, int var, const size_t* start, const size_t* count, const ptrdiff_t* stride, float* values)
{
    return -1;
}
int nc_get_vars_double(
    int id, int var, const size_t* start, const size_t* count, const ptrdiff_t* stride, double* values)
{
    return -1;
}
int nc_get_var_double(int id, int var, double* values)
{
    return -1;
}
const char* nc_strerror(int status)
{
    return "lifetime fixture";
}
