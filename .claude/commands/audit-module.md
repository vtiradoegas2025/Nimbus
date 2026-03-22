Perform a detailed audit of the specified module ($ARGUMENTS). Read all source and header files in the module. Report:

1. **Implementation status**: What's fully implemented vs stubbed/placeholder
2. **Code quality**: Duplicated logic, missing error handling, dead code, magic numbers
3. **Header/impl consistency**: Do public headers match actual implementations?
4. **TODO/FIXME/HACK comments**: List all with file and line
5. **Dependencies**: What does this module depend on? Are there circular dependencies?
6. **Performance concerns**: Unparallelized hot loops, unnecessary allocations, poor cache access patterns
7. **Suggested improvements**: Prioritized by impact, with estimated effort

If no module is specified, ask which one: core, microphysics, boundary_layer, turbulence, radiation, radar, terrain, chaos, soundings, numerics, advection, validation, vulkan