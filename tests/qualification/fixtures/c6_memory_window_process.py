#!/usr/bin/env python3
"""Synthetic process-binding fixture; this is not a native Vulkan measurement."""

import json
import os
import sys
import time

if "--hang" in sys.argv:
    time.sleep(30)
if "--flood" in sys.argv:
    print("x" * 100_000)
    raise SystemExit(0)
token = os.environ["NIMBUS_C6_MEMORY_RUN_TOKEN"]
first = int(os.environ["NIMBUS_C6_MEMORY_FIRST_BOUNDARY"])
end = int(os.environ["NIMBUS_C6_MEMORY_END_BOUNDARY_EXCLUSIVE"])
names = ("staging_buffer_extent_bytes", "device_buffer_extent_bytes", "staging_memory_reserved_bytes",
         "device_memory_reserved_bytes", "staging_memory_allocation_count", "device_memory_allocation_count")
for event, boundary in (("begin", first), ("end", end)):
    record = {"schema": "nimbus.c6-vulkan-memory-window-event.v1", "runToken": token, "pid": os.getpid(),
        "owner": "VulkanComputeBackend", "source": "vulkan/include/compute/vulkan_compute_telemetry.hpp",
        "scope": "backend_residency_boundaries", "aggregation": "separate_nonadditive_compartments",
        "recordCapacity": 2, "firstBoundary": first, "endBoundaryExclusive": end, "event": event,
        "boundary": boundary, "completed": event == "end", "valid": True,
        "allocationHookEvents": 0 if event == "begin" else 4,
        "metrics": {name: {"unit": "bytes" if i < 4 else "allocations", "begin": 0, "current": 0,
            "highWater": (128 if i < 4 else 1) if event == "end" else 0} for i, name in enumerate(names)}}
    print("[C6 MEMORY WINDOW] " + json.dumps(record), flush=True)
    if "--abort" in sys.argv:
        break
if "--exit-failure" in sys.argv:
    raise SystemExit(3)
