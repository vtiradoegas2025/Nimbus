.PHONY: help
help:
	@printf '%s\n' \
	  'Nimbus build commands' \
	  '' \
	  '  make [all]                 Build simulator, validator, and changed shaders' \
	  '  make test                  Run the default backend test aggregate' \
	  '  make test-script-suite     Run fast Python architecture/static checks' \
	  '  make check-deps            Report compiler and optional dependencies' \
	  '  make build-fingerprint     Write build/metadata/build-fingerprint.json' \
	  '  make compile-commands      Regenerate compile_commands.json' \
	  '  make check-public-configs Validate the resolution config matrix' \
	  '  make clean                 Remove artifacts for the selected profile' \
	  '' \
	  'Common profiles' \
	  '  make CPU_ONLY=1            CPU-only native release build' \
	  '  make CPU_ONLY=1 PORTABLE=1 Portable CPU release build' \
	  '  make CHECKED=1             Assertions enabled, isolated objects/binary' \
	  '  make LTO=1                 Link-time-optimized, isolated objects/binary' \
	  '  make sanitize-address      AddressSanitizer checked build' \
	  '  make sanitize-undefined    UndefinedBehaviorSanitizer checked build' \
	  '' \
	  'Controls' \
	  '  CXX=clang++                Select the C++ compiler' \
	  '  WARNINGS=0                 Disable -Wall -Wextra -Wpedantic' \
	  '  WERROR=1                   Promote compiler warnings to errors' \
	  '  COMPILER_CACHE=auto        Prefer sccache, then ccache (default)' \
	  '  COMPILER_CACHE=none        Disable compiler caching' \
	  '  SANITIZE=address|undefined|address-undefined' \
	  '' \
	  'See docs/BUILDING.md for platform setup and qualification targets.'
