BUILD_FINGERPRINT ?= build/metadata/build-fingerprint.json

.PHONY: build-fingerprint print-build-fingerprint \
	sanitize-address sanitize-undefined sanitize-address-undefined \
	test-sanitize-address test-sanitize-undefined

build-fingerprint:
	@python3 scripts/dev/build/write_build_fingerprint.py \
		--root "$(CURDIR)" --output "$(BUILD_FINGERPRINT)" \
		--compiler "$(CXX)" --cache-mode "$(COMPILER_CACHE)" \
		--cache "$(CACHE_PROGRAM)" \
		--cxxflags "$(CXXFLAGS) $(METAL_MAIN_FLAGS) $(MAIN_ASSERT_FLAGS) $(CPPFLAGS)" \
		--ldflags "$(LDFLAGS) $(LDLIBS) $(METAL_LDLIBS)" \
		--setting CPU_ONLY=$(CPU_ONLY) --setting CHECKED=$(CHECKED) \
		--setting PORTABLE=$(PORTABLE) --setting DISTRIBUTION=$(DISTRIBUTION) \
		--setting LTO=$(LTO) \
		--setting SANITIZE=$(SANITIZE) --setting WARNINGS=$(WARNINGS) \
		--setting WERROR=$(WERROR) --setting EXPORT_NPY=$(EXPORT_NPY) \
		--setting ZFP=$(if $(ZFP),$(ZFP),auto) --setting NETCDF=$(NETCDF) \
		--setting simulator=$(BIN) --setting field_validator=$(FIELD_VALIDATOR) \
		--artifact simulator="$(BIN)" \
		--artifact field_validator="$(FIELD_VALIDATOR)" \
		--dependency openmp_flag="$(OPENMP_FLAG)" \
		--dependency openmp_library="$(OPENMP_LDLIBS)" \
		--dependency vulkan_headers_prefix="$(VULKAN_HEADERS_PREFIX)" \
		--dependency vulkan_sdk="$(VULKAN_SDK)" \
		--dependency metal_flags="$(METAL_MAIN_FLAGS)" \
		--dependency zfp_prefix="$(ZFP_PREFIX)" \
		--dependency zfp_link="$(ZFP_LDLIBS)" \
		--dependency netcdf_config="$(NETCDF_CONFIG)" \
		--dependency netcdf_link="$(NETCDF_LDLIBS)"
	@printf 'Wrote %s\n' "$(BUILD_FINGERPRINT)"

print-build-fingerprint: build-fingerprint
	@cat "$(BUILD_FINGERPRINT)"

sanitize-address:
	$(MAKE) SANITIZE=address CHECKED=1 all

sanitize-undefined:
	$(MAKE) SANITIZE=undefined CHECKED=1 all

sanitize-address-undefined:
	$(MAKE) SANITIZE=address-undefined CHECKED=1 all

test-sanitize-address:
	$(MAKE) SANITIZE=address CHECKED=1 test

test-sanitize-undefined:
	$(MAKE) SANITIZE=undefined CHECKED=1 test
