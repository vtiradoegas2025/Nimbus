# ---------------------------------------------------------------------------
# Developer tooling: compile DB + formatting (modularity cleanup findings 8, 9)
# ---------------------------------------------------------------------------

# Regenerate compile_commands.json from the live build flags so clangd /
# clang-tidy / IDEs stay accurate. Self-maintaining: it globs src/, vulkan/src/,
# and tests/ (test TUs get the test recipe's -O0 -I tests), so new files are
# picked up automatically and the DB never drifts the way the old one did.
.PHONY: compile-commands clang-dead-code generate-public-configs check-public-configs check-docs

check-docs:
	PYTHONDONTWRITEBYTECODE=1 python3 scripts/tests/architecture/check_docs.py

generate-public-configs:
	python3 scripts/dev/configs/generate_resolution_catalog.py

check-public-configs:
	python3 scripts/dev/configs/generate_resolution_catalog.py --check
	python3 -m unittest tests/integration/test_public_config_catalog.py

compile-commands:
	@python3 scripts/dev/build/gen_compile_commands.py "$(CXX)" "$(CXXFLAGS)" "$(TEST_CXXFLAGS)" "$(CURDIR)" > compile_commands.json
	@echo "Wrote $$(grep -c '\"file\"' compile_commands.json) entries to compile_commands.json"

# Run a strict, backend-only Clang analyzer pass. The wrapper locates either a
# PATH or Homebrew LLVM installation and supplies Xcode's SDK/libc++ paths on
# macOS, where Homebrew clang-tidy cannot otherwise parse Apple C++ headers.
CLANG_TIDY_JOBS ?= 8
clang-dead-code: compile-commands
	@python3 scripts/dev/build/run_clang_dead_code.py --root "$(CURDIR)" --jobs "$(CLANG_TIDY_JOBS)"

# clang-format applied only to the lines changed vs FORMAT_BASE (via
# git-clang-format), so the existing tree is cleaned gradually as files are
# touched rather than in one big-bang reformat. `make format` rewrites in place;
# `make format-check` is the read-only CI gate (used by .github/workflows/ci.yml,
# which passes the PR base as FORMAT_BASE).
CLANG_FORMAT     ?= clang-format
GIT_CLANG_FORMAT ?= git-clang-format
FORMAT_BASE      ?= origin/main
FORMAT_EXTS      := cpp,hpp,h,cc,cxx
FORMAT_BASE_COMMAND = if [ "$(FORMAT_BASE)" = "empty-tree" ]; then git hash-object -w -t tree /dev/null; else git merge-base HEAD "$(FORMAT_BASE)"; fi

format:
	@command -v $(GIT_CLANG_FORMAT) >/dev/null 2>&1 || { \
	  echo "git-clang-format not found; install clang-format / LLVM tools"; exit 1; }
	@base=$$($(FORMAT_BASE_COMMAND)) || exit 1; \
	 $(GIT_CLANG_FORMAT) --binary $(CLANG_FORMAT) --extensions $(FORMAT_EXTS) --commit "$$base"

format-check:
	@command -v $(GIT_CLANG_FORMAT) >/dev/null 2>&1 || { \
	  echo "git-clang-format not found; install clang-format / LLVM tools"; exit 1; }
	@base=$$($(FORMAT_BASE_COMMAND)) || exit 1; \
	 if ! diff=$$($(GIT_CLANG_FORMAT) --binary $(CLANG_FORMAT) --extensions $(FORMAT_EXTS) --diff --commit "$$base"); then \
	   printf '%s\n' "$$diff"; echo "Formatting check failed; inspect the output above."; exit 1; \
	 fi; \
	 if [ "$$diff" = "no modified files to format" ] || \
	    [ "$$diff" = "clang-format did not modify any files" ] || [ -z "$$diff" ]; then \
	   echo "Formatting OK (changed lines vs $(FORMAT_BASE))."; \
	 else \
	   printf '%s\n' "$$diff"; \
	   echo "Formatting issues on changed lines. Run 'make format'."; exit 1; \
	 fi

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean-vulkan:
	$(MAKE) -C vulkan clean

clean:
	rm -rf $(BUILDDIR) $(ACTIVE_PHYSICS_OPT_BUILDDIR) \
		$(NESTED_RELOCATION_BENCHMARK_BUILDDIR) \
		$(ACTIVE_PHYSICS_LEGACY_BUILDDIR) $(SMAG_FILTER_LEGACY_BUILDDIR) \
		$(SGS_VERTICAL_STRESS_LEGACY_BUILDDIR) \
		$(ZFP_HISTORY_COMPARATOR_BUILDDIR) \
		$(ZFP_BRICK_GATHER_COMPARATOR_BUILDDIR) \
		$(ZFP_TRANSIENT_WORKSPACE_COMPARATOR_BUILDDIR) \
		$(ZFP_WORKSPACE_PHASE_BUILDDIR) \
		$(VULKAN_SCALAR_ATTRIBUTION_COMPARATOR_BUILDDIR) \
		$(VULKAN_SCALAR_REGIONS_EXPERIMENT_BUILDDIR) \
		$(VULKAN_NOOP_BUDGET_COMPARATOR_BUILDDIR) \
		$(VULKAN_ENDPOINT_MOISTURE_COMPARATOR_BUILDDIR) \
		$(VULKAN_MICROPHYSICS_GATE_COMPARATOR_BUILDDIR) \
		$(VULKAN_MICROPHYSICS_VALIDATION_COMPARATOR_BUILDDIR) \
		$(VULKAN_MICROPHYSICS_INPUT_VALIDATION_COMPARATOR_BUILDDIR)
	rm -f $(BIN) $(BENCHMARK_BIN) $(ACTIVE_PHYSICS_OPT_BIN) \
		$(NESTED_RELOCATION_BENCHMARK_BIN) \
		$(ACTIVE_PHYSICS_LEGACY_BIN) $(SMAG_FILTER_LEGACY_BIN) \
		$(SGS_VERTICAL_STRESS_LEGACY_BIN) \
		$(ZFP_HISTORY_COMPARATOR_BIN) \
		$(ZFP_BRICK_GATHER_COMPARATOR_BIN) \
		$(ZFP_TRANSIENT_WORKSPACE_COMPARATOR_BIN) \
		$(ZFP_WORKSPACE_PHASE_BIN) \
		$(VULKAN_SCALAR_ATTRIBUTION_COMPARATOR_BIN) \
		$(VULKAN_SCALAR_REGIONS_EXPERIMENT_BIN) \
		$(VULKAN_NOOP_BUDGET_COMPARATOR_BIN) \
		$(VULKAN_ENDPOINT_MOISTURE_COMPARATOR_BIN) \
		$(VULKAN_MICROPHYSICS_GATE_COMPARATOR_BIN) \
		$(VULKAN_MICROPHYSICS_VALIDATION_COMPARATOR_BIN) \
		$(VULKAN_MICROPHYSICS_INPUT_VALIDATION_COMPARATOR_BIN) \
		$(FIELD_VALIDATOR) $(CATCH2_BINS) $(BUILD_FINGERPRINT)
	@$(MAKE) -C vulkan clean >/dev/null 2>&1 || true
