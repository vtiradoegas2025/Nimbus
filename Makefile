# Nimbus build entry point.
#
# Public targets stay here conceptually, while their implementations are grouped
# by concern under mk/. Keep this include order: later modules consume variables
# and targets declared by earlier ones.

.DEFAULT_GOAL := all

include mk/options.mk
include mk/dependencies.mk

SOURCE_MANIFEST := mk/sources.mk
include $(SOURCE_MANIFEST)

include mk/help.mk
include mk/infrastructure.mk
include mk/build.mk
include mk/workflows.mk
include mk/tests.mk
include mk/benchmarks.mk
include mk/developer.mk
