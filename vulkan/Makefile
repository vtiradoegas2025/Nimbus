CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS += -Iinclude -I../include
UNAME_S := $(shell uname -s)
GLSLANG := $(shell command -v glslangValidator 2>/dev/null)
SHADER_DIR := shaders
SHADER_SPVS := $(SHADER_DIR)/rendering/volume.vert.spv $(SHADER_DIR)/rendering/volume.frag.spv

PKG_CONFIG := $(shell command -v pkg-config 2>/dev/null)
ifeq ($(PKG_CONFIG),)
  VULKAN_CFLAGS :=
  VULKAN_LIBS := -lvulkan
  GLFW_CFLAGS :=
  GLFW_PKG_LIBS :=
else
  VULKAN_CFLAGS := $(shell pkg-config --cflags vulkan 2>/dev/null)
  VULKAN_LIBS := $(shell pkg-config --libs vulkan 2>/dev/null)
  GLFW_CFLAGS := $(shell pkg-config --cflags glfw3 2>/dev/null)
  GLFW_PKG_LIBS := $(shell pkg-config --libs glfw3 2>/dev/null)
  ifeq ($(strip $(VULKAN_LIBS)),)
    VULKAN_LIBS := -lvulkan
  endif
endif

CPPFLAGS += $(VULKAN_CFLAGS)
CPPFLAGS += $(GLFW_CFLAGS)

GLFW_LIBS := $(GLFW_PKG_LIBS)
GLFW_AVAILABLE := 0
ifneq ($(strip $(GLFW_LIBS)),)
  GLFW_AVAILABLE := 1
endif

BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
ifneq ($(BREW_PREFIX),)
  VULKAN_HEADERS_PREFIX := $(shell brew --prefix vulkan-headers 2>/dev/null)
  VULKAN_LOADER_PREFIX := $(shell brew --prefix vulkan-loader 2>/dev/null)
  MOLTENVK_PREFIX := $(shell brew --prefix molten-vk 2>/dev/null)
  SFML_PREFIX := $(shell brew --prefix sfml 2>/dev/null)
  GLFW_PREFIX := $(shell brew --prefix glfw 2>/dev/null)

  ifneq ($(VULKAN_HEADERS_PREFIX),)
    CPPFLAGS += -I$(VULKAN_HEADERS_PREFIX)/include
  endif
  ifneq ($(VULKAN_LOADER_PREFIX),)
    CPPFLAGS += -I$(VULKAN_LOADER_PREFIX)/include
    LDFLAGS += -L$(VULKAN_LOADER_PREFIX)/lib
  endif
  ifneq ($(MOLTENVK_PREFIX),)
    CPPFLAGS += -I$(MOLTENVK_PREFIX)/include
    LDFLAGS += -L$(MOLTENVK_PREFIX)/lib
  endif
  # macOS: embed RPATH so the binary finds Vulkan/MoltenVK at runtime
  ifeq ($(UNAME_S),Darwin)
    ifneq ($(VULKAN_LOADER_PREFIX),)
      LDFLAGS += -Wl,-rpath,$(VULKAN_LOADER_PREFIX)/lib
    endif
    ifneq ($(MOLTENVK_PREFIX),)
      LDFLAGS += -Wl,-rpath,$(MOLTENVK_PREFIX)/lib
    endif
  endif
  ifneq ($(SFML_PREFIX),)
    CPPFLAGS += -I$(SFML_PREFIX)/include
    LDFLAGS += -L$(SFML_PREFIX)/lib
  endif
  ifneq ($(GLFW_PREFIX),)
    ifeq ($(shell test -f $(GLFW_PREFIX)/include/GLFW/glfw3.h && echo yes),yes)
      CPPFLAGS += -I$(GLFW_PREFIX)/include
      LDFLAGS += -L$(GLFW_PREFIX)/lib
      ifeq ($(strip $(GLFW_LIBS)),)
        GLFW_LIBS := -lglfw
      endif
      GLFW_AVAILABLE := 1
    endif
  endif
endif

SRCS := src/core/main.cpp src/core/app.cpp src/core/context.cpp
ifeq ($(UNAME_S),Darwin)
  SFML_LIBS := -lsfml-window -lsfml-system -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
else
  SFML_LIBS := -lsfml-window -lsfml-system -pthread -ldl
endif

ifeq ($(GLFW_AVAILABLE),1)
  CPPFLAGS += -DVKCPP_USE_GLFW=1
else
  CPPFLAGS += -DVKCPP_USE_GLFW=0
endif

SRCS += src/rendering/window_renderer.cpp
SRCS += src/rendering/clear_backend.cpp
SRCS += src/rendering/volume_backend.cpp
SRCS += src/camera/camera_controller.cpp
SRCS += src/camera/camera_input_tracker.cpp
SRCS += src/data/export_dataset.cpp
SRCS += src/data/npy_reader.cpp
SRCS += src/data/shm_dataset.cpp
BIN := ../bin/vulkan_viewer

all: check-deps check-shader-tools $(SHADER_SPVS) $(BIN)

check-deps:
	@printf '#include <vulkan/vulkan.h>\nint main(){return 0;}\n' | \
	$(CXX) -x c++ - $(CPPFLAGS) $(LDFLAGS) $(VULKAN_LIBS) -o /tmp/vulkan_scaffold_check >/dev/null 2>&1 || \
	{ echo "Vulkan headers/loader not found."; \
	  echo "Install dependencies first:"; \
	  echo "  macOS (Homebrew): brew install vulkan-headers vulkan-loader molten-vk"; \
	  exit 1; }
	@rm -f /tmp/vulkan_scaffold_check

check-shader-tools:
	@if [ -z "$(GLSLANG)" ]; then \
		echo "glslangValidator not found."; \
		echo "Install it with Vulkan SDK tools (macOS Homebrew): brew install glslang"; \
		exit 1; \
	fi

$(SHADER_DIR)/rendering/%.vert.spv: $(SHADER_DIR)/rendering/%.vert
	$(GLSLANG) -V -S vert $< -o $@

$(SHADER_DIR)/rendering/%.frag.spv: $(SHADER_DIR)/rendering/%.frag
	$(GLSLANG) -V -S frag $< -o $@

$(SHADER_DIR)/compute/%.comp.spv: $(SHADER_DIR)/compute/%.comp
	$(GLSLANG) -V -S comp $< -o $@

$(BIN): $(SRCS) | ../bin
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(SRCS) $(LDFLAGS) $(VULKAN_LIBS) $(GLFW_LIBS) $(SFML_LIBS) -o $(BIN)

../bin:
	mkdir -p ../bin

clean:
	rm -f $(BIN)
	rm -f $(SHADER_SPVS)

.PHONY: all clean check-deps check-shader-tools
