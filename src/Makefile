NVCC ?= nvcc
# Override for the allocated GPU. Separate output directories prevent stale
# objects when switching architecture or debug mode.
CUDA_ARCH ?= sm_80
DEBUG ?= 0
BUILD_DIR := build/$(CUDA_ARCH)/debug-$(DEBUG)
TARGET := $(BUILD_DIR)/checkpoint
SOURCES := main.cu simulation.cu kernels.cu disturbance.cu
OBJECTS := $(SOURCES:%.cu=$(BUILD_DIR)/%.o)
DEPS := $(OBJECTS:.o=.d)

CPPFLAGS += -Iinclude
NVCCFLAGS += -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra
ifeq ($(DEBUG),1)
NVCCFLAGS += -O0 -g -G
else
NVCCFLAGS += -O2 -lineinfo
endif

.PHONY: all help device-info run simulate disturbance collision-check streaming-check memcheck clean
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -c $< -o $@

help: $(TARGET)
	./$(TARGET) --help

device-info: $(TARGET)
	./$(TARGET) --device-info

run: $(TARGET)
	./$(TARGET) --run

simulate: $(TARGET)
	./$(TARGET) --simulate

disturbance: $(TARGET)
	./$(TARGET) --disturbance

collision-check: $(TARGET)
	./$(TARGET) --collision-check

streaming-check: $(TARGET)
	./$(TARGET) --streaming-check

# Check initialization, isolated updates, and repeated rest-state evolution.
memcheck: $(TARGET)
	compute-sanitizer --tool memcheck --error-exitcode 1 ./$(TARGET) --run
	compute-sanitizer --tool memcheck --error-exitcode 1 ./$(TARGET) --collision-check
	compute-sanitizer --tool memcheck --error-exitcode 1 ./$(TARGET) --streaming-check
	compute-sanitizer --tool memcheck --error-exitcode 1 ./$(TARGET) --simulate

clean:
	rm -rf build

-include $(DEPS)
