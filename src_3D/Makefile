NVCC ?= nvcc
CXX ?= c++
CUDA_ARCH ?= sm_80
DEBUG ?= 0
BUILD := build/$(CUDA_ARCH)/debug-$(DEBUG)
TARGET := $(BUILD)/fluid3d
OBJECTS := $(BUILD)/main.o $(BUILD)/simulation.o $(BUILD)/output.o
FLAGS := -std=c++17 -arch=$(CUDA_ARCH) -Iinclude -Xcompiler=-Wall,-Wextra
ifeq ($(DEBUG),1)
FLAGS += -O0 -g -G
else
FLAGS += -O2 -lineinfo
endif
OUTPUT ?= ../results/3D
STEPS ?= 50
OUTPUT_EVERY ?= 1
.PHONY: all help device-info check run disturbance rest memcheck host-check clean
all: $(TARGET)
$(TARGET): $(OBJECTS)
	$(NVCC) $(FLAGS) $^ -o $@
$(BUILD)/%.o: %.cu
	@mkdir -p $(BUILD)
	$(NVCC) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/%.o: %.cpp
	@mkdir -p $(BUILD)
	$(NVCC) $(FLAGS) -MMD -MP -c $< -o $@
help: $(TARGET)
	./$(TARGET) --help
device-info: $(TARGET)
	./$(TARGET) --device-info
check: $(TARGET)
	./$(TARGET) --check
run disturbance: $(TARGET)
	./$(TARGET) --disturbance --steps $(STEPS) --output-every $(OUTPUT_EVERY) --output "$(OUTPUT)"
rest: $(TARGET)
	./$(TARGET) --rest --steps $(STEPS) --output-every $(OUTPUT_EVERY) --output "$(OUTPUT)"
memcheck: $(TARGET)
	compute-sanitizer --tool memcheck --error-exitcode 1 ./$(TARGET) --check
host-check:
	@mkdir -p build/host
	$(CXX) -std=c++17 -Wall -Wextra -Iinclude tests/host_checks.cpp output.cpp -o build/host/check
	./build/host/check
	python3 tests/check_vti.py build/host/sample.vti
clean:
	rm -rf build
-include $(OBJECTS:.o=.d)
