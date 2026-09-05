MAKEFLAGS += -j$(shell nproc)

CXX ?= g++

BUILD := build
BIN   := $(BUILD)/teleop-walking-benchmark

MUJOCO ?= /opt/mujoco
CUDA   ?= /usr/local/cuda
EIGEN  ?= /usr/include/eigen3
FREETYPE ?= /usr/include/freetype2
TRT    ?= $(HOME)/TensorRT

CXXFLAGS := -std=c++20 -O2 -DNDEBUG -Wall -Wextra -MMD -MP \
            -D_GLIBCXX_USE_CXX11_ABI=1
INCLUDES := -I. -isystem $(EIGEN) -isystem $(FREETYPE) \
            -isystem $(MUJOCO)/include \
            -isystem $(TRT)/include -isystem $(CUDA)/include
LDFLAGS  := -L$(MUJOCO)/lib -L$(TRT)/lib -L$(CUDA)/lib64 \
            -Wl,--disable-new-dtags \
            -Wl,-rpath,$(MUJOCO)/lib:$(TRT)/lib:$(CUDA)/lib64
LDLIBS   := -lmujoco -lnvinfer -lnvonnxparser -lcudart -lpthread -lfreetype \
            -lglfw -lGL

SRCS := main.cpp
POLICIES := $(wildcard policies/*/*.cpp)
OBJS := $(patsubst %.cpp,$(BUILD)/%.o,$(SRCS))

.PHONY: all clean info table preview
.DEFAULT_GOAL := all

all: $(BIN)

TABLE := $(BUILD)/table
PREVIEW := $(BUILD)/preview

table: $(TABLE)

preview: $(PREVIEW)

$(TABLE): table.cpp | $(BUILD)
	@echo "CXX $<"
	@$(CXX) -std=c++20 -O2 -DNDEBUG -Wall -Wextra $< -o $@

$(PREVIEW): preview.cpp | $(BUILD)
	@echo "CXX $<"
	@$(CXX) -std=c++20 -O2 -DNDEBUG -Wall -Wextra $< -o $@

info:
	@echo "mujoco:   $(MUJOCO)"
	@echo "tensorrt: $(TRT)"
	@echo "cuda:     $(CUDA)"
	@echo "eigen:    $(EIGEN)"
	@echo "policies: $(words $(POLICIES)) files"

$(BUILD):
	@mkdir -p $@

FLAGS_STAMP := $(BUILD)/flags
$(shell mkdir -p $(BUILD); \
        printf '%s' "$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LDLIBS)" \
        | cmp -s - $(FLAGS_STAMP) \
        || printf '%s' "$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LDLIBS)" > $(FLAGS_STAMP))

$(BUILD)/%.o: %.cpp $(POLICIES) $(FLAGS_STAMP) | $(BUILD)
	@mkdir -p $(dir $@)
	@echo "CXX $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN): $(OBJS)
	@echo "LD  $@"
	@$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d)
