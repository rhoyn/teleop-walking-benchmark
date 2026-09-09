MAKEFLAGS += -j$(shell nproc)

CXX ?= g++

BUILD := build
BIN   := $(BUILD)/teleop-walking-benchmark

MUJOCO ?= /opt/mujoco
CUDA   ?= /usr/local/cuda
EIGEN  ?= /usr/include/eigen3
FREETYPE ?= /usr/include/freetype2
TRT    ?= $(HOME)/TensorRT

POLICIES := $(wildcard policies/*/*.cpp)

.PHONY: all clean info table physx-sdk
.DEFAULT_GOAL := all

all: $(BIN)

PHYSX_SRC  ?= $(BUILD)/physx-sdk
PHYSX_REPO ?= https://github.com/NVIDIA-Omniverse/PhysX.git
PHYSX_REF  ?= 517a0073715120e114ee055b63b26c95e00d9039
PHYSX_PRESET ?= linux-gcc-nosnip
GPU_SASS   ?= 89
PHYSX  ?= $(PHYSX_SRC)/physx
PXLIB  ?= $(PHYSX)/bin/linux.x86_64/release
PXRPATH := $$ORIGIN/physx-sdk/physx/bin/linux.x86_64/release
NVCC   ?= $(CUDA)/bin/nvcc
GPU_ARCH ?= sm_89
PX_INCLUDES := -isystem $(TRT)/include -isystem $(PHYSX)/include \
               -isystem $(CUDA)/include -isystem $(MUJOCO)/include \
               -isystem $(EIGEN) -isystem $(FREETYPE) -I.
PX_NVFLAGS  := -std=c++20 -O3 -DNDEBUG -arch=$(GPU_ARCH) -DPX_PHYSX_STATIC_LIB \
               $(PX_INCLUDES)
PX_LDLIBS := -L$(PXLIB) -Wl,--start-group \
             -lPhysXExtensions_static_64 -lPhysX_static_64 -lPhysXPvdSDK_static_64 \
             -lPhysXCooking_static_64 -lPhysXCommon_static_64 \
             -lPhysXFoundation_static_64 -Wl,--end-group \
             -L$(TRT)/lib -lnvinfer -lnvonnxparser \
             -L$(MUJOCO)/lib -lmujoco -lglfw -lGL -lfreetype \
             -L$(CUDA)/lib64 -lcuda -lcudart -lpthread -ldl \
             -Wl,--disable-new-dtags -Wl,-rpath,'$(PXRPATH)':$(TRT)/lib:$(CUDA)/lib64:$(MUJOCO)/lib

OBJS := $(BUILD)/main.o $(BUILD)/model.o $(BUILD)/physics_physx.o \
        $(BUILD)/physics_mjwarp.o

$(BUILD)/main.o: main.cpp $(POLICIES) model.h physics.h physics_physx.h Makefile | $(BUILD)
	@echo "NVCC main.cpp"
	@$(NVCC) $(PX_NVFLAGS) -x cu -c main.cpp -o $@

$(BUILD)/model.o: model.cpp model.h Makefile | $(BUILD)
	@echo "NVCC model.cpp"
	@$(NVCC) $(PX_NVFLAGS) -c model.cpp -o $@

$(BUILD)/physics_physx.o: physics_physx.cpp physics_physx.h physics.h model.h Makefile | $(BUILD)
	@echo "NVCC physics_physx.cpp"
	@$(NVCC) $(PX_NVFLAGS) -x cu -c physics_physx.cpp -o $@

$(BUILD)/physics_mjwarp.o: physics_mjwarp.cpp physics.h model.h Makefile | $(BUILD)
	@echo "NVCC physics_mjwarp.cpp"
	@$(NVCC) $(PX_NVFLAGS) -x cu -c physics_mjwarp.cpp -o $@

$(BIN): $(OBJS)
	@echo "LD  $@"
	@$(CXX) $^ -o $@ $(PX_LDLIBS)

physx-sdk:
	test -d $(PHYSX_SRC) || git clone $(PHYSX_REPO) $(PHYSX_SRC)
	git -C $(PHYSX_SRC) checkout --quiet $(PHYSX_REF)
	sed -i 's/GENERATE_ARCH_CODE_LIST(SASS "[^"]*" PTX "[^"]*")/GENERATE_ARCH_CODE_LIST(SASS "$(GPU_SASS)" PTX "$(GPU_SASS)")/' \
		$(PHYSX)/source/compiler/cmakegpu/CMakeLists.txt
	sed -e 's/name="linux-gcc"/name="$(PHYSX_PRESET)"/' \
	    -e 's/\(PX_BUILDSNIPPETS" value="\)True/\1False/' \
	    -e 's/\(PX_BUILDPVDRUNTIME" value="\)True/\1False/' \
	    -e 's|install/linux-gcc/|install/$(PHYSX_PRESET)/|' \
	    $(PHYSX)/buildtools/presets/public/linux-gcc.xml > $(PHYSX)/buildtools/presets/public/$(PHYSX_PRESET).xml
	cd $(PHYSX) && CUDACXX=$(NVCC) PATH=$(CUDA)/bin:$$PATH ./generate_projects.sh $(PHYSX_PRESET)
	$(MAKE) -C $(PHYSX)/compiler/$(PHYSX_PRESET)-release -j$(shell nproc)

MJWARP_NWORLD ?= 1024
MJWARP_GRAPH  ?= $(BUILD)/mjwarp/g1

.PHONY: capture

capture:
	./mjwarp_capture.sh --nworld $(MJWARP_NWORLD) --out $(MJWARP_GRAPH)

TABLE := $(BUILD)/table

table: $(TABLE)

$(TABLE): table.cpp | $(BUILD)
	@echo "CXX $<"
	@$(CXX) -std=c++20 -O2 -DNDEBUG -Wall -Wextra $< -o $@


info:
	@echo "mujoco:   $(MUJOCO)"
	@echo "tensorrt: $(TRT)"
	@echo "cuda:     $(CUDA)"
	@echo "eigen:    $(EIGEN)"
	@echo "physx:    $(PHYSX)"
	@echo "policies: $(words $(POLICIES)) files"
	@echo "mjwarp:   $(MJWARP_GRAPH) (nworld $(MJWARP_NWORLD))"

$(BUILD):
	@mkdir -p $@

clean:
	find $(BUILD) -mindepth 1 -maxdepth 1 ! -name physx-sdk -exec rm -rf {} +
