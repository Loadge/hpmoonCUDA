################################################################################
#
# Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

# Location of the CUDA Toolkit
CUDA_PATH       ?= /usr/local/cuda-7.5

##############################
# start deprecated interface #
##############################
ifeq ($(x86_64),1)
    $(info WARNING - x86_64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=x86_64 instead)
    TARGET_ARCH ?= x86_64
endif
ifeq ($(ARMv7),1)
    $(info WARNING - ARMv7 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=armv7l instead)
    TARGET_ARCH ?= armv7l
endif
ifeq ($(aarch64),1)
    $(info WARNING - aarch64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=aarch64 instead)
    TARGET_ARCH ?= aarch64
endif
ifeq ($(ppc64le),1)
    $(info WARNING - ppc64le variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=ppc64le instead)
    TARGET_ARCH ?= ppc64le
endif
ifneq ($(GCC),)
    $(info WARNING - GCC variable has been deprecated)
    $(info WARNING - please use HOST_COMPILER=$(GCC) instead)
    HOST_COMPILER ?= $(GCC)
endif
ifneq ($(abi),)
    $(error ERROR - abi variable has been removed)
endif
############################
# end deprecated interface #
############################

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
    TARGET_SIZE := 64
else ifeq ($(TARGET_ARCH),armv7l)
    TARGET_SIZE := 32
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
ifeq ($(TARGET_OS),darwin)
    ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
        HOST_COMPILER ?= clang++
    endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        ifeq ($(TARGET_OS),linux)
            HOST_COMPILER ?= arm-linux-gnueabihf-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
        else ifeq ($(TARGET_OS),android)
            HOST_COMPILER ?= arm-linux-androideabi-g++
        endif
    else ifeq ($(TARGET_ARCH),aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g+v+
        else ifeq ($(TARGET_OS), android)
            HOST_COMPILER ?= aarch64-linux-android-g++
        endif
    else ifeq ($(TARGET_ARCH),ppc64le)
        HOST_COMPILER ?= powerpc64le-linux-gnu-g++
    endif
endif
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     :=

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
    LDFLAGS += -pie
    CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
    endif
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDESCUDA  := -I../../common/inc 
# Includes for hpmoonCUDA
INCLUDES = $(INC)/tinyxml2.h $(INC)/xml.h $(INC)/bd.h $(INC)/initialization.h $(INC)/evaluation.h $(INC)/sort.h $(INC)/tournament.h $(INC)/crossover.h
OBJECTS = $(OBJ)/tinyxml2.o $(OBJ)/xml.o $(OBJ)/bd.o $(OBJ)/individual.o $(OBJ)/initialization.o $(OBJ)/evaluation.o $(OBJ)/sort.o $(OBJ)/tournament.o $(OBJ)/crossover.o $(OBJ)/main.o
LIBRARIES :=

################################################################################

UBUNTU = $(shell lsb_release -i -s 2>/dev/null | grep -i ubuntu)

PROGRAM_ENABLED := 1

PTX_FILE := matrixMul_kernel${TARGET_SIZE}.ptx

# Gencode arguments
SMS ?=

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ifeq ($(SMS),)
# Generate PTX code from SM 20
GENCODE_FLAGS += -gencode arch=compute_20,code=compute_20
endif

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(TARGET_OS),darwin)
  ALL_LDFLAGS += -Xlinker -framework -Xlinker CUDA
else
  CUDA_SEARCH_PATH ?=
  ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
      ifneq ($(TARGET_FS),)
        LIBRARIES += -L$(TARGET_FS)/usr/lib
        CUDA_SEARCH_PATH += $(TARGET_FS)/usr/lib
        CUDA_SEARCH_PATH += $(TARGET_FS)/usr/lib/arm-linux-gnueabihf
      endif
      CUDA_SEARCH_PATH += $(CUDA_PATH)/targets/armv7-linux-gnueabihf/lib/stubs
      CUDA_SEARCH_PATH += /usr/arm-linux-gnueabihf/lib
    else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-android)
      CUDA_SEARCH_PATH += $(CUDA_PATH)/targets/armv7-linux-androideabi/lib/stubs
    else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-android)
      CUDA_SEARCH_PATH += $(CUDA_PATH)/targets/aarch64-linux-androideabi/lib/stubs
    else ifeq ($(TARGET_ARCH)-$(TARGET_OS),ppc64le-linux)
      CUDA_SEARCH_PATH += $(CUDA_PATH)/targets/ppc64le-linux/lib/stubs
    endif
  else
    ifneq ($(UBUNTU),)
      CUDA_SEARCH_PATH += /usr/lib
    else
      CUDA_SEARCH_PATH += /usr/lib64
    endif

    ifeq ($(TARGET_ARCH),x86_64)
      CUDA_SEARCH_PATH += $(CUDA_PATH)/lib64/stubs
    endif

    ifeq ($(TARGET_ARCH),armv7l)
      CUDA_SEARCH_PATH += $(CUDA_PATH)/targets/armv7-linux-gnueabihf/lib/stubs
      CUDA_SEARCH_PATH += /usr/lib/arm-linux-gnueabihf
    endif

    ifeq ($(TARGET_ARCH),aarch64)
      CUDA_SEARCH_PATH += /usr/lib
    endif

    ifeq ($(TARGET_ARCH),ppc64le)
      CUDA_SEARCH_PATH += /usr/lib/powerpc64le-linux-gnu
    endif
  endif

  CUDALIB ?= $(shell find -L $(CUDA_SEARCH_PATH) -maxdepth 1 -name libcuda.so)
  ifeq ("$(CUDALIB)","")
    $(info >>> WARNING - libcuda.so not found, CUDA Driver is not installed.  Please re-install the driver. <<<)
    PROGRAM_ENABLED := 0
  endif

  LIBRARIES += -lcuda
endif

ifeq ($(PROGRAM_ENABLED),0)
EXEC ?= @echo "[@]"
endif

# Custom hpmoon files
SRC = src
OBJ = obj
BIN = bin
LIB = lib
DOC = doc
INC  = include
GNUPLOT = gnuplot

CXXFLAGS = -c -Iinclude 
OPT = -O3

POPSIZE = -D POPULATION_SIZE=$(POPULATION_SIZE)
NFEATURES = -D N_FEATURES=$(N_FEATURES)
NOBJECTIVES = -D N_OBJECTIVES=$(N_OBJECTIVES)
NINSTANCES = -D N_INSTANCES=$(N_INSTANCES)
VARS = $(POPSIZE) $(NFEATURES) $(NOBJECTIVES) $(NINSTANCES)

################################################################################


# Target rules
all: build

build: $(LIB)/libhv.a $(BIN)/hpmoonCUDA #vectorAdd

# Documentation
documentation:
	doxygen $(DOC)/Doxyfile

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Program will be waived due to the above missing dependencies"
else
	@echo "Program is ready - all dependencies have been met"
endif

#Compilation of modules

$(OBJ)/tinyxml2.o: $(SRC)/tinyxml2.cpp $(INC)/tinyxml2.h
	@echo "Making tinyxml2.o"
	#g++ $(CXXFLAGS) $(OPT) $(SRC)/tinyxml2.cu -o $(OBJ)/tinyxml2.o
	$(EXEC) $(NVCC) $(INCLUDESCUDA) $(CXXFLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(OPT) $(SRC)/tinyxml2.cpp -o $(OBJ)/tinyxml2.o

$(OBJ)/xml.o: $(SRC)/xml.cpp $(INC)/xml.h
	@echo "Making xml.o"
	$(EXEC) $(NVCC) $(INCLUDESCUDA) $(CXXFLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(OPT) $(SRC)/xml.cpp -o $(OBJ)/xml.o
$(OBJ)/bd.o: $(SRC)/bd.cpp $(INC)/bd.h	
	@echo "Making bd.o"
	$(EXEC) $(NVCC) $(INCLUDESCUDA) $(CXXFLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(OPT) $(SRC)/bd.cu -o $(OBJ)/bd.o
$(OBJ)/individual.o: $(SRC)/individual.cpp $(INC)/individual.h
	@echo "Making individual.o"
	$(EXEC) $(NVCC) $(INCLUDESCUDA) $(CXXFLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(OPT) $(NFEATURES) $(NOBJECTIVES) $(SRC)/individual.cpp -o $(OBJ)/individual.o
$(OBJ)/initialization.o: $(SRC)/initialization.cu $(INC)/initialization.h $(INC)/individual.h
	@echo "Making initialization.o"
	$(EXEC) $(NVCC) $(INCLUDESCUDA) $(CXXFLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(OPT) $(NFEATURES) $(NOBJECTIVES)  $(SRC)/initialization.cu -o $(OBJ)/initialization.o
$(OBJ)/evaluation.o: $(SRC)/evaluation.cu $(INC)/evaluation.h $(INC)/individual.h $(INC)/hv.h
	@echo "Making evaluation.o"
	$(EXEC) $(NVCC) $(INCLUDESCUDA) $(CXXFLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(OPT) $(NFEATURES) $(NOBJECTIVES) $(NINSTANCES) $(SRC)/evaluation.cu -o $(OBJ)/evaluation.o
$(OBJ)/sort.o: $(SRC)/sort.cu $(INC)/sort.h $(INC)/individual.h
	@echo "Making sort.o"
	$(EXEC) $(NVCC) $(INCLUDESCUDA) $(CXXFLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(OPT) $(NFEATURES) $(NOBJECTIVES) $(SRC)/sort.cu -o $(OBJ)/sort.o
$(OBJ)/tournament.o: $(SRC)/tournament.cu $(INC)/tournament.h
	@echo "Making tournament.o"
	$(EXEC) $(NVCC) $(INCLUDESCUDA) $(CXXFLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(OPT) $(SRC)/tournament.cu -o $(OBJ)/tournament.o
$(OBJ)/crossover.o: $(SRC)/crossover.cu $(INC)/crossover.h $(INC)/individual.h
	@echo "Making crossover.o"
	$(EXEC) $(NVCC) $(INCLUDESCUDA) $(CXXFLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(OPT) $(NFEATURES) $(NOBJECTIVES) $(SRC)/crossover.cu -o $(OBJ)/crossover.o

$(OBJ)/main.o: $(SRC)/main.cu $(INCLUDES)
	@echo "Making obj/main.o"
	$(EXEC) $(NVCC) $(INCLUDESCUDA) $(CXXFLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(OPT) $(VARS) $(SRC)/main.cu -o $(OBJ)/main.o

#Necessary library
$(LIB)/libhv.a:
	@echo "Making libhv.a"
	@cd hv-1.3-Fonseca; $(MAKE) -s
	@\cp hv-1.3-Fonseca/libhv.a $(LIB)

#Linking and creating executable
$(BIN)/hpmoonCUDA: $(OBJECTS) $(LIB)/libhv.a
	@echo "Making bin/hpmoonCUDA"
	#$(HOST_COMPILER) $(OBJECTS) -o $(BIN)/hpmoonCUDA -Llib -lhv
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(OBJECTS) -o $(BIN)/hpmoonCUDA -Llib -lhv









#################################
#vectorAdd.o:vectorAdd.cu
#	@echo "Making vectorAdd.o"
#	$(EXEC) $(NVCC) $(INCLUDESCUDA) $(CXXFLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
#
#vectorAdd: vectorAdd.o
#	@echo "Making vectorAdd"
#	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
#	$(EXEC) mkdir -p /bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
#	$(EXEC) cp $@ /bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
#################################
#run: build
#	@echo "Running vectorAdd"
#	$(EXEC) ./vectorAdd
#
clean:
	@echo "Partial cleaning..."
	@printf "\t- Binary files\n"
	@printf "\t- .o files\n"
	@printf "\t- ~ files\n"
	@printf "\t- .a libraries\n"
	@printf "\t- Hypervolume project of Fonseca\n"
	@\rm -rf $(OBJ)/* $(BIN)/* $(LIB)/* *~
	@cd hv-1.3-Fonseca; $(MAKE) -s clean

eraseAll: clean
	@echo "Additionally..."
	@printf "\t- gnuplot files\n"
	@printf "\t- Documentation files\n"
	@\rm -rf $(GNUPLOT)/*
	@\rm -rf $(DOC)/html/* $(DOC)/*.db