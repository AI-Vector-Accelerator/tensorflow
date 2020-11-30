# Settings for RISCV 32-bit MCU toolchain.
ifeq ($(TARGET), mcu_riscv)

  ifeq ($(TARGET_ARCH), cv32e40p)
    TARGET_TOOLCHAIN_PREFIX := riscv32-unknown-elf-

  else
    TARGET_ARCH := riscv32_mcu
    TARGET_TOOLCHAIN_PREFIX := riscv64-unknown-elf-
  endif

  $(eval $(call add_third_party_download,$(RISCV_TOOLCHAIN_URL),$(RISCV_TOOLCHAIN_MD5),riscv_toolchain,))
  $(eval $(call add_third_party_download,$(SIFIVE_FE310_LIB_URL),$(SIFIVE_FE310_LIB_MD5),sifive_fe310_lib,))

  PLATFORM_FLAGS = \
    -mabi=ilp32 \
    -mcmodel=medany \
    -mexplicit-relocs \
    -fno-builtin-printf \
    -fno-exceptions \
    -DTF_LITE_MCU_DEBUG_LOG \
    -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS \
    -fno-unwind-tables \
    -ffunction-sections \
    -fdata-sections \
    -funsigned-char \
    -Wvla \
    -Wall \
    -Wextra \
    -Wsign-compare \
    -Wdouble-promotion \
    -Wshadow \
    -Wunused-variable \
    -Wmissing-field-initializers \
    -Wno-unused-parameter \
    -Wno-write-strings \
    -Wunused-function \
    -fno-delete-null-pointer-checks \
    -fomit-frame-pointer \
    -Os


  # Set the extensions used in a particular architecture
  ifeq ($(TARGET_ARCH), cv32e40p)
    PLATFORM_FLAGS += -march=rv32imc \
      -DCV32E40P

  else
    PLATFORM_FLAGS += -march=rv32imac
  endif

  CXXFLAGS += $(PLATFORM_FLAGS) \
    -fpermissive \
    -fno-rtti \
    --std=gnu++11 \
    -fno-use-cxa-atexit \
    -fno-threadsafe-statics 

  CCFLAGS += $(PLATFORM_FLAGS)

  BUILD_TYPE := micro

  INCLUDES += \
    -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/include \
    -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/drivers/ \
    -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env \
    -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/freedom-e300-hifive1

  MICROLITE_CC_SRCS += \
    $(wildcard tensorflow/lite/micro/riscv32_mcu/*.cc)

  # Set the linking instructions for each supported architecture
  ifeq ($(TARGET_ARCH), cv32e40p)

    ifneq (,$(findstring spike,$(TAGS)))
      LDFLAGS += --specs=nano.specs -u _printf_float
    else
      LDFLAGS += -L $(MAKEFILE_DIR)/targets/cv32e40p \
        -l cv-verif \
        -T $(MAKEFILE_DIR)/targets/cv32e40p/link.ld \
        -nostartfiles \
        $(MAKEFILE_DIR)/targets/cv32e40p/crtbegin.o \
        --specs=nano.specs -u _printf_float
    endif

    ifneq (,$(findstring cv32e40p-ml, $(TAGS)))
      MICROLITE_LIB_OBJS += \
        $(MAKEFILE_DIR)/targets/cv32e40p/vector_operations.o \
        $(MAKEFILE_DIR)/targets/cv32e40p/vector_operations_assembly.o 
    endif

  else
    LDFLAGS += \
      -T$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/freedom-e300-hifive1/flash.lds \
      -nostartfiles \
      -L$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env \
      --specs=nano.specs
  endif

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
  OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy
  $(BINDIR)/%.bin: $(BINDIR)/%
		@mkdir -p $(dir $@)
		$(OBJCOPY) $< $@ -O binary
endif
