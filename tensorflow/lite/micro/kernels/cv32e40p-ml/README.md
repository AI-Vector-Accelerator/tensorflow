# Building TFL Micro for the ML Accelerated CV32E40P

This README contains brief instructions on how to Build TFL Micro for the CV32E40P, both accelerated and standard.
It does not contain a description of the changes needed to add these capabilities to the codebase.
This information has been encoded in the commit messages that accompany those changes. 
Look for commits made in November 2020 by author pa4g17 starting with commit `0238b4bc70b7c39dc6589a5f963f199216863708`

These instructions assume you have a version of GCC installed that supports assembly of RISC-V Vector ISA instructions.

## Make Flags
The command to build TFL Micro for the standard ISA CV32E40P is as follows:   
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=mcu_riscv TARGET_ARCH=cv32e40p
```
The command to build TFL Micro for the ML Accelerated CV32E40P is as follows:
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=mcu_riscv TARGET_ARCH=cv32e40p TAGS="cv32e40p-ml"
```
When run at the top-level directory of this repository these commands will build the archive file `libtesorflow-microlite.a` required by TinyMLPerf. 
The file will be placed in the directory `tensorflow/lite/micro/tools/make/gen/mcu_riscv_cv32e40p/lib/`  
Below is a short explanation of the flags:

### TARGET
This flag is used to specify toolchain settings. It is used to set the GCC prefix, `riscv32-unkown-elf-` in this case, and any compilation or linking flags that will be the same across all hardware variations that use this toolchain.  
In reality, it is used to select a makefile.inc file that is included into the top-level makefile. In this case the mcu_riscv_makefile.inc file is included. 
This file can be found in the directory `tensorflow/lite/micro/tools/make/targets/`

### TARGET_ARCH
Used to specify any hardware specific compilation and linking options.
In reality, this flag is evaluated in the included makefile.inc file and compilation/linking flags are set depending on the outcomes. 

### TAGS 
Used in a variety of ways throughout the build system. 
It's core use is to tell TFL Micro to use a specific optimized version of it's kernel over the less-efficient but portable reference implementation.
This flag is used to specify that the user wants to use the kernel optimized for the ML Accelerated CV32E40P.
The quotation marks are used as more than one TAG can be included.
All TAGS should be separated with a space within the quotation marks.

## Adding New Optimizations

Any TFL Micro source file can be replaced with a platform-optimized version using the TAGS make flag. 
The build system will look in every source directory for subdirectories with names matching any string specified in the TAGS flag.  

If a subdirectory is found it will be searched for any `.cc` file with a matching name to one in the directory above if one is found, it replaces the file in the directory above at compile time. For example, the file `tensorflow/lite/micro/kernels/conv.cc` will be replaced with the file `tensorflow/lite/micro/kernels/cv32e40p-ml/conv.cc` if TAGS contains the string `cv32e40p-ml`. This allows a user to add optimized versions of source files without overwriting the existing portable versions. 

To add a new optimized kernel operation follow the steps below:  

1. Navigate to the directory `tensorflow/lite/micro/kernels`. 
2. Locate the `.cc` file containing the implementation of the operation you would like to optimize and copy it into the `./cv32e40p-ml` directory.
3. Replace all calls to functions in the `reference_ops` namespace with the code implemented within those functions.  
This ensures you are not making edits to the reference implementation of that operation. The reference implementation should remain portable. 
4. Make incremental changes to the implementation. Do not change the function signatures.
5. Run the unit test suite for the operation after each incremental edit.
This process is described in the following section.

## Running Unit Tests

Each `<op>.cc` in the `tensorflow/lite/micro/kernels` directory has an accompanying `<op>_test.cc` file containing unit tests for that operation. This file can be compiled an run to check that that operation still functions. 

Running the following command will instruct TFL Micro to build and run all unit tests:
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=mcu_riscv TARGET_ARCH=cv32e40p TAGS="cv32e40p-ml" test
```
Any unit test binaries produced in this process will end up in the directory `tensorflow/lite/micro/tools/make/gen/mcu_riscv_cv32e40p/bin/`.
The unit test system seems to assume that unit tests are being compiled for the host architecture so it tries to run them.
This, of course, fails as the ELF file produced is for the wrong architecture.
A quirk of this system makes TFL Micro build and run each unit test one at a time, rather than building all of them then running all of them. 
This means that only the first unit test will be built.

A work-around for this is shown in the top-level makefile at `tensorflow/lite/micro/tools/make/Makefile` on lines 226-234.
This work-around was present at the time of writing so you may need to rewind to the commit this makefile was added in to see it in action.
Normally the variable `MICROLITE_TEST_SRCS` contains all of the unit tests. 
To ensure the unit test we want to run is compiled first we remove all other unit tests, e.g:
```makefile
MICROLITE_TEST_SRCS := \
$(wildcard tensorflow/lite/micro/kernels/conv_test.cc) \
#$(wildcard tensorflow/lite/micro/*test.cc) \
#$(wildcard tensorflow/lite/micro/kernels/*test.cc) \
#$(wildcard tensorflow/lite/micro/memory_planner/*test.cc)
```

Unit test sources are also defined using the variable `MICRO_LITE_EXAMPLE_TESTS`. 
These are the unit tests for the example programs. 
The definition of this variable can simply be commented out. 

```makefile
#MICRO_LITE_EXAMPLE_TESTS := $(shell find tensorflow/lite/micro/examples/ -maxdepth 2 -name Makefile.inc)
#MICRO_LITE_EXAMPLE_TESTS += $(shell find tensorflow/lite/micro/examples/ -name Makefile_internal.inc)
```

With these changes made the following command from the top-level directory of this repository will build the convolution unit tests.
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=mcu_riscv TARGET_ARCH=cv32e40p TAGS="cv32e40p-ml" test
```

The unit test can be selected by swapping `conv_test.cc` for another `<op>_test.cc` file in the definition of `MICROLITE_TEST_SRCS`

## Running Tests on Spike

By default, unit tests built with the make flags `TARGET=mcu_riscv TARGET_ARCH=cv32e40p` will not run correctly on the Spike ISA sim. 
Specifying `TARGET_ARCH=cv32e40p` builds the test for running bare-metal on the CV32E40P. 
However, Spike accompanied by pk requires programs to be built for running on Linux. 

To build a test to run on Spike run the following command:
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=mcu_riscv TARGET_ARCH=cv32e40p TAGS="cv32e40p-ml spike" -j18 test
```

The addition of the `spike` tag to the `TAGS` flag tells the build system not to include the linking options that enable support for the CV32E40P. 
Without that information, the build system defaults to building for Linux. 
The resulting ELF file can be run on Spike as shown below for the conv unit test:
```
spike --isa=RV32IMCV --varch=vlen:32,elen:32,slen:32 /opt/riscv/riscv32-unknown-elf/bin/pk  kernel_conv_test
```