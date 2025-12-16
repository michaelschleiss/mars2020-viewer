#!/usr/bin/env python3
"""
Setup script for ARM NEON-optimized Cython BSQ conversion.

This build configuration is specifically tuned for Apple Silicon (M1/M2/M3)
with aggressive compiler optimizations and NEON SIMD support.

Usage:
    python3 setup_cython_neon.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform
import os
import sys

# Detect Apple Silicon
is_apple_silicon = (
    platform.machine() == 'arm64' and
    platform.system() == 'Darwin'
)

if not is_apple_silicon:
    print("=" * 80)
    print("ERROR: bsq_cython_neon only supports Apple Silicon (arm64 macOS)")
    print(f"Current platform: {platform.machine()} on {platform.system()}")
    print("=" * 80)
    print()
    sys.exit(1)

# Get pixi environment paths for OpenMP
pixi_env = os.path.join(os.getcwd(), '.pixi', 'envs', 'default')
omp_include = os.path.join(pixi_env, 'include')
omp_lib = os.path.join(pixi_env, 'lib')

# Check if OpenMP is available
if not os.path.exists(omp_include) or not os.path.exists(omp_lib):
    print("=" * 80)
    print("ERROR: OpenMP not found in pixi environment")
    print("Please run: pixi install")
    print("=" * 80)
    sys.exit(1)

# ARM-specific compiler flags
if is_apple_silicon:
    compile_args = [
        # ============================================================
        # ARM64 / Apple Silicon Optimizations
        # ============================================================

        # Target native CPU (M1/M2/M3/M4)
        '-mcpu=native',              # Use all available ARM instructions
        # NOTE: Don't use -march=native (not supported on Apple ARM)

        # Optimization level
        '-O3',                       # Maximum optimization
        '-flto',                     # Link-time optimization

        # Vectorization
        '-ftree-vectorize',          # Enable auto-vectorization
        '-fvectorize',               # Clang vectorizer
        '-fslp-vectorize',           # Superword-level parallelism

        # Loop optimizations
        '-funroll-loops',            # Unroll loops
        '-fomit-frame-pointer',      # Remove frame pointer (more registers)

        # Fast math (safe for image processing)
        '-ffast-math',               # Fast floating point
        '-fno-math-errno',           # Don't set errno for math functions

        # Memory optimizations
        '-fstrict-aliasing',         # Assume strict aliasing rules
        '-falign-functions=32',      # Align functions for better cache
        '-falign-loops=32',          # Align loops for better cache

        # ARM NEON specific
        '-DUSE_NEON',                # Enable NEON code paths
        # NOTE: No need for -mfpu=neon on Apple Silicon (always available)

        # OpenMP parallelization
        '-Xpreprocessor', '-fopenmp',

        # Debugging (optional, remove for production)
        # '-g',                      # Debug symbols
        # '-fno-omit-frame-pointer', # Keep frame pointer for profiling
    ]

    link_args = [
        '-lomp',                     # Link OpenMP library
        '-flto',                     # Link-time optimization
        # Note: -framework Accelerate is implicit on macOS
    ]

    # Additional defines
    define_macros = [
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ('__ARM_NEON', '1'),         # Enable NEON header
        ('__ARM_FEATURE_UNALIGNED', '1'),  # ARM supports unaligned access
    ]

else:
    raise RuntimeError("unreachable: Apple Silicon check should have exited")

# Extension configuration
extensions = [
    Extension(
        "bsq_cython_neon",
        sources=["bsq_cython_neon.pyx"],
        include_dirs=[
            omp_include,
            np.get_include(),
        ],
        library_dirs=[omp_lib],
        libraries=['omp'],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=define_macros,
        language="c",
    )
]

# Cython compiler directives
compiler_directives = {
    'language_level': "3",       # Python 3 syntax
    'boundscheck': False,        # Disable bounds checking
    'wraparound': False,         # Disable negative indexing
    'cdivision': True,           # C-style division (no zero check)
    'initializedcheck': False,   # Skip initialization checks
    'nonecheck': False,          # Skip None checks
    'overflowcheck': False,      # Skip overflow checks
    'embedsignature': True,      # Embed function signatures
    'binding': False,            # Don't create Python wrappers
    'profile': False,            # Disable profiling
}

# Build
setup(
    name='bsq_cython_neon',
    version='1.0.0',
    description='ARM NEON-optimized BSQ→HWC conversion for Apple Silicon',
    author='Mars 2020 Viewer Project',
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=False,
        nthreads=8,     # Parallel Cython compilation
    ),
)

# Post-build information
print()
print("=" * 80)
print("Build Configuration Summary")
print("=" * 80)
print(f"Platform:        {platform.machine()} on {platform.system()}")
print(f"Python:          {sys.version.split()[0]}")
print(f"NumPy:           {np.__version__}")
print(f"Optimization:    {'ARM NEON + OpenMP' if is_apple_silicon else 'Generic'}")
print(f"Compile flags:   {' '.join(compile_args[:5])} ...")
print()
print("Generated files:")
print("  - bsq_cython_neon.c        (C source)")
print("  - bsq_cython_neon.html     (annotation)")
print("  - bsq_cython_neon.*.so     (compiled module)")
print()
print("To benchmark:")
print("  python3 benchmark_arm_neon.py")
print("=" * 80)
