# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    # Orbit bundles the minimal llama.cpp tool surface required to build libmtmd.
    add_subdirectory(mtmd)
endif()
