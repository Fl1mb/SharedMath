find_package(Doxygen OPTIONAL_COMPONENTS dot)

if(NOT DOXYGEN_FOUND)
    message(STATUS "[docs] Doxygen not found — skipping documentation target")
    return()
endif()

# ── Collect all public header directories ────────────────────────────────────
set(_input_dirs
    "${CMAKE_SOURCE_DIR}/core/include"
    "${CMAKE_SOURCE_DIR}/LinearAlgebra/include"
    "${CMAKE_SOURCE_DIR}/geometry/include"
    "${CMAKE_SOURCE_DIR}/functions/include"
    "${CMAKE_SOURCE_DIR}/Graphs/include"
    "${CMAKE_SOURCE_DIR}/DSP/include"
    "${CMAKE_SOURCE_DIR}/NumericalMethods/include"
    "${CMAKE_SOURCE_DIR}/ML/include"
    "${CMAKE_SOURCE_DIR}/docs"           # mainpage.dox
)
# Doxygen INPUT expects space-separated list
list(JOIN _input_dirs " " DOXYGEN_INPUT_DIRS)

# ── Dot / Graphviz ───────────────────────────────────────────────────────────
if(Doxygen_dot_FOUND)
    set(DOXYGEN_HAVE_DOT "YES")
    get_filename_component(DOXYGEN_DOT_PATH "${DOXYGEN_DOT_EXECUTABLE}" DIRECTORY)
else()
    set(DOXYGEN_HAVE_DOT "NO")
    set(DOXYGEN_DOT_PATH "")
    message(STATUS "[docs] Graphviz dot not found — dependency graphs disabled")
endif()

# ── Output directory ─────────────────────────────────────────────────────────
set(DOXYGEN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/docs")

# ── Configure the Doxyfile ───────────────────────────────────────────────────
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    @ONLY
)

# ── Custom target ─────────────────────────────────────────────────────────────
add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
)

# ── Stamp file so the target is not rerun needlessly ────────────────────────
set_property(
    DIRECTORY APPEND PROPERTY
    ADDITIONAL_MAKE_CLEAN_FILES "${DOXYGEN_OUTPUT_DIR}"
)

message(STATUS "[docs] Doxygen ${DOXYGEN_VERSION} found — run 'cmake --build build --target docs'")
if(Doxygen_dot_FOUND)
    message(STATUS "[docs] Graphviz dot found at ${DOXYGEN_DOT_EXECUTABLE}")
endif()
