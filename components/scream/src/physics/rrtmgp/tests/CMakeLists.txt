include(ScreamUtils)

# Required libraries
SET (rrtmgpLib rrtmgp)
SET (NEED_LIBS ${rrtmgpLib} scream_control scream_share yakl)

# NOTE: tests inside this if statement won't be built in a baselines-only build

if (NOT ${SCREAM_BASELINES_ONLY})
    set_source_files_properties("rrtmgp_tests.cpp" PROPERTIES CXX_STANDARD 14)
    CreateUnitTest(rrtmgp_tests "rrtmgp_tests.cpp" "${NEED_LIBS}" LABELS "rrtmgp")
    #CreateUnitTest(rrtmgp_tests rrtmgp_tests.cpp "rrtmgp;scream_share")
    set_target_properties(rrtmgp_tests PROPERTIES CXX_STANDARD 14)
endif()
target_compile_options(rrtmgp_tests PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>)

# Copy RRTMGP absorption coefficient lookup tables to local data directory
FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)
CONFIGURE_FILE(${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc
               ${CMAKE_CURRENT_BINARY_DIR}/data COPYONLY)
CONFIGURE_FILE(${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc
               ${CMAKE_CURRENT_BINARY_DIR}/data COPYONLY)
