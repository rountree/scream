include(ScreamUtils)

# Required libraries
set (NEED_LIBS rrtmgp scream_control scream_share yakl)

#set (additional_src ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/examples/all-sky/mo_garand_atmos_io.cpp)

# NOTE: tests inside this if statement won't be built in a baselines-only build
if (NOT ${SCREAM_BASELINES_ONLY})
    set_source_files_properties("rrtmgp_tests.cpp" PROPERTIES CXX_STANDARD 14)
    set_source_files_properties("rrtmgp_tests.cpp" PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
    CreateUnitTest(rrtmgp_tests "rrtmgp_tests.cpp" "${NEED_LIBS}" LABELS "rrtmgp")
    set_target_properties(rrtmgp_tests PROPERTIES CXX_STANDARD 14)
    #target_include_directories(rrtmgp_tests PUBLIC ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/examples/all-sky)
endif()

# Copy RRTMGP absorption coefficient lookup tables to local data directory
FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)
CONFIGURE_FILE(${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc
               ${CMAKE_CURRENT_BINARY_DIR}/data COPYONLY)
CONFIGURE_FILE(${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc
               ${CMAKE_CURRENT_BINARY_DIR}/data COPYONLY)
CONFIGURE_FILE(${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/examples/all-sky/rrtmgp-allsky.nc
               ${CMAKE_CURRENT_BINARY_DIR}/data COPYONLY)
CONFIGURE_FILE(${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-sw.nc
               ${CMAKE_CURRENT_BINARY_DIR}/data COPYONLY)
CONFIGURE_FILE(${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-lw.nc
               ${CMAKE_CURRENT_BINARY_DIR}/data COPYONLY)
