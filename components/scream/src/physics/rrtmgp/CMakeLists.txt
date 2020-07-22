# RRTMGP++ requires YAKL
add_subdirectory(${SCREAM_BASE_DIR}/../../externals/yakl ${CMAKE_CURRENT_BINARY_DIR}/yakl_build)
target_compile_options(yakl PUBLIC $<$<COMPILE_LANGUAGE:Fortran>:-fno-default-real-8 -fno-default-double-8>)
target_compile_options(yakl PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>)
set_target_properties(yakl PROPERTIES CXX_STANDARD 14)

# Add RRTMGP source files.
set(RRTMGP_SRCS
  atmosphere_radiation.cpp
  scream_rrtmgp_interface.cpp
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rrtmgp/kernels/mo_gas_optics_kernels.cpp
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rrtmgp/kernels/mo_rrtmgp_util_reorder_kernels.h
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rrtmgp/mo_gas_concentrations.h
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rrtmgp/mo_gas_optics_rrtmgp.h
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rrtmgp/mo_rrtmgp_constants.cpp
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rrtmgp/mo_rrtmgp_util_reorder.cpp
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rrtmgp/mo_rrtmgp_util_string.h
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rte/mo_fluxes.h
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rte/mo_optical_props.h
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rte/mo_rte_lw.cpp
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rte/mo_rte_sw.cpp
  #${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rte/mo_rte_util_array.cpp
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rte/mo_source_functions.h
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rte/kernels/mo_fluxes_broadband_kernels.cpp
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rte/kernels/mo_optical_props_kernels.cpp
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rte/kernels/mo_rte_solver_kernels.cpp
  #${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/examples/mo_load_coefficients.cpp
  mo_load_coefficients.cpp
  ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/const.h
  #${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/extensions/mo_fluxes_byband.cpp
  #${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/extensions/mo_fluxes_byband_kernels.cpp
  #${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/extensions/mo_fluxes_bygpoint.cpp
  #${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/extensions/mo_heating_rates.cpp
  #${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/extensions/mo_rrtmgp_clr_all_sky.cpp
)

set(RRTMGP_HEADERS
  rrtmgp.hpp
  scream_rrtmgp_interface.hpp
  atmosphere_radiation.hpp
)

add_library(rrtmgp ${RRTMGP_SRCS})
set_source_files_properties(${RRTMGP_SRCS} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
target_include_directories(rrtmgp PUBLIC ${SCREAM_INCLUDE_DIRS})
target_include_directories(rrtmgp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_include_directories(rrtmgp SYSTEM PUBLIC ${SCREAM_TPL_INCLUDE_DIRS} ${CIMEROOT}/src/share/include)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rte)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rte/kernels)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rrtmgp)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rrtmgp/kernels)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/examples)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../../externals/yakl)
set_target_properties(rrtmgp PROPERTIES Fortran_MODULE_DIRECTORY ${SCREAM_F90_MODULES})
set_target_properties(rrtmgp PROPERTIES CXX_STANDARD 14)
target_link_libraries(rrtmgp yakl physics_common scream_share ${SCREAM_TPL_LIBRARIES})

if (NOT SCREAM_LIB_ONLY)
  add_subdirectory(tests)
endif()

# Old sources for Fortran code
set(RRTMGP_FORTRAN_SRC
  scream_rrtmgp_interface.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rrtmgp/kernels/mo_gas_optics_kernels.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rrtmgp/kernels/mo_rrtmgp_util_reorder_kernels.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rrtmgp/mo_gas_concentrations.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rrtmgp/mo_gas_optics.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rrtmgp/mo_gas_optics_rrtmgp.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rrtmgp/mo_rrtmgp_constants.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rrtmgp/mo_rrtmgp_util_reorder.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rrtmgp/mo_rrtmgp_util_string.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rte/mo_fluxes.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rte/mo_optical_props.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rte/mo_rte_kind.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rte/mo_rte_lw.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rte/mo_rte_sw.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rte/mo_rte_util_array.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rte/mo_source_functions.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rte/kernels/mo_fluxes_broadband_kernels.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rte/kernels/mo_optical_props_kernels.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/rte/kernels/mo_rte_solver_kernels.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/extensions/mo_fluxes_byband.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/extensions/mo_fluxes_byband_kernels.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/extensions/mo_fluxes_bygpoint.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/extensions/mo_heating_rates.F90
  ${SCREAM_BASE_DIR}/../eam/src/physics/rrtmgp/external/extensions/mo_rrtmgp_clr_all_sky.F90
)

set(RRTMGP_HEADERS
  rrtmgp.hpp
  scream_rrtmgp_interface.hpp
  atmosphere_radiation.hpp
)

add_subdirectory(${SCREAM_BASE_DIR}/../../externals/yakl ${CMAKE_CURRENT_BINARY_DIR}/yakl_build)
set_source_files_properties(${RRTMGP_SRCS} PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
target_compile_options(yakl PUBLIC $<$<COMPILE_LANGUAGE:Fortran>:-fno-default-real-8 -fno-default-double-8>)
target_compile_options(yakl PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>)
set_target_properties(yakl PROPERTIES CXX_STANDARD 14)

add_library(rrtmgp ${RRTMGP_SRCS})
target_include_directories(rrtmgp PUBLIC ${SCREAM_INCLUDE_DIRS})
target_include_directories(rrtmgp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../share)
target_include_directories(rrtmgp SYSTEM PUBLIC ${SCREAM_TPL_INCLUDE_DIRS} ${CIMEROOT}/src/share/include)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rte)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rte/kernels)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rrtmgp)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/rrtmgp/kernels)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../cam/src/physics/rrtmgp/external/cpp/examples)
target_include_directories(rrtmgp PUBLIC ${SCREAM_BASE_DIR}/../../externals/yakl)
set_target_properties(rrtmgp PROPERTIES Fortran_MODULE_DIRECTORY ${SCREAM_F90_MODULES})
set_target_properties(rrtmgp PROPERTIES CXX_STANDARD 14)
target_link_libraries(rrtmgp physics_common scream_share ${SCREAM_TPL_LIBRARIES})

if (NOT SCREAM_LIB_ONLY)
  add_subdirectory(tests)
endif()
