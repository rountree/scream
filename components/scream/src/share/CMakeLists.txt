set(SHARE_SRC
  scream_session.cpp
  util/scream_utils.cpp
  util/array_io.cpp
  util/array_io_mod.f90
)

set(SHARE_HEADERS
  scream_session.hpp
  util/math_utils.hpp
  util/scream_utils.hpp
)

add_library(scream_share ${SHARE_SRC})
target_include_directories(scream_share PUBLIC ${SCREAM_INCLUDE_DIRS} ${SCREAM_TPL_INCLUDE_DIRS})
set_target_properties(scream_share PROPERTIES
  Fortran_MODULE_DIRECTORY ${SCREAM_F90_MODULES})
# link_directories(${SCREAM_TPL_LIBRARY_DIRS})
# target_link_libraries(scream_share PUBLIC ${SCREAM_TPL_LIBRARIES})

add_subdirectory(tests)