#include "dynamics/homme/dynamics_driven_grids_manager.hpp"
#include "dynamics/homme/interface/scream_homme_interface.hpp"
#include "dynamics/homme/physics_dynamics_remapper.hpp"

#include "share/grid/se_grid.hpp"
#include "share/grid/point_grid.hpp"
#include "share/grid/remap/inverse_remapper.hpp"

// Get all Homme's compile-time dims
#include "homme_dimensions.hpp"

namespace scream
{

DynamicsDrivenGridsManager::
DynamicsDrivenGridsManager (const ekat::Comm& /* comm */,
                            const ekat::ParameterList& /* p */)
{
  // Valid names for the dyn grid
  auto& dgn = m_dyn_grid_aliases;

  dgn.insert("Dynamics");
  dgn.insert("SE Dynamics");
  dgn.insert("GLL");
  dgn.insert("Dyn Grid");

  // Valid names for the different phys grids

  // No redistribution of columns
  auto& pg0 = m_phys_grid_aliases[0];
  pg0.insert("Physics");
  pg0.insert("Physics GLL"); // Phys columns are SE gll points

  auto& pg2 = m_phys_grid_aliases[2];
  pg2.insert("Phys Grid");
  pg2.insert("Physics PG2"); // Phys columns are FV points in 2x2 subcells of SE cell

  auto& pg3 = m_phys_grid_aliases[3];
  pg3.insert("Physics PG3"); // Phys columns are FV points in 3x3 subcells of SE cell

  // Twin columns
  auto& pg10 = m_phys_grid_aliases[10];
  pg10.insert("Physics Twin");     // Same as GLL
  pg10.insert("Physics GLL Twin"); // Phys columns are SE gll points

  auto& pg12 = m_phys_grid_aliases[10];
  pg12.insert("Phys Grid Twin");
  pg12.insert("Physics PG2 Twin"); // Phys columns are FV points in 2x2 subcells of SE cell

  auto& pg13 = m_phys_grid_aliases[10];
  pg13.insert("Physics PG3 Twin"); // Phys columns are FV points in 2x2 subcells of SE cell

  // TODO: add other rebalancing?

  for (const auto& gn : dgn) {
    m_valid_grid_names.insert(gn);
  }
  for (const auto& it : m_phys_grid_aliases) {
    for (const auto& gn : it.second) {
      m_valid_grid_names.insert(gn);
    }
  }

  // Create the grid codes map
  build_grid_codes ();
}

DynamicsDrivenGridsManager::
~DynamicsDrivenGridsManager () {
  // Cleanup the grids stuff
  finalize_geometry_f90 ();
}

DynamicsDrivenGridsManager::remapper_ptr_type
DynamicsDrivenGridsManager::do_create_remapper (const grid_ptr_type from_grid,
                                                const grid_ptr_type to_grid) const {
  using PDR = PhysicsDynamicsRemapper<remapper_type::real_type>;
  auto pd_remapper = std::make_shared<PDR>(m_grids.at("Physics"),m_grids.at("Dynamics"));
  if (from_grid->name()=="SE Physics" &&
      to_grid->name()=="SE Dynamics") {
    return pd_remapper;
  } else if (from_grid->name()=="SE Dynamics" &&
             to_grid->name()=="SE Physics") {
    return std::make_shared<InverseRemapper<Real>>(pd_remapper);
  }
  return nullptr;
}

void DynamicsDrivenGridsManager::
build_grids (const std::set<std::string>& grid_names,
             const std::string& reference_grid) {
  // Retrieve all grid codes
  std::set<int> codes;
  for (const auto& gn : grid_names) {
    // Sanity check first
    EKAT_REQUIRE_MSG (supported_grids().count(gn)==1,
                      "Error! Grid '" + gn + "' is not supported by this grid manager.\n");

    codes.insert(m_grid_codes.at(gn));
  }

  // Deduce the phys grid type we need
  int pgN = -1;
  for (auto code : codes) {
    if (code>=0) {
      int N = code % 10;
      EKAT_REQUIRE_MSG (pgN==-1 || N==pgN,
                        "Error! Mixing different types of physics grid is not allowed.\n"
                        "       You can, however, have phys grids with different *balancing* options.\n");
      pgN = N;
    }
  }

  // Nobody should have init-ed the geometries yet. So error out if someone did.
  EKAT_REQUIRE_MSG (!is_geometry_inited_f90(), "Error! Geometry was somehow already init-ed.\n");

  init_grids_f90 (pgN);

  // We know we need the dyn grid, so build it
  build_dynamics_grid ();

  for (const auto& gn : grid_names) {
    if (m_dyn_grid_aliases.count(gn)==0) {
      build_physics_grid(gn);
    }
  }

  // Now we can cleanup all the grid stuff in f90
  cleanup_grid_init_data_f90 ();

  // Set the ptr to the ref grid
  m_grids["Reference"] = get_grid(reference_grid); 
}

void DynamicsDrivenGridsManager::build_dynamics_grid () {
  if (m_grids.find("Dynamics")==m_grids.end()) {

    // Get dimensions and create "empty" grid
    const int nlelem = get_num_local_elems_f90();
    const int ngelem = get_num_global_elems_f90();
    const int nlev   = get_nlev_f90();
    auto dyn_grid = std::make_shared<SEGrid>("SE Dynamics",ngelem,nlelem,NP,nlev);

    const int ndofs = nlelem*NP*NP;

    // Create the gids, elgpgp, coords, area views
    AbstractGrid::dofs_list_type      dofs("dyn dofs",ndofs);
    AbstractGrid::lid_to_idx_map_type elgpgp("dof idx",ndofs,3);
    AbstractGrid::geo_view_type       lat("lat",ndofs);
    AbstractGrid::geo_view_type       lon("lon",ndofs);
    auto h_dofs   = Kokkos::create_mirror_view(dofs);
    auto h_elgpgp = Kokkos::create_mirror_view(elgpgp);
    auto h_lat    = Kokkos::create_mirror_view(lat);
    auto h_lon    = Kokkos::create_mirror_view(lon);

    // Get (ie,igp,jgp,gid) data for each dof
    get_dyn_grid_data_f90 (h_dofs.data(),h_elgpgp.data(), h_lat.data(), h_lon.data());

    Kokkos::deep_copy(dofs,h_dofs);
    Kokkos::deep_copy(elgpgp,h_elgpgp);
    Kokkos::deep_copy(lat,h_lat);
    Kokkos::deep_copy(lon,h_lon);

    // Set dofs and geo data in the grid
    dyn_grid->set_dofs (dofs, elgpgp);
    dyn_grid->set_geometry_data ("lat", lat);
    dyn_grid->set_geometry_data ("lon", lon);

    // Set the grid in the map
    for (const auto& gn : m_dyn_grid_aliases) {
      m_grids[gn] = dyn_grid;
    }
  }
}

void DynamicsDrivenGridsManager::
build_physics_grid (const std::string& name) {

  // Build only if not built yet
  if (m_grids.find(name)==m_grids.end()) {

    // Get the grid pg_type
    const int pg_type = m_grid_codes.at(name);

    // Get dimensions and create "empty" grid
    const int nlev  = get_nlev_f90();
    const int nlcols = get_num_local_columns_f90 ();
    const int ngcols = get_num_global_columns_f90 ();

    auto phys_grid = std::make_shared<PointGrid>("Physics",ngcols,nlcols,nlev);

    // Create the gids, coords, area views
    AbstractGrid::dofs_list_type dofs("phys dofs",nlcols);
    AbstractGrid::geo_view_type  lat("lat",nlcols);
    AbstractGrid::geo_view_type  lon("lon",nlcols);
    AbstractGrid::geo_view_type  area("area",nlcols);
    auto h_dofs = Kokkos::create_mirror_view(dofs);
    auto h_lat  = Kokkos::create_mirror_view(lat);
    auto h_lon  = Kokkos::create_mirror_view(lon);
    auto h_area = Kokkos::create_mirror_view(area);

    // Get all specs of phys grid cols (gids, coords, area)
    get_phys_grid_data_f90 (pg_type, h_dofs.data(), h_lat.data(), h_lon.data(), h_area.data());

    Kokkos::deep_copy(dofs,h_dofs);
    Kokkos::deep_copy(lat, h_lat);
    Kokkos::deep_copy(lon, h_lon);
    Kokkos::deep_copy(area,h_area);

    // Set dofs and geo data in the grid
    phys_grid->set_dofs(dofs);
    phys_grid->set_geometry_data("lat",lat);
    phys_grid->set_geometry_data("lon",lon);
    phys_grid->set_geometry_data("area",area);

    // Set the grid in the map (for all its aliases)
    for (const auto& gn : m_phys_grid_aliases[pg_type]) {
      m_grids[gn] = phys_grid;
    }
  }
}

void DynamicsDrivenGridsManager::
build_grid_codes () {

  // Codes for the physics grids supported
  constexpr int dyn   = -1;  // Dyanamics grid (not a phys grid)
  constexpr int gll   =  0;  // Physics GLL
  constexpr int pg2   =  2;  // Physics PG2
  constexpr int pg3   =  3;  // Physics PG3
  constexpr int pg4   =  3;  // Physics PG4
  constexpr int gll_t = 10;  // Physics GLL Twin
  constexpr int pg2_t = 12;  // Physics PG2 Twin
  constexpr int pg3_t = 13;  // Physics PG3 Twin
  constexpr int pg4_t = 13;  // Physics PG4 Twin

  for (const auto& name : m_valid_grid_names) {
    int code;

    if (name=="Physics" || name=="Physics GLL") {
      code = gll;
    } else if (name=="Phys Grid" || name=="Physics PG2") {
      code = pg2;
    } else if (name=="Physics Twin" || name=="Physics GLL Twin") {
      code = gll_t;
    } else if (name=="Phys Grid Twin" || name=="Physics PG2 Twin") {
      code = pg2_t;
    } else if (name=="Physics PG3") {
      code = pg3;
    } else if (name=="Physics PG3 Twin") {
      code = pg3_t;
    } else if (name=="Physics PG4") {
      code = pg4;
    } else if (name=="Physics PG4 Twin") {
      code = pg4_t;
    } else {
      code = dyn;
    }

    m_grid_codes[name] = code;
  } 
} 

} // namespace scream
