#include "share/grid/point_grid.hpp"
#include "control/surface_coupling.hpp"

#include <ekat/util/ekat_test_utils.hpp>

#include <catch2/catch.hpp>
#include <memory>

TEST_CASE ("surface_coupling")
{
  /*
   * This test performs an export-import sequence, and it verifies that
   * the resulting fields (after the import) match the fields that were
   * originally exported. If successful, the test "ensures" that the
   * import/export kernels in the SurfaceCoupling class are correct.
   */

  // Some namespaces/aliases
  using namespace scream;
  using namespace ShortFieldTagsNames;
  using namespace ekat::units;
  using FL = FieldLayout;
  using FID = FieldIdentifier;
  using RPDF = std::uniform_real_distribution<Real>;
  using rngAlg = std::mt19937_64;

  // Some constants
  constexpr int ncols = 4;
  constexpr int nlevs = 8;
  constexpr int nruns = 10;

  // Create a comm
  ekat::Comm comm (MPI_COMM_WORLD);

  // The random numbers generator
  std::random_device rd;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  if (comm.am_i_root()) {
    std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  }
  rngAlg engine(seed);
  RPDF pdf(0.0,1.0);

  // Create a grid
  auto grid = create_point_grid("my grid",ncols*comm.size(), nlevs, comm);

  // Create some field ids, and register them in a field manager
  // Note: we create two fms, so we can compare outputs with inputs
  FID s2d_id("s2d",FL{{COL},{ncols}},Pa,grid->name());
  FID s3d_id("s3d",FL{{COL,LEV},{ncols,nlevs}},Pa,grid->name());
  FID v3d_id("v3d",FL{{COL,CMP,LEV},{ncols,2,nlevs}},m/s,grid->name());

  // NOTE: if you add fields abovew, you will have to modify these counters too.
  const int num_s2d = 1;
  const int num_s3d = 1;
  const int num_v3d = 1;
  const int num_fields = num_s2d+num_s3d+2*num_v3d;

  // Keep two separate fms, so we can compare original and final fields.
  auto fm_in = std::make_shared<FieldManager<Real>> (grid);
  auto fm_out = std::make_shared<FieldManager<Real>> (grid);
  fm_in->registration_begins();
  fm_in->register_field(s2d_id);
  fm_in->register_field(s3d_id);
  fm_in->register_field(v3d_id);
  fm_in->registration_ends();
  fm_out->registration_begins();
  fm_out->register_field(s2d_id);
  fm_out->register_field(s3d_id);
  fm_out->register_field(v3d_id);
  fm_out->registration_ends();

  // Create two SC objects, to import and export
  control::SurfaceCoupling importer(fm_in);
  control::SurfaceCoupling exporter(fm_out);

  importer.set_num_fields(num_fields,0); // Recall that SC counts *scalar* fields, so vector3d counts as 2 fields
  exporter.set_num_fields(0,num_fields);

  // Register fields in the importer/exporter
  // Note: the 1st integer is the field "idx" (the idx used by the component cpl to retrieve it),
  //       which here is not really used (though it still needs to be passed). The 2nd integer
  //       is needed for vector fields, to tell the importer/exporter which component of the
  //       vector field is imported/exported.
  importer.register_import("s2d",0);
  importer.register_import("s3d",1);
  importer.register_import("v3d",2,0);
  importer.register_import("v3d",3,1);
  exporter.register_export("s2d",0);
  exporter.register_export("s3d",1);
  exporter.register_export("v3d",2,0);
  exporter.register_export("v3d",3,1);

  // Create a raw array big enough to contain all the 2d data for import/export
  double* raw_data = new double[ncols*num_fields];

  // Complete setup of importer/exporter
  importer.registration_ends(raw_data,nullptr);
  exporter.registration_ends(nullptr,raw_data);

  // Repeat experiment N times: fill export fields, export, import, check import fields
  auto s2d_exp = fm_out->get_field(s2d_id);
  auto s3d_exp = fm_out->get_field(s3d_id);
  auto v3d_exp = fm_out->get_field(v3d_id);
  auto s2d_imp = fm_in->get_field(s2d_id);
  auto s3d_imp = fm_in->get_field(s3d_id);
  auto v3d_imp = fm_in->get_field(v3d_id);

  auto s2d_exp_d = s2d_exp.get_reshaped_view<Real*>();
  auto s3d_exp_d = s3d_exp.get_reshaped_view<Real**>();
  auto v3d_exp_d = v3d_exp.get_reshaped_view<Real***>();
  auto s2d_imp_d = s2d_imp.get_reshaped_view<Real*>();
  auto s3d_imp_d = s3d_imp.get_reshaped_view<Real**>();
  auto v3d_imp_d = v3d_imp.get_reshaped_view<Real***>();

  auto s2d_exp_h = s2d_exp.get_reshaped_view<Real*,Host>();
  auto s3d_exp_h = s3d_exp.get_reshaped_view<Real**,Host>();
  auto v3d_exp_h = v3d_exp.get_reshaped_view<Real***,Host>();
  auto s2d_imp_h = s2d_imp.get_reshaped_view<Real*,Host>();
  auto s3d_imp_h = s3d_imp.get_reshaped_view<Real**,Host>();
  auto v3d_imp_h = v3d_imp.get_reshaped_view<Real***,Host>();

  for (int i=0; i<nruns; ++i) {
    // Fill export fields
    ekat::genRandArray(s2d_exp_d,engine,pdf);
    ekat::genRandArray(s3d_exp_d,engine,pdf);
    ekat::genRandArray(v3d_exp_d,engine,pdf);

    // Set all raw_data to -1 (might be helpful for debugging)
    std::fill_n(raw_data,4*ncols,-1);

    // Perform export
    exporter.do_export();

    // Perform import
    importer.do_import();

    // Check f_imported==f_exported (on surface only)
    s2d_exp.sync_to_host();
    s3d_exp.sync_to_host();
    v3d_exp.sync_to_host();
    s2d_imp.sync_to_host();
    s3d_imp.sync_to_host();
    v3d_imp.sync_to_host();
    for (int icol=0; icol<ncols; ++icol) {
      REQUIRE (s2d_exp_h(icol)==s2d_imp_h(icol));
      REQUIRE (s3d_exp_h(icol,0)==s3d_imp_h(icol,0));
      REQUIRE (v3d_exp_h(icol,0,0)==v3d_imp_h(icol,0,0));
      REQUIRE (v3d_exp_h(icol,0,1)==v3d_imp_h(icol,0,1));
    }
  }

  // Clean up
  delete[] raw_data;
}
