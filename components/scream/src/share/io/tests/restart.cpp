#include <catch2/catch.hpp>

#include "scream_config.h"
#include "share/scream_types.hpp"

#include "share/io/output_manager.hpp"
#include "share/io/scorpio_output.hpp"
#include "share/io/scorpio_input.hpp"
#include "share/io/scream_scorpio_interface.hpp"

#include "share/grid/user_provided_grids_manager.hpp"
#include "share/grid/simple_grid.hpp"

#include "share/field/field_identifier.hpp"
#include "share/field/field_header.hpp"
#include "share/field/field.hpp"
#include "share/field/field_repository.hpp"

#include "ekat/ekat_parameter_list.hpp"
namespace {
using namespace scream;
using namespace ekat::units;
using Device = DefaultDevice;
using input_type = AtmosphereInput;

std::shared_ptr<FieldRepository<Real,DefaultDevice>> get_test_repo(const Int num_cols, const Int num_levs);
std::shared_ptr<UserProvidedGridsManager>            get_test_gm(const ekat::Comm io_comm, const Int num_cols, const Int num_levs);
ekat::ParameterList                                  get_om_params(const Int casenum);
// ekat::ParameterList                                  get_in_params(const std::string type);


TEST_CASE("restart","")
{
  // Note to AaronDonahue:  You are trying to figure out why you can't change the number of cols and levs for this test.  
  // Something having to do with freeing up and then resetting the io_decompositions.
  ekat::Comm io_comm(MPI_COMM_WORLD);  // MPI communicator group used for I/O set as ekat object.
  Int num_cols = 1;
  Int num_levs = 2;

  // First set up a field manager and grids manager to interact with the output functions
  std::shared_ptr<UserProvidedGridsManager>            grid_man   = get_test_gm(io_comm,num_cols,num_levs);
  std::shared_ptr<FieldRepository<Real,DefaultDevice>> field_repo = get_test_repo(num_cols,num_levs);

  // Create an Output manager for testing output
  OutputManager m_output_manager;
  auto output_params = get_om_params(2);
  m_output_manager.set_params(output_params);
  m_output_manager.set_comm(io_comm);
  m_output_manager.set_grids(grid_man);
  m_output_manager.set_repo(field_repo);
  m_output_manager.init();

  // Construct a timestamp
  util::TimeStamp time (0,0,0,0);

  //  Cycle through data and write output
  Int max_steps = 15;
  Real dt = 1.0;
  for (Int ii=0;ii<max_steps;++ii)
  {
    auto& out_fields = field_repo->get_field_groups().at("output");
    for (auto it : out_fields)
    {
      auto f_dev  = field_repo->get_field(it,"Physics").get_view();
      auto f_host = Kokkos::create_mirror_view( f_dev );
      for (size_t jj=0;jj<f_host.size();++jj)
      {
        f_host(jj) += dt;
      }
      Kokkos::deep_copy(f_dev,f_host);
    }
    time += dt;
    m_output_manager.run(time);
  }
  m_output_manager.finalize(time);

  // At this point we should have produced 2 files, a restart and a restart history file.
  
  // Finalize everything
  scorpio::eam_pio_finalize();
  (*grid_man).clean_up();
} // TEST_CASE restart
/*===================================================================================================================*/
std::shared_ptr<FieldRepository<Real,DefaultDevice>> get_test_repo(const Int num_cols, const Int num_levs)
{
  // Create a repo
  std::shared_ptr<FieldRepository<Real,DefaultDevice>>  repo = std::make_shared<FieldRepository<Real,DefaultDevice>>();
  // Create some fields for this repo
  std::vector<FieldTag> tag_h  = {FieldTag::Column};
  std::vector<FieldTag> tag_v  = {FieldTag::VerticalLevel};
  std::vector<FieldTag> tag_2d = {FieldTag::Column,FieldTag::VerticalLevel};

  std::vector<Int>     dims_h  = {num_cols};
  std::vector<Int>     dims_v  = {num_levs};
  std::vector<Int>     dims_2d = {num_cols,num_levs};

  FieldIdentifier fid1("field_1",tag_h,m);
  FieldIdentifier fid2("field_2",tag_v,kg);
  FieldIdentifier fid3("field_3",tag_2d,kg/m);

  fid1.set_dimensions(dims_h);
  fid2.set_dimensions(dims_v);
  fid3.set_dimensions(dims_2d);

  fid1.set_grid_name("Physics");
  fid2.set_grid_name("Physics");
  fid3.set_grid_name("Physics");
  // Register fields with repo
  repo->registration_begins();
  repo->register_field(fid1,{"output","restart"});
  repo->register_field(fid2,{"output","restart"});
  repo->register_field(fid3,{"output"});
  repo->registration_ends();

  // Initialize these fields
  auto f1_dev = repo->get_field(fid1).get_view(); 
  auto f2_dev = repo->get_field(fid2).get_view(); 
  auto f3_dev = repo->get_field(fid3).get_reshaped_view<Real**>();
  auto f1_hst = Kokkos::create_mirror_view( f1_dev );
  auto f2_hst = Kokkos::create_mirror_view( f2_dev ); 
  auto f3_hst = Kokkos::create_mirror_view( f3_dev );
  for (int ii=0;ii<num_cols;++ii)
  {
    f1_hst(ii) = ii;
    for (int jj=0;jj<num_levs;++jj)
    {
      f2_hst(jj) = (jj+1)/10.0;
      f3_hst(ii,jj) = (ii) + (jj+1)/10.0;
    }
  }
  Kokkos::deep_copy(f1_dev,f1_hst);
  Kokkos::deep_copy(f2_dev,f2_hst);
  Kokkos::deep_copy(f3_dev,f3_hst);

  return repo;
}
/*===================================================================================================================*/
std::shared_ptr<UserProvidedGridsManager> get_test_gm(const ekat::Comm io_comm, const Int num_cols, const Int num_levs)
{

  auto& gm_factory = GridsManagerFactory::instance();
  gm_factory.register_product("User Provided",create_user_provided_grids_manager);
  std::shared_ptr<UserProvidedGridsManager> upgm;
  upgm = std::make_shared<UserProvidedGridsManager>();
  auto dummy_grid= std::make_shared<SimpleGrid>("Physics",num_cols,num_levs,io_comm);
  upgm->set_grid(dummy_grid);

  return upgm;
}
/*===================================================================================================================*/
ekat::ParameterList get_om_params(const Int casenum)
{
  ekat::ParameterList om_params("Output Manager");
  om_params.set<Int>("PIO Stride",1);
  if (casenum == 1)
  {
    om_params.set<std::vector<std::string>>("Output YAML Files",{"io_test_instant.yaml","io_test_average.yaml",
           "io_test_max.yaml","io_test_min.yaml"});
  } 
  else if (casenum == 2)
  {
    om_params.set<std::vector<std::string>>("Output YAML Files",{"io_test_restart.yaml"});
    auto& res_sub = om_params.sublist("Restart Control");
    auto& freq_sub = res_sub.sublist("FREQUENCY");
    freq_sub.set<Int>("OUT_N",5);
    freq_sub.set<std::string>("OUT_OPTION","Steps");
  }
  else
  {
    EKAT_REQUIRE_MSG(false,"Error, incorrect case number for get_om_params");
  }

  return om_params;  
}
/*===================================================================================================================*/
// ekat::ParameterList get_in_params(const std::string type)
// {
//   ekat::ParameterList in_params("Input Parameters");
//   in_params.set<std::string>("FILENAME","io_output_test_"+type+"_0.nc");
//   in_params.set<std::string>("GRID","Physics");
//   auto& f_list = in_params.sublist("FIELDS");
//   f_list.set<Int>("Number of Fields",3);
//   for (int ii=1;ii<=3+1;++ii)
//   {
//     f_list.set<std::string>("field "+std::to_string(ii),"field_"+std::to_string(ii));
//   }
//   return in_params;
// }
/*===================================================================================================================*/
} // undefined namespace