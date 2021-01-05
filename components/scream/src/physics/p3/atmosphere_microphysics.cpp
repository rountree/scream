#include "physics/p3/atmosphere_microphysics.hpp"
#include "physics/p3/p3_inputs_initializer.hpp"
#include "physics/p3/p3_main_impl.hpp"

// Needed for p3_init, the only F90 code still used.
#include "physics/p3/p3_f90.hpp"

#include "physics/share/physics_constants.hpp"
#include "physics/share/physics_functions.hpp" // also for ETI not on GPUs
#include "physics/share/physics_universal_impl.hpp"

#include "ekat/ekat_assert.hpp"
#include "ekat/ekat_pack_kokkos.hpp"
#include "ekat/ekat_pack.hpp"
#include "ekat/kokkos/ekat_kokkos_utils.hpp"

#include <array>

namespace scream
{
/*
 * P3 Microphysics routines
*/
  using namespace p3;
  using P3F          = Functions<Real, DefaultDevice>;
  using PC           = scream::physics::Constants<Real>;
  using Spack        = typename P3F::Spack;
  using Smask        = typename P3F::Smask;
  using Pack         = ekat::Pack<Real,Spack::n>;
  using IntSmallPack = typename ekat::Pack<Int, Spack::n>;

  using view_1d  = typename P3F::view_1d<Real>;
  using view_2d  = typename P3F::view_2d<Spack>;
  using sview_2d = typename KokkosTypes<DefaultDevice>::template view_2d<Real>;
// =========================================================================================
P3Microphysics::P3Microphysics (const ekat::Comm& comm, const ekat::ParameterList& params)
 : m_p3_comm (comm)
 , m_p3_params (params)
{
/* Anything that can be initialized without grid information can be initialized here.
 * Like universal constants, table lookups, p3 options.
*/
  m_initializer = create_field_initializer<P3InputsInitializer>();
}

// =========================================================================================
void P3Microphysics::set_grids(const std::shared_ptr<const GridsManager> grids_manager)
{
  using namespace ekat::units;

  // The units of mixing ratio Q are technically non-dimensional.
  // Nevertheless, for output reasons, we like to see 'kg/kg'.
  auto Q = kg/kg;
  Q.set_string("kg/kg");
  auto nondim = m/m;
  auto mm = m/1000;

  m_num_levs = SCREAM_NUM_VERTICAL_LEV;

  const auto& grid_name = m_p3_params.get<std::string>("Grid");
  auto grid = grids_manager->get_grid(grid_name);
  m_num_cols = grid->get_num_local_dofs();


  using namespace ShortFieldTagsNames;

  FieldLayout scalar3d_layout_mid { {COL,VL}, {m_num_cols,m_num_levs} }; // Note that C++ and Fortran read array dimensions in reverse
  FieldLayout scalar3d_layout_int { {COL,VL}, {m_num_cols,m_num_levs+1} }; // Note that C++ and Fortran read array dimensions in reverse

  // Variables needed, but not passed to p3_main
  m_required_fields.emplace("ast",   scalar3d_layout_mid, nondim, grid_name);
  m_required_fields.emplace("pmid",  scalar3d_layout_mid, Pa,     grid_name);
  m_required_fields.emplace("zi",    scalar3d_layout_int, m,      grid_name);
  m_required_fields.emplace("T_atm", scalar3d_layout_mid, K,      grid_name);
  m_computed_fields.emplace("T_atm", scalar3d_layout_mid, K,      grid_name);

  // Prognostic State
  m_required_fields.emplace("qv",     scalar3d_layout_mid, Q, grid_name);
  m_required_fields.emplace("qc",     scalar3d_layout_mid, Q, grid_name);
  m_required_fields.emplace("qr",     scalar3d_layout_mid, Q, grid_name);
  m_required_fields.emplace("qi",     scalar3d_layout_mid, Q, grid_name);
  m_required_fields.emplace("qm",     scalar3d_layout_mid, Q, grid_name);
  m_required_fields.emplace("nc",     scalar3d_layout_mid, 1/kg, grid_name);
  m_required_fields.emplace("nr",     scalar3d_layout_mid, 1/kg, grid_name);
  m_required_fields.emplace("ni",     scalar3d_layout_mid, 1/kg, grid_name);
  m_required_fields.emplace("bm",     scalar3d_layout_mid, 1/kg, grid_name);
//  m_required_fields.emplace("th_atm", scalar3d_layout_mid, K, grid_name);
  //
  m_computed_fields.emplace("qv",     scalar3d_layout_mid, Q, grid_name);
  m_computed_fields.emplace("qc",     scalar3d_layout_mid, Q, grid_name);
  m_computed_fields.emplace("qr",     scalar3d_layout_mid, Q, grid_name);
  m_computed_fields.emplace("qi",     scalar3d_layout_mid, Q, grid_name);
  m_computed_fields.emplace("qm",     scalar3d_layout_mid, Q, grid_name);
  m_computed_fields.emplace("nc",     scalar3d_layout_mid, 1/kg, grid_name);
  m_computed_fields.emplace("nr",     scalar3d_layout_mid, 1/kg, grid_name);
  m_computed_fields.emplace("ni",     scalar3d_layout_mid, 1/kg, grid_name);
  m_computed_fields.emplace("bm",     scalar3d_layout_mid, 1/kg, grid_name);
  m_computed_fields.emplace("th_atm", scalar3d_layout_mid, K, grid_name);
  // Diagnostic Inputs
  m_required_fields.emplace("nc_nuceat_tend",  scalar3d_layout_mid, 1/(kg*s), grid_name);
  m_required_fields.emplace("nccn_prescribed", scalar3d_layout_mid, nondim, grid_name);
  m_required_fields.emplace("ni_activated",    scalar3d_layout_mid, 1/kg, grid_name);
  m_required_fields.emplace("inv_qc_relvar",   scalar3d_layout_mid, nondim, grid_name);
  m_required_fields.emplace("dp",              scalar3d_layout_mid, Pa, grid_name);
  m_required_fields.emplace("qv_prev",         scalar3d_layout_mid, Q, grid_name);
  m_required_fields.emplace("T_prev",          scalar3d_layout_mid, K, grid_name); 
  //
  m_computed_fields.emplace("qv_prev",         scalar3d_layout_mid, Q, grid_name);
  m_computed_fields.emplace("T_prev",          scalar3d_layout_mid, K, grid_name);
  // Diagnostic Outputs
  m_computed_fields.emplace("mu_c",               scalar3d_layout_mid, nondim, grid_name);
  m_computed_fields.emplace("lamc",               scalar3d_layout_mid, nondim, grid_name);
  m_computed_fields.emplace("diag_eff_radius_qc", scalar3d_layout_mid, m, grid_name);
  m_computed_fields.emplace("diag_eff_radius_qi", scalar3d_layout_mid, m, grid_name);
  m_computed_fields.emplace("precip_total_tend",  scalar3d_layout_mid, mm, grid_name);
  m_computed_fields.emplace("nevapr",             scalar3d_layout_mid, nondim, grid_name);
  m_computed_fields.emplace("qr_evap_tend",       scalar3d_layout_mid, mm/s, grid_name);
  // History Only
  m_computed_fields.emplace("liq_ice_exchange", scalar3d_layout_mid, nondim, grid_name);
  m_computed_fields.emplace("vap_liq_exchange", scalar3d_layout_mid, nondim, grid_name);
  m_computed_fields.emplace("vap_ice_exchange", scalar3d_layout_mid, nondim, grid_name);
}

// =========================================================================================
void P3Microphysics::initialize_impl (const util::TimeStamp& t0)
{
  m_current_ts = t0;

  // Call p3 initilization routines from main code.
  p3_init();

  // We may have to init some fields from within P3. This can be the case in a P3 standalone run.
  // Some options:
  //  - we can tell P3 it can init all inputs or specify which ones it can init. We call the
  //    resulting list of inputs the 'initializaable' (or initable) inputs. The default is
  //    that no inputs can be inited.
  //  - we can request that P3 either inits no inputs or all of the initable ones (as specified
  //    at the previous point). The default is that P3 must be in charge of init ing ALL or NONE
  //    of its initable inputs.
  // Recall that:
  //  - initable fields may not need initialization (e.g., some other atm proc that
  //    appears earlier in the atm dag might provide them).

//  std::vector<std::string> p3_inputs = {"q","T","FQ","ast","ni_activated","nc_nuceat_tend","pmid","dp","zi","qv_prev","T_prev"};
  std::vector<std::string> p3_inputs = {"T_atm","ast","ni_activated","nc_nuceat_tend","pmid","dp","zi","qv_prev","T_prev",
                                        "qv", "qc", "qr", "qi", "qm", "nc", "nr", "ni", "bm","nccn_prescribed","inv_qc_relvar"
                                       };
  using strvec = std::vector<std::string>;
  const strvec& allowed_to_init = m_p3_params.get<strvec>("Initializable Inputs",strvec(0));
  const bool can_init_all = m_p3_params.get<bool>("Can Initialize All Inputs", false);
  const bool init_all_or_none = m_p3_params.get<bool>("Must Init All Inputs Or None", true);

  const strvec& initable = can_init_all ? p3_inputs : allowed_to_init;
  if (initable.size()>0) {
    bool all_inited = true, all_uninited = true;
    for (const auto& name : initable) {
      const auto& f = m_p3_fields_in.at(name);
      const auto& track = f.get_header().get_tracking();
      if (track.get_init_type()==InitType::None) {
        // Nobody claimed to init this field. P3InputsInitializer will take care of it
        m_initializer->add_me_as_initializer(f);
        all_uninited &= true;
        all_inited &= false;
      } else {
        all_uninited &= false;
        all_inited &= true;
      }
    }

    // In order to gurantee some consistency between inputs, it is best if P3
    // initializes either none or all of the inputs.
    EKAT_REQUIRE_MSG (!init_all_or_none || all_inited || all_uninited,
                      "Error! Some p3 inputs were marked to be inited by P3, while others weren't.\n"
                      "       P3 was requested to init either all or none of the inputs.\n");
  }
}

// =========================================================================================
void P3Microphysics::run_impl (const Real dt)
{

  std::vector<const Real*> in;
  std::vector<Real*> out;

  // Copy inputs to host. Copy also outputs, cause we might "update" them, rather than overwrite them.
  for (auto& it : m_p3_fields_in) {
    Kokkos::deep_copy(m_p3_host_views_in.at(it.first),it.second.get_view());
  }
  for (auto& it : m_p3_fields_out) {
    Kokkos::deep_copy(m_p3_host_views_out.at(it.first),it.second.get_view());
  }

  // --Prognostic State Variables: 
  auto qc     = m_p3_fields_out["qc"].get_reshaped_view<Pack**>();
  auto nc     = m_p3_fields_out["nc"].get_reshaped_view<Pack**>();
  auto qr     = m_p3_fields_out["qr"].get_reshaped_view<Pack**>();
  auto nr     = m_p3_fields_out["nr"].get_reshaped_view<Pack**>();
  auto qi     = m_p3_fields_out["qi"].get_reshaped_view<Pack**>();
  auto qm     = m_p3_fields_out["qm"].get_reshaped_view<Pack**>();
  auto ni     = m_p3_fields_out["ni"].get_reshaped_view<Pack**>();
  auto bm     = m_p3_fields_out["bm"].get_reshaped_view<Pack**>();
  auto qv     = m_p3_fields_out["qv"].get_reshaped_view<Pack**>();
  auto th_atm = m_p3_fields_out["th_atm"].get_reshaped_view<Pack**>();
  // --Diagnostic Input Variables:
  // local arrays
  view_2d cld_frac_i("cld_frac_i",m_num_cols,m_num_levs);
  view_2d cld_frac_l("cld_frac_l",m_num_cols,m_num_levs);
  view_2d cld_frac_r("cld_frac_r",m_num_cols,m_num_levs);
  view_2d dz("dz",m_num_cols,m_num_levs);
  view_2d exner("exner",m_num_cols,m_num_levs);
  // field managed arrays
  auto nc_nuceat_tend  = m_p3_fields_in["nc_nuceat_tend"].get_reshaped_view<const Pack**>();
  auto nccn_prescribed = m_p3_fields_in["nccn_prescribed"].get_reshaped_view<const Pack**>();
  auto ni_activated    = m_p3_fields_in["ni_activated"].get_reshaped_view<const Pack**>();
  auto inv_qc_relvar   = m_p3_fields_in["inv_qc_relvar"].get_reshaped_view<const Pack**>();
  auto pmid            = m_p3_fields_in["pmid"].get_reshaped_view<const Pack**>();
  auto dp              = m_p3_fields_in["dp"].get_reshaped_view<const Pack**>();
  auto qv_prev         = m_p3_fields_out["qv_prev"].get_reshaped_view<Pack**>();
  auto T_prev          = m_p3_fields_out["T_prev"].get_reshaped_view<Pack**>();
  // --Diagnostic Outputs
  view_1d precip_liq_surf("precip_liq_surf",m_num_cols);
  view_1d precip_ice_surf("precip_ice_surf",m_num_cols);
  view_2d qv2qi_depos_tend("qv2qi_depos_tend",m_num_cols,m_num_levs);
  view_2d rho_qi("rho_qi",m_num_cols,m_num_levs);
  view_2d precip_liq_flux("precip_liq_flux",m_num_cols,m_num_levs);
  view_2d precip_ice_flux("precip_ice_flux",m_num_cols,m_num_levs);
  auto mu_c               = m_p3_fields_out["mu_c"].get_reshaped_view<Pack**>();
  auto lamc               = m_p3_fields_out["lamc"].get_reshaped_view<Pack**>();
  auto diag_eff_radius_qc = m_p3_fields_out["diag_eff_radius_qc"].get_reshaped_view<Pack**>();
  auto diag_eff_radius_qi = m_p3_fields_out["diag_eff_radius_qi"].get_reshaped_view<Pack**>();
  auto precip_total_tend  = m_p3_fields_out["precip_total_tend"].get_reshaped_view<Pack**>();
  auto nevapr             = m_p3_fields_out["nevapr"].get_reshaped_view<Pack**>();
  auto qr_evap_tend       = m_p3_fields_out["qr_evap_tend"].get_reshaped_view<Pack**>();
  // --Infrastructure
  // dt is passed to run
  m_it++;
  Int its = 0;
  Int ite = m_num_cols-1;
  Int kts = 0;
  Int kte = m_num_levs-1;
  bool do_predict_nc = true;     // Hard-coded for now, TODO: make this a runtime option 
  bool do_prescribed_CCN = true; // Hard-coded for now, TODO: make this a runtime option
  sview_2d col_location("col_location", m_num_cols, 3);
  // --History Only
  auto liq_ice_exchange = m_p3_fields_out["liq_ice_exchange"].get_reshaped_view<Pack**>();
  auto vap_liq_exchange = m_p3_fields_out["vap_liq_exchange"].get_reshaped_view<Pack**>();
  auto vap_ice_exchange = m_p3_fields_out["vap_ice_exchange"].get_reshaped_view<Pack**>();

  // Variables needed, but not passed to P3
  auto ast    = m_p3_fields_in["ast"].get_reshaped_view<const Pack**>();
  auto zi     = m_p3_fields_in["zi"].get_reshaped_view<const Pack**>();
  auto T_atm  = m_p3_fields_out["T_atm"].get_reshaped_view<Pack**>();

  Real mucon  = 5.3;
  Real dcon   = 25.0 * pow(10.0,-6);
  Real qsmall = pow(10.0,-14); 
  Real mincld = 0.0001;  // TODO: These should be stored somewhere as more universal constants.  Or maybe in the P3 class hpp

  Kokkos::deep_copy(mu_c,Spack(mucon));
  Kokkos::deep_copy(lamc,Spack( (mucon-1.0)/dcon ));
  Kokkos::deep_copy(cld_frac_l,Spack(mincld));
  Kokkos::deep_copy(cld_frac_i,Spack(mincld));
  Kokkos::deep_copy(cld_frac_r,Spack(mincld));
  using physics    = scream::physics::Functions<Real, DefaultDevice>;
//  using KT = KokkosTypes<DefaultDevice>;
//  using ExeSpace = typename KT::ExeSpace;
//  using MemberType = typename KT::MemberType;
//  const auto policy = ekat::ExeSpaceUtils<ExeSpace>::get_default_team_policy(1,1);
  // TODO: Improve the performance of this step.  Currently just using a parallel_for loop over
  // 1 to ensure code will work on GPU.  Perhaps need to use ekat::subview.
  Kokkos::parallel_for(
    "set_p3_local_vars",
    1,
    KOKKOS_LAMBDA(const int& i_dum) {
    for (int icol=0;icol<m_num_cols;icol++)
    {
      for (int jlev=0;jlev<m_num_levs;jlev++)
      {
        int ipack = jlev / Spack::n;
        int ivec  = jlev % Spack::n;
        int ipack_p1 = (jlev+1) / Spack::n;
        int ivec_p1  = (jlev+1) % Spack::n;
        int ipack_m1 = (jlev-1) / Spack::n;
        int ivec_m1  = (jlev-1) % Spack::n;
        
        const Spack opmid(pmid(icol,ipack)[ivec]);
        const Spack oT_atm(T_atm(icol,ipack)[ivec]);
        const Spack zi_top(zi(icol,ipack)[ivec]);
        const Spack zi_bot(zi(icol,ipack_p1)[ivec_p1]);
        
        auto oexner  = physics::get_exner(opmid,Smask(true));
        exner(icol,ipack)[ivec] = oexner[0];

        auto oth = physics::T_to_th(oT_atm,oexner,Smask(true));
        th_atm(icol,ipack)[ivec] = oth[0];

        auto odz = physics::get_dz(zi_top,zi_bot,Smask(true));
        dz(icol,ipack)[ivec] = odz[0];

        if (ast(icol,ipack)[ivec]>mincld)
        {
          cld_frac_l(icol,ipack)[ivec] = ast(icol,ipack)[ivec]; 
        }
        cld_frac_i(icol,ipack)[ivec] = cld_frac_l(icol,ipack)[ivec];
        cld_frac_r(icol,ipack)[ivec] = cld_frac_l(icol,ipack)[ivec];
        if (jlev != 0)
        {
          if (qr(icol,ipack)[ivec]>=qsmall or qi(icol,ipack)[ivec]>=qsmall)
          {
            cld_frac_r(icol,ipack)[ivec] = ast(icol,ipack_m1)[ivec_m1]>cld_frac_r(icol,ipack)[ivec] ? 
                                             ast(icol,ipack_m1)[ivec_m1] :
                                             cld_frac_r(icol,ipack)[ivec];
          }
        }
      }
    }
  });
//  Kokkos::fence();
// --Eventually delete from here...
  Real q_before;
  Kokkos::parallel_reduce(
    "q_before",
    1,
    KOKKOS_LAMBDA(const int& i_dum, Real& sum) {
  for (int i_col=0;i_col<m_num_cols;i_col++)
  {
    for (int i_lev=0;i_lev<m_num_levs;i_lev++)
    {
      int ipack = i_lev / Spack::n;
      int ivec  = i_lev % Spack::n;
      sum += (qv(i_col,ipack)[ivec] + qc(i_col,ipack)[ivec] + qr(i_col,ipack)[ivec] + qi(i_col,ipack)[ivec] + qm(i_col,ipack)[ivec]);
    }
  }
  },q_before);
// to here.
  // Pack our data into structs and ship it off to p3_main.
  P3F::P3PrognosticState prog_state{ qc, nc, qr, nr, qi, qm, ni, bm, qv, th_atm };
  prog_state.dump();
  P3F::P3DiagnosticInputs diag_inputs{ nc_nuceat_tend, nccn_prescribed, ni_activated, inv_qc_relvar, 
                                       cld_frac_i, cld_frac_l, cld_frac_r, pmid, dz, dp, exner, qv_prev, T_prev };
  P3F::P3DiagnosticOutputs diag_outputs{ mu_c, lamc, qv2qi_depos_tend, precip_liq_surf,
                                         precip_ice_surf, diag_eff_radius_qc, diag_eff_radius_qi, rho_qi,
                                         precip_total_tend, nevapr, qr_evap_tend, precip_liq_flux, precip_ice_flux };
  diag_outputs.dump();
  P3F::P3Infrastructure infrastructure{ dt, m_it, its, ite, kts, kte, do_predict_nc, do_prescribed_CCN, col_location };
  P3F::P3HistoryOnly history_only{ liq_ice_exchange, vap_liq_exchange, vap_ice_exchange };
  history_only.dump();

  auto elapsed_microsec = P3F::p3_main(prog_state, diag_inputs, diag_outputs, infrastructure,
                                       history_only, m_num_cols, m_num_levs);
// Eventually delete from here...
  Real q_after; 
  Kokkos::parallel_reduce(
    "q_after",
    1,
    KOKKOS_LAMBDA(const int& i_dum, Real& sum) {
  for (int i_col=0;i_col<m_num_cols;i_col++)
  {
    for (int i_lev=0;i_lev<m_num_levs;i_lev++)
    {
      int ipack = i_lev / Spack::n;
      int ivec  = i_lev % Spack::n;
      sum += (qv(i_col,ipack)[ivec] + qc(i_col,ipack)[ivec] + qr(i_col,ipack)[ivec] + qi(i_col,ipack)[ivec] + qm(i_col,ipack)[ivec]);
      qv_prev(i_col,ipack)[ivec] = qv(i_col,ipack)[ivec];
      T_prev(i_col,ipack)[ivec] = T_atm(i_col,ipack)[ivec];
    }
  }
  },q_after);
  printf("ASD = q_diff::  %.10e, %.10e, %.10e -- %d, %d\n",q_before,q_after,q_after-q_before,m_num_cols,m_num_levs);
// to here.

  // Copy outputs back to device
  for (auto& it : m_p3_fields_out) {
    Kokkos::deep_copy(it.second.get_view(),m_p3_host_views_out.at(it.first));
  }

  // Get a copy of the current timestamp (at the beginning of the step) and
  // advance it, updating the p3 fields.
  auto ts = timestamp();
  ts += dt;
//  m_p3_fields_out.at("q").get_header().get_tracking().update_time_stamp(ts); TODO: Delete, keep as an example
}

// =========================================================================================
void P3Microphysics::finalize_impl()
{
  // Do nothing for now
}

// =========================================================================================
void P3Microphysics::register_fields (FieldRepository<Real>& field_repo) const {
  for (auto& fid : m_required_fields) {
    field_repo.register_field<Pack>(fid);
  }
  for (auto& fid : m_computed_fields) {
    field_repo.register_field<Pack>(fid);
  }
}

void P3Microphysics::set_required_field_impl (const Field<const Real>& f) {
  // Store a copy of the field. We need this in order to do some tracking checks
  // at the beginning of the run call. Other than that, there would be really
  // no need to store a scream field here; we could simply set the view ptr
  // in the Homme's view, and be done with it.
  const auto& name = f.get_header().get_identifier().name();
  m_p3_fields_in.emplace(name,f);
  m_p3_host_views_in[name] = Kokkos::create_mirror_view(f.get_view());
  m_raw_ptrs_in[name] = m_p3_host_views_in[name].data();

  // Add myself as customer to the field
  add_me_as_customer(f);
}

void P3Microphysics::set_computed_field_impl (const Field<      Real>& f) {
  // Store a copy of the field. We need this in order to do some tracking updates
  // at the end of the run call. Other than that, there would be really
  // no need to store a scream field here; we could simply set the view ptr
  // in the Homme's view, and be done with it.
  const auto& name = f.get_header().get_identifier().name();
  m_p3_fields_out.emplace(name,f);
  m_p3_host_views_out[name] = Kokkos::create_mirror_view(f.get_view());
  m_raw_ptrs_out[name] = m_p3_host_views_out[name].data();

  // Add myself as provider for the field
  add_me_as_provider(f);
}

} // namespace scream
