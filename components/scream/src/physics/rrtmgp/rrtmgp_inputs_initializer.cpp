#include "physics/rrtmgp/rrtmgp_inputs_initializer.hpp"
#include "physics/rrtmgp/scream_rrtmgp_interface.hpp"

#include <array>

namespace scream {

    void RRTMGPInputsInitializer::add_field (const field_type &f) {
        const auto& id = f.get_header().get_identifier();
        m_fields.emplace(id.name(),f);
        m_fields_id.insert(id);
    }

    // =========================================================================================
    void RRTMGPInputsInitializer::initialize_fields () {
        // Safety check: if we're asked to init anything at all,
        // then we should have been asked to init 13 fields.
        int count = 0;
        count += m_fields.count("pmid");
        count += m_fields.count("pint");
        count += m_fields.count("tmid");
        count += m_fields.count("tint");
        count += m_fields.count("col_dry");
        count += m_fields.count("gas_vmr");
        count += m_fields.count("sfc_alb_dir");
        count += m_fields.count("sfc_alb_dif");
        count += m_fields.count("mu0");
        count += m_fields.count("lwp");
        count += m_fields.count("iwp");
        count += m_fields.count("rel");
        count += m_fields.count("rei");
  
        if (count==0) {
          return;
        }
  
        EKAT_REQUIRE_MSG(
            count==13,
            "Error! RRTMGPInputsInitializer is expected to init\n"
            "       pmid, pint, tmid, tint, col_dry, gas_vmr, sfc_alb_dir, sfc_alb_dif, mu0, lwp, iwp, rel, rei,\n"
            "       but only " + std::to_string(count) + " of those have been found.\n"
            "       Please, check the atmosphere processes you are using,"
            "       and make sure they agree on who's initializing each field.\n"
        );
  
        // Get device views
        auto d_pmid = m_fields.at("pmid").get_view();
        auto d_pint = m_fields.at("pint").get_view();
        auto d_tmid = m_fields.at("tmid").get_view();
        auto d_tint = m_fields.at("tint").get_view();
        auto d_col_dry = m_fields.at("col_dry").get_view();
        auto d_gas_vmr = m_fields.at("gas_vmr").get_view();
        auto d_sfc_alb_dir = m_fields.at("sfc_alb_dir").get_view();
        auto d_sfc_alb_dif = m_fields.at("sfc_alb_dif").get_view();
        auto d_mu0 = m_fields.at("mu0").get_view();
        auto d_lwp = m_fields.at("lwp").get_view();
        auto d_iwp = m_fields.at("iwp").get_view();
        auto d_rel = m_fields.at("rel").get_view();
        auto d_rei = m_fields.at("rei").get_view();
  
        // Create host mirrors
        auto h_pmid = Kokkos::create_mirror_view(d_pmid);
        auto h_pint = Kokkos::create_mirror_view(d_pint);
        auto h_tmid = Kokkos::create_mirror_view(d_tmid);
        auto h_tint = Kokkos::create_mirror_view(d_tint);
        auto h_col_dry = Kokkos::create_mirror_view(d_col_dry);
        auto h_gas_vmr = Kokkos::create_mirror_view(d_gas_vmr);
        auto h_sfc_alb_dir = Kokkos::create_mirror_view(d_sfc_alb_dir);
        auto h_sfc_alb_dif = Kokkos::create_mirror_view(d_sfc_alb_dif);
        auto h_mu0 = Kokkos::create_mirror_view(d_mu0);
        auto h_lwp = Kokkos::create_mirror_view(d_lwp);
        auto h_iwp = Kokkos::create_mirror_view(d_iwp);
        auto h_rel = Kokkos::create_mirror_view(d_rel);
        auto h_rei = Kokkos::create_mirror_view(d_rei);
  
        // Get host mirrors raw pointers
        auto pmid = h_pmid.data();
        auto pint = h_pint.data();
        auto tmid = h_tmid.data();
        auto tint = h_tint.data();
        auto col_dry = h_col_dry.data();
        auto gas_vmr = h_gas_vmr.data();
        auto sfc_alb_dir = h_sfc_alb_dir.data();
        auto sfc_alb_dif = h_sfc_alb_dif.data();
        auto mu0 = h_mu0.data();
        auto lwp = h_lwp.data();
        auto iwp = h_iwp.data();
        auto rel = h_rel.data();
        auto rei = h_rei.data();
  
        // Call initialization routine
        //p3_standalone_init_f90 (q, T, zi, pmid, dpres, ast, ni_activated, nc_nuceat_tend);
        //std::string inputfile;
        //dummy_atmos(
        //    std::string inputfile,
        //    int ncol, real2d &p_lay, real2d &t_lay, real2d &p_lev, real2d &t_lev, GasConcs &gas_concs, real2d &col_dry,
        //    real2d &sfc_alb_dir, real2d &sfc_alb_dif, real1d &mu0,
        //    real2d &lwp, real2d &iwp, real2d &rel, real2d &rei
        //);

  
        // Deep copy back to device
        Kokkos::deep_copy(d_pmid,h_pmid);
        Kokkos::deep_copy(d_pint,h_pint);
        Kokkos::deep_copy(d_tmid,h_tmid);
        Kokkos::deep_copy(d_tint,h_tint);
        Kokkos::deep_copy(d_col_dry,h_col_dry);
        Kokkos::deep_copy(d_gas_vmr,h_gas_vmr);
        Kokkos::deep_copy(d_sfc_alb_dir,h_sfc_alb_dir);
        Kokkos::deep_copy(d_sfc_alb_dif,h_sfc_alb_dif);
        Kokkos::deep_copy(d_mu0,h_mu0);
        Kokkos::deep_copy(d_lwp,h_lwp);
        Kokkos::deep_copy(d_iwp,h_iwp);
        Kokkos::deep_copy(d_rel,h_rel);
        Kokkos::deep_copy(d_rei,h_rei);
    }

} // namespace scream
