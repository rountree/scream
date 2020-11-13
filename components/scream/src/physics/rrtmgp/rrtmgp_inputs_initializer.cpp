#include "physics/rrtmgp/rrtmgp_inputs_initializer.hpp"
#include "physics/rrtmgp/scream_rrtmgp_interface.hpp"
#include "rrtmgp_test_utils.hpp"
#include "YAKL.h"

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

        // These are actually outputs, but we need to initialize them in the ad
        count += m_fields.count("sw_flux_up");
        count += m_fields.count("sw_flux_dn");
        count += m_fields.count("sw_flux_dn_dir");
        count += m_fields.count("lw_flux_up");
        count += m_fields.count("lw_flux_dn");
  
        if (count==0) {
          return;
        }
  
        EKAT_REQUIRE_MSG(
            count==18,
            "Error! RRTMGPInputsInitializer is expected to init\n"
            "       pmid, pint, tmid, tint, col_dry, gas_vmr, sfc_alb_dir, sfc_alb_dif,\n"
            "       mu0, lwp, iwp, rel, rei,\n"
            "       sw_flux_up, sw_flux_dn, sw_flux_dn_dir, lw_flux_up, lw_flux_dn,\n"
            "       but only " + std::to_string(count) + " of those have been found.\n"
            "       Please, check the atmosphere processes you are using,\n"
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
        auto d_sw_flux_up = m_fields.at("sw_flux_up").get_view();
        auto d_sw_flux_dn = m_fields.at("sw_flux_dn").get_view();
        auto d_sw_flux_dn_dir = m_fields.at("sw_flux_dn_dir").get_view();
        auto d_lw_flux_up = m_fields.at("lw_flux_up").get_view();
        auto d_lw_flux_dn = m_fields.at("lw_flux_dn").get_view();
  
        // Create host mirror views
        auto h_pmid = Kokkos::create_mirror_view(d_pmid);

        // RRTMGP initialization routine needs YAKL Fortran-style arrays, which
        // we can map from the C-style Kokkos views. Note that this assumes that
        // the Kokkos views are LayoutRight, and that dimension ordering is
        // opposite what RRTMGP expects (that is, (nx,ny,nz) -> (nz,ny,nx))
        // Read in dummy Garand atmosphere; if this were an actual model simulation,
        // these would be passed as inputs to the driver
        // NOTE: set ncol to size of col_flx dimension in the input file. This is so
        // that we can compare to the reference data provided in that file. Note that
        // this will copy the first column of the input data (the first profile) ncol
        // times. We will then fill some fraction of these columns with clouds for
        // the test problem.
        int ngas =  8;
        int ncol = 10;
        int nlay = 42;
        int nswbands = 14;
        yakl::Array<double,2,memDevice,yakl::styleFortran> p_lay  ("p_lay", d_pmid.data(), ncol, nlay);
        yakl::Array<double,2,memDevice,yakl::styleFortran> t_lay  ("t_lay", d_tmid.data(), ncol, nlay);
        yakl::Array<double,2,memDevice,yakl::styleFortran> p_lev  ("p_lev", d_pint.data(), ncol, nlay+1);
        yakl::Array<double,2,memDevice,yakl::styleFortran> t_lev  ("t_lev", d_tint.data(), ncol, nlay+1);
        yakl::Array<double,2,memDevice,yakl::styleFortran> col_dry("col_dry", d_col_dry.data(), ncol, nlay);
        yakl::Array<double,3,memDevice,yakl::styleFortran> gas_vmr("gas_vmr", d_gas_vmr.data(), ngas, ncol, nlay);
        yakl::Array<double,2,memDevice,yakl::styleFortran> sfc_alb_dir("sfc_alb_dir", d_sfc_alb_dir.data(), nswbands, ncol);
        yakl::Array<double,2,memDevice,yakl::styleFortran> sfc_alb_dif("sfc_alb_dif", d_sfc_alb_dif.data(), nswbands, ncol);
        yakl::Array<double,1,memDevice,yakl::styleFortran> mu0("mu0", d_mu0.data(), ncol);
        yakl::Array<double,2,memDevice,yakl::styleFortran> lwp("lwp", d_lwp.data(), ncol, nlay);
        yakl::Array<double,2,memDevice,yakl::styleFortran> iwp("iwp", d_iwp.data(), ncol, nlay);
        yakl::Array<double,2,memDevice,yakl::styleFortran> rel("rel", d_rel.data(), ncol, nlay);
        yakl::Array<double,2,memDevice,yakl::styleFortran> rei("rei", d_rei.data(), ncol, nlay);

        // Read in values from input file
        std::string inputfile = "data/rrtmgp-allsky.nc";
        GasConcs gas_concs;
        rrtmgpTest::dummy_atmos(
            inputfile, ncol, p_lay, t_lay, p_lev, t_lev, gas_concs, col_dry,
            sfc_alb_dir, sfc_alb_dif, mu0,
            lwp, iwp, rel, rei
        );

        // Pull gas volume mixing ratios out of gas_concs to populate gas_vmr array.
        // The reason for this is we want to use the RRTMGP routine to read the
        // data, but we want to store the data as something easily representable
        // with Kokkos views or yakl arrays.
        //gas_vmr = gas_concs.concs;
        string1d gas_names("gas_names", ngas); 
        for (int igas = 1; igas <= ngas; igas++) {
            gas_names(igas) = gas_concs.gas_name(igas);
            for (int icol = 1; icol <= ncol; icol++) {
                for (int ilay = 1; ilay <= nlay; ilay++) {
                    gas_vmr(igas,icol,ilay) = gas_concs.concs(icol,ilay,igas);
                }
            }
        }
        // Check values
        for (int ilay = 1; ilay <= nlay; ilay++) {
            std::cout << "p_lay from init: " << p_lay(1,ilay) << std::endl;
            //std::cout << "gas_vmr at init: " << gas_vmr(1,1,ilay) << std::endl;
        }

        // Copy back to host?
        Kokkos::deep_copy(h_pmid, d_pmid); 
    }
} // namespace scream
