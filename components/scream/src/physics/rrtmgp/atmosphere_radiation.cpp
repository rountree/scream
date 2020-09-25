#include "ekat/ekat_assert.hpp"
#include "physics/rrtmgp/scream_rrtmgp_interface.hpp"
#include "physics/rrtmgp/atmosphere_radiation.hpp"

namespace scream {
    RRTMGPRadiation::RRTMGPRadiation (const ekat::Comm& comm, const ekat::ParameterList& params) : m_rrtmgp_comm (comm), m_rrtmgp_params (params) {
        /*
         * Anything that can be initialized without grid information can be initialized here.
         * I.e., universal constants, options, etc.
         */
    }  // RRTMGPRadiation::RRTMGPRadiation

    void RRTMGPRadiation::set_grids(const std::shared_ptr<const GridsManager> grids_manager) {

        using namespace units;

        auto kgkg = kg/kg;
        kgkg.set_string("kg/kg");
        auto m3 = m * m * m;
        m3.set_string("m3");
        auto Wm2 = W / m / m;
        Wm2.set_string("Wm2");

        auto VL  = FieldTag::VerticalLevel;
        auto COL = FieldTag::Column;
        constexpr int NVL = 72;  /* TODO THIS NEEDS TO BE CHANGED TO A CONFIGURABLE */

        auto grid = grids_manager->get_grid("Physics");
        const int num_dofs = grid->get_num_local_dofs();
        const int nc = num_dofs;

        // Set up dimension layouts
        FieldLayout scalar2d_layout     { {COL   }, {nc    } };
        FieldLayout scalar3d_layout_mid { {COL,VL}, {nc,NVL} };
        FieldLayout scalar3d_layout_int { {COL,VL}, {nc,NVL+1} };
        FieldLayout gas_layout          { {GAS,COL,VL}, {ngas,nc,NVL} };

        // Set required (input) fields here
        m_required_fields.emplace("pmid" , scalar3d_layout_mid, Pa, grid->name());
        m_required_fields.emplace("pint", scalar3d_layout_int, Pa, grid->name());
        m_required_fields.emplace("tmid" , scalar3d_layout_mid, K , grid->name());
        m_required_fields.emplace("tint" , scalar3d_layout_int, K , grid->name());
        m_required_fields.emplace("col_dry", scalar3d_layout_mid, kgkg, grid->name());
        //m_required_fields.emplace("gas_names", gas_names_layout, none, grid->name());
        m_required_fields.emplace("gas_vmr", gas_layout, kgkg, grid->name());
        m_required_fields.emplace("sfc_alb_dir", scalar2d_band_layout, none, grid->name());
        m_required_fields.emplace("sfc_alb_dif", scalar2d_band_layout, none, grid->name());
        m_required_fields.emplace("mu0", scalar2d_layout, none, grid->name());
        m_required_fields.emplace("lwp", scalar3d_layout_mid, kg/m3, grid->name());
        m_required_fields.emplace("iwp", scalar3d_layout_mid, kg/m3, grid->name());
        m_required_fields.emplace("rel", scalar3d_layout_mid, micron, grid->name());
        m_required_fields.emplace("rei", scalar3d_layout_mid, micron, grid->name());

        // Set computed (output) fields
        m_computed_fields.emplace("flux_sw_dn", scalar3d_layout_int, Wm2, grid->name());
        m_computed_fields.emplace("flux_sw_up", scalar3d_layout_int, Wm2, grid->name());
        m_computed_fields.emplace("flux_lw_dn", scalar3d_layout_int, Wm2, grid->name());
        m_computed_fields.emplace("flux_lw_up", scalar3d_layout_int, Wm2, grid->name());

    }  // RRTMGPRadiation::set_grids
    void RRTMGPRadiation::initialize_impl(const util::TimeStamp& t0) {
        rrtmgp::rrtmgp_initialize();

        // We may have to init some fields from within RRTMGP. This can be the case in a RRTMGP standalone run.
        // Some options:
        //  - we can tell RRTMGP it can init all inputs or specify which ones it can init. We call the
        //    resulting list of inputs the 'initializable' (or initable) inputs. The default is
        //    that no inputs can be inited.
        //  - we can request that RRTMGP either inits no inputs or all of the initable ones (as specified
        //    at the previous point). The default is that RRTMGP must be in charge of init ing ALL or NONE
        //    of its initable inputs.
        // Recall that:
        //  - initable fields may not need initialization (e.g., some other atm proc that
        //    appears earlier in the atm dag might provide them).
        std::vector<std::string> rrtmgp_inputs = {"pmid","pint","tmid","tint","col_dry","gas_vmr","sfc_alb_dir","sfc_alb_dif","mu0","lwp","iwp","rel","rei"};
        using strvec = std::vector<std::string>;
        const strvec& allowed_to_init = m_rrtmgp_params.get<strvec>("Initializable Inputs",strvec(0));
        const bool can_init_all = m_rrtmgp_params.get<bool>("Can Initialize All Inputs", false);
        const bool init_all_or_none = m_rrtmgp_params.get<bool>("Must Init All Inputs Or None", true);
        const strvec& initable = can_init_all ? rrtmgp_inputs : allowed_to_init;
        if (initable.size()>0) {
            bool all_inited = true, all_uninited = true;
            for (const auto& name : initable) {
                const auto& f = m_rrtmgp_fields_in.at(name);
                const auto& track = f.get_header().get_tracking();
                if (track.get_init_type()==InitType::None) {
                    // Nobody claimed to init this field. RRTMGPInputsInitializer will take care of it
                    m_initializer->add_me_as_initializer(f);
                    all_uninited &= true;
                    all_inited &= false;
                } else {
                    all_uninited &= false;
                    all_inited &= true;
                }
            }
    
            // In order to gurantee some consistency between inputs, it is best if RRTMGP
            // initializes either none or all of the inputs.
            scream_require_msg (!init_all_or_none || all_inited || all_uninited,
                                "Error! Some rrtmgp inputs were marked to be inited by RRTMGP, while others weren't.\n"
                                "       RRTMGP was requested to init either all or none of the inputs.\n");
        }
    }
    void RRTMGPRadiation::run_impl      (const Real dt) {
        rrtmgp::rrtmgp_main(); 
    }
    void RRTMGPRadiation::finalize_impl  () {
        rrtmgp::rrtmgp_finalize();
    }

    // Register all required and computed field in the field repository
    void RRTMGPRadiation::register_fields(FieldRepository<Real>& field_repo) const {
        for (auto& fid: m_required_fields) { field_repo.register_field(fid); }
        for (auto& fid: m_computed_fields) { field_repo.register_field(fid); }
    }

    void RRTMGPRadiation::set_required_field_impl(const Field<const Real>& f) {
        const auto& name = f.get_header().get_identifier().name();
        m_rrtmgp_fields_in.emplace(name,f);
        m_rrtmgp_host_views_in[name] = Kokkos::create_mirror_view(f.get_view());
        m_raw_ptrs_in[name] = m_rrtmgp_host_views_in[name].data();

        // Add myself as customer to the field
        add_me_as_customer(f);

    }

    void RRTMGPRadiation::set_computed_field_impl(const Field<      Real>& f) {
        const auto& name = f.get_header().get_identifier().name();
        m_rrtmgp_fields_out.emplace(name,f);
        m_rrtmgp_host_views_out[name] = Kokkos::create_mirror_view(f.get_view());
        m_raw_ptrs_out[name] = m_rrtmgp_host_views_out[name].data();

        // Add myself as provider for the field
        add_me_as_provider(f);
    }

}  // namespace scream
