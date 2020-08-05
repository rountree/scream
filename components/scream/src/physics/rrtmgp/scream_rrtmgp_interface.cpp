#include "scream_rrtmgp_interface.hpp"
#include "mo_load_coefficients.h"
#include "mo_gas_concentrations.h"
#include "mo_gas_optics_rrtmgp.h"
#include "mo_rte_sw.h"
#include "const.h"

namespace scream {
    namespace rrtmgp {

        /* 
         * Objects containing k-distribution information need to be initialized
         * once and then persist throughout the life of the program, so we
         * declare them here within the rrtmgp namespace.
         */
        GasOpticsRRTMGP k_dist_sw;
        GasOpticsRRTMGP k_dist_lw;

        /* 
         * Define some dummy routines so we can start working on the interface
         * between SCREAM and RRTMGP
         */
        void rrtmgp_initialize() {

            /* Initialize YAKL */
            yakl::init();

            /* 
             * Names of active gases; this should come from the calling program 
             * NOTE: YAKL uses 1-based indexing!!!!
             */
            int ngas = 8;
            string1d gas_names("gas_names",ngas);
            gas_names(1) = std::string("h2o");
            gas_names(2) = std::string("co2");
            gas_names(3) = std::string("o3" );
            gas_names(4) = std::string("n2o");
            gas_names(5) = std::string("co" );
            gas_names(6) = std::string("ch4");
            gas_names(7) = std::string("o2" );
            gas_names(8) = std::string("n2" );

            // Initialize GasConcs object but populate with dummy values 
            int ncol = 10;
            int nlay = 2;
            GasConcs gas_concs;
            gas_concs.init(gas_names,ncol,nlay);
            for (int igas = 1; igas < ngas+1; igas++) {
                gas_concs.set_vmr(gas_names(igas), 0.0);
            }

            // Load and initialize absorption coefficient data
            std::string coefficients_file_sw = "./data/rrtmgp-data-sw-g224-2018-12-04.nc";
            std::string coefficients_file_lw = "./data/rrtmgp-data-lw-g256-2018-12-04.nc";
            load_and_init(k_dist_sw, coefficients_file_sw, gas_concs);
            load_and_init(k_dist_lw, coefficients_file_lw, gas_concs);

            // Verify that we loaded absorption coefficient data properly
            //std::cout << k_dist_sw.press_ref << "\n";
        }

        void rrtmgp_finalize() {}

        void rrtmgp_main(
                real2d &p_lay, real2d &t_lay, real2d &p_lev, real2d &t_lev, 
                GasConcs &gas_concs, real2d &col_dry,
                real2d &sfc_alb_dir, real2d &sfc_alb_dif, real1d &mu0, OpticalProps2str &clouds,
                FluxesBroadband &fluxes_sw, FluxesBroadband &fluxes_lw) {

            // Do shortwave
            rrtmgp_sw(
                    k_dist_sw, p_lay, t_lay, p_lev, t_lev, gas_concs, col_dry, 
                    sfc_alb_dir, sfc_alb_dif, mu0, clouds, fluxes_sw);

            // Do longwave
            //rrtmgp_lw(
            //        k_dist_lw, p_lay, t_lay, p_lev, t_lev, gas_concs, col_dry,
            //        fluxes_lw);
            
            // Calculate heating rates
        }

        void rrtmgp_sw(
                GasOpticsRRTMGP &k_dist, 
                real2d &p_lay, real2d &t_lay, real2d &p_lev, real2d &t_lev, 
                GasConcs &gas_concs, real2d &col_dry,
                real2d &sfc_alb_dir, real2d &sfc_alb_dif, real1d &mu0, OpticalProps2str &clouds,
                FluxesBroadband &fluxes) {

            // Get problem sizes
            int nbnd = k_dist.get_nband();
            int ngpt = k_dist.get_ngpt();
            int ncol = p_lay.dimension[0];
            int nlay = p_lay.dimension[1];

            // Allocate space for optical properties
            OpticalProps2str optics;
            optics.alloc_2str(ncol, nlay, k_dist);

            // Do gas optics
            real2d toa_flux("toa_flux", ncol, ngpt);
            bool top_at_1 = p_lay(1, 1) < p_lay(1, nlay);
            k_dist.gas_optics(top_at_1, p_lay, p_lev, t_lay, gas_concs, optics, toa_flux);

            // Combine gas and cloud optics
            clouds.delta_scale();
            clouds.increment(optics);

            // Compute fluxes
            rte_sw(optics, top_at_1, mu0, toa_flux, sfc_alb_dir, sfc_alb_dif, fluxes);
        }

    }  // namespace rrtmgp
}  // namespace scream
