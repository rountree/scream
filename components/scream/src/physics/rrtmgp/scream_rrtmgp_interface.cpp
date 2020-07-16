#include "scream_rrtmgp_interface.hpp"
#include "mo_load_coefficients.h"
#include "mo_gas_concentrations.h"
#include "mo_gas_optics_rrtmgp.h"
#include "const.h"

namespace scream {
    namespace rrtmgp {
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
            GasOpticsRRTMGP k_dist_sw;
            std::string coefficients_file_sw = "./data/rrtmgp-data-sw-g224-2018-12-04.nc";
            std::string coefficients_file_lw = "./data/rrtmgp-data-lw-g256-2018-12-04.nc";
            load_and_init(k_dist_sw, coefficients_file_sw, gas_concs);

            // Verify that we loaded absorption coefficient data properly
            //std::cout << k_dist_sw.press_ref << "\n";
        }

        void rrtmgp_finalize() {}

        void rrtmgp_main() {}

    }  // namespace rrtmgp
}  // namespace scream
