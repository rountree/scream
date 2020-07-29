#include "catch2/catch.hpp"
#include "physics/rrtmgp/scream_rrtmgp_interface.hpp"
#include "netcdf.h"
#include "mo_gas_concentrations.h"
#include "mo_garand_atmos_io.h"
namespace {

    TEST_CASE("rrtmgp_read_netcdf", "") {

        /* Try to read absorption coefficient data */
        int ncid;
        
        // Check that we can open aborption coefficient files
        const char *coefficients_file_sw = "./data/rrtmgp-data-sw-g224-2018-12-04.nc";
        const char *coefficients_file_lw = "./data/rrtmgp-data-lw-g256-2018-12-04.nc";
        REQUIRE(nc_open(coefficients_file_sw, NC_NOWRITE, &ncid) == NC_NOERR);

        // Try reading something relevant

        // If we got this far, we won
        REQUIRE(1 == 1);
    }

    TEST_CASE("rrtmgp_init", "") {
        scream::rrtmgp::rrtmgp_initialize();
        REQUIRE(1 == 1);

        // Sanity check to see if we were able to load the data correctly. 
        // Check integer part of reference pressure to avoid floating point 
        // differences.
        REQUIRE(int(scream::rrtmgp::k_dist_sw.press_ref(1)) == 109663);
        REQUIRE(int(scream::rrtmgp::k_dist_sw.press_ref(size(scream::rrtmgp::k_dist_sw.press_ref,1))) == 1);
    }

    TEST_CASE("rrtmgp_run", "") {
        // Initialize absorption coefficients
        scream::rrtmgp::rrtmgp_initialize();

        // Read in dummy Garand atmosphere; if this were an actual model simulation, 
        // these would be passed as inputs to the driver
        std::string inputfile = "./data/rrtmgp-allsky.nc";
        real2d p_lay;
        real2d t_lay;
        real2d p_lev;
        real2d t_lev;
        real2d col_dry;
        GasConcs gas_concs;
        int ncol = 1;
        read_atmos(inputfile, p_lay, t_lay, p_lev, t_lev, gas_concs, col_dry, ncol);

        // Check that data was loaded properly; just sanity check a few values
        REQUIRE(int(p_lay(1,1)) == 100933);
        REQUIRE(int(p_lay(1,size(p_lay))) == 19);

        // Get dimension sizes
        int nlay = p_lay.dimension[1];

        // Setup boundary conditions, solar zenith angle, etc
        // NOTE: this stuff would come from the model in a real run
        int nbndsw = scream::rrtmgp::k_dist_sw.get_nband();
        real2d sfc_alb_dir("sfc_alb_dir", nbndsw, ncol);
        real2d sfc_alb_dif("sfc_alb_dif", nbndsw, ncol);

        // Ocean-ish values for surface albedos, just for example
        memset(sfc_alb_dir , 0.06_wp );
        memset(sfc_alb_dif , 0.06_wp );

        // Pick a solar zenith angle; this should come from the model
        real1d mu0("mu0", ncol);
        memset(mu0, 0.86_wp );

        // Run RRTMGP code on dummy atmosphere; this might get ugly
        // Inputs should be atmosphere state, outputs should be fluxes
        // TODO: should absorption coefficients be an input, or should that be initialized
        // and kept in the scream::rrtmgp namespace?
        scream::rrtmgp::rrtmgp_main(
                p_lay, t_lay, p_lev, t_lev, gas_concs, col_dry, 
                sfc_alb_dir, sfc_alb_dif, mu0);
 
        // Check fluxes against reference; note that input file contains reference fluxes
        //read_fluxes(inputfile, flux_up_sw, flux_dn_sw, flux_up_lw, flux_up_lw, flux_dn_lw);

        // ALTERNATIVELY: create a single or two-layer atmosphere to do a dummy calc
    }

} // empty namespace
