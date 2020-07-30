#include "catch2/catch.hpp"
#include "physics/rrtmgp/scream_rrtmgp_interface.hpp"
#include "netcdf.h"
#include "mo_gas_concentrations.h"
#include "mo_garand_atmos_io.h"
#include "mo_load_cloud_coefficients.h"
#include "mo_fluxes.h"
namespace {

    OpticalProps2str dummy_clouds(std::string cloud_optics_file, GasOpticsRRTMGP &kdist, real2d &t_lay);
    void read_fluxes(
            std::string inputfile, 
            real2d &sw_flux_up, real2d &sw_flux_dn,
            real2d &lw_flux_up, real2d &lw_flux_dn);

    bool all_equals(real2d &arr1, real2d &arr2);

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
        // NOTE: set ncol to size of col_flx dimension in the input file. This is so
        // that we can compare to the reference data provided in that file. Note that
        // this will copy the first column of the input data (the first profile) ncol
        // times. We will then fill some fraction of these columns with clouds for
        // the test problem.
        std::string inputfile = "./data/rrtmgp-allsky.nc";
        real2d p_lay;
        real2d t_lay;
        real2d p_lev;
        real2d t_lev;
        real2d col_dry;
        GasConcs gas_concs;
        int ncol = 128;
        read_atmos(inputfile, p_lay, t_lay, p_lev, t_lev, gas_concs, col_dry, ncol);

        // Check that data was loaded properly; just sanity check a few values
        REQUIRE(int(p_lay(1,1)) == 100933);
        REQUIRE(int(p_lay(1,p_lay.dimension[1])) == 19);

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

        // Setup flux outputs; In a real model run, the fluxes would be
        // input/outputs into the driver (persisting between calls), and
        // we would just have to setup the pointers to them in the
        // FluxesBroadband object
        FluxesBroadband fluxes_sw;
        real2d sw_flux_up ("sw_flux_up" ,ncol,nlay+1);
        real2d sw_flux_dn ("sw_flux_dn" ,ncol,nlay+1);
        real2d sw_flux_dn_dir("sw_flux_dn_dir",ncol,nlay+1);
        fluxes_sw.flux_up = sw_flux_up;
        fluxes_sw.flux_dn = sw_flux_dn;
        fluxes_sw.flux_dn_dir = sw_flux_dn_dir;

        FluxesBroadband fluxes_lw;
        real2d lw_flux_up ("lw_flux_up" ,ncol,nlay+1);
        real2d lw_flux_dn ("lw_flux_dn" ,ncol,nlay+1);
        fluxes_lw.flux_up = lw_flux_up;
        fluxes_lw.flux_dn = lw_flux_dn;

        // Get dummy clouds so we can compare with reference fluxes
        // OR do clearsky problem?
        std::string cloud_optics_file_sw = "./data/rrtmgp-cloud-optics-coeffs-sw.nc";
        OpticalProps2str clouds = dummy_clouds(cloud_optics_file_sw, scream::rrtmgp::k_dist_sw, t_lay);

        // Run RRTMGP code on dummy atmosphere; this might get ugly
        // Inputs should be atmosphere state, outputs should be fluxes
        // TODO: should absorption coefficients be an input, or should that be initialized
        // and kept in the scream::rrtmgp namespace?
        scream::rrtmgp::rrtmgp_main(
                p_lay, t_lay, p_lev, t_lev, gas_concs, col_dry, 
                sfc_alb_dir, sfc_alb_dif, mu0,
                fluxes_sw, fluxes_lw);
 
        // Check fluxes against reference; note that input file contains reference fluxes
        read_fluxes(inputfile, sw_flux_up, sw_flux_dn, lw_flux_up, lw_flux_dn);

        REQUIRE(all_equals(sw_flux_up, fluxes_sw.flux_up));

        // ALTERNATIVELY: create a single or two-layer atmosphere to do a dummy calc
    }

    OpticalProps2str dummy_clouds(std::string cloud_optics_file, GasOpticsRRTMGP &kdist, real2d &t_lay) {

        // Problem sizes
        int ncol = t_lay.dimension[0];
        int nlay = t_lay.dimension[1];

        REQUIRE(ncol == 128);

        // Initialize optics
        OpticalProps2str clouds;
        clouds.init(kdist.get_band_lims_wavenumber());
        clouds.alloc_2str(ncol, nlay);  // this is dumb, why do we need to init and alloc separately?!

        // Initialize working optics class...this is kind of strange, but there's a separate
        // class that does RRTMGP cloud optics. The actual cloud optical properties will then
        // get copied over to the "clouds" instance of OpticalProps2str we created above. Not
        // sure why the CloudOptics class didn't just inherit from OpticalProps2str so we could
        // have used it directly in the calls to the driver routines.
        CloudOptics cloud_optics;
        load_cld_lutcoeff(cloud_optics, cloud_optics_file);

        // Return optics
        return clouds;
    }

    // Function to read fluxes from input file so we can compare our answers against the reference
    void read_fluxes(
            std::string inputfile, 
            real2d &sw_flux_up, real2d &sw_flux_dn,
            real2d &lw_flux_up, real2d &lw_flux_dn) {

        // Initialize netcdf reader
        yakl::SimpleNetCDF io;
        io.open(inputfile, yakl::NETCDF_MODE_READ);

        // Initialize arrays to hold fluxes
        int nlev = io.getDimSize("lev");
        int ncol = io.getDimSize("col_flx");
        sw_flux_up = real2d("sw_flux_up", ncol, nlev);
        sw_flux_dn = real2d("sw_flux_dn", ncol, nlev);
        lw_flux_up = real2d("lw_flux_up", ncol, nlev);
        lw_flux_dn = real2d("lw_flux_dn", ncol, nlev);

        // Read data
        io.read(sw_flux_up, "sw_flux_up");
        io.read(sw_flux_dn, "sw_flux_dn");
        io.read(lw_flux_up, "lw_flux_up");
        io.read(lw_flux_dn, "lw_flux_dn");
    }
   
    bool all_equals(real2d &arr1, real2d &arr2) {
        int nx = arr1.dimension[0];
        int ny = arr2.dimension[1];
        for (int i=0; i<nx; i++) {
            for (int j=0; j<ny; j++) {
                if (arr1(i,j) != arr2(i,j)) {
                    return false;
                }
            }
        }
        return true;
    }

} // empty namespace
