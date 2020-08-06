#include <iostream>
#include <cmath>
#include "catch2/catch.hpp"
#include "physics/rrtmgp/scream_rrtmgp_interface.hpp"
#include "netcdf.h"
#include "mo_gas_concentrations.h"
#include "mo_garand_atmos_io.h"
#include "mo_load_cloud_coefficients.h"
#include "mo_fluxes.h"
#include "FortranIntrinsics.h"
namespace {

    OpticalProps2str dummy_clouds_sw(std::string cloud_optics_file, GasOpticsRRTMGP &kdist, real2d &p_lay, real2d &t_lay);
    OpticalProps1scl dummy_clouds_lw(std::string cloud_optics_file, GasOpticsRRTMGP &kdist, real2d &p_lay, real2d &t_lay);
    void read_fluxes(
            std::string inputfile, 
            real2d &sw_flux_up, real2d &sw_flux_dn, real2d &sw_flux_dn_dir,
            real2d &lw_flux_up, real2d &lw_flux_dn);

    // TODO: use YAKL intrinsics for this; this won't work on the GPU
    bool all_equals(real2d &arr1, real2d &arr2) {
        double tolerance = 0.01;
        /*
        real2d residual = arr1 - arr2;
        if (yakl::fortran::anyGT(residual, tolerance) || yakl::fortran::anyLT(residual, -tolerance)) {
            printf("max(arr1 - arr2) = %f\n", yakl::fortran::maxval(residual));
            return false;
        } else {
            return true;
        }
        */
        int nx = arr1.dimension[0];
        int ny = arr2.dimension[1];
        for (int i=1; i<nx+1; i++) {
            for (int j=1; j<ny+1; j++) {
                if (abs(arr1(i,j) - arr2(i,j)) > tolerance) {
                    printf("arr1 = %f, arr2 = %f\n", arr1(i,j), arr2(i,j));
                    return false;
                }
            }
        }
        return true;
    }

    template <class T> double arrmin(T &arr) {
        double minval = arr.myData[0];
        for (int i = 0; i<arr.totElems(); i++) {
            if (arr.myData[i] < minval) {
                minval = arr.myData[i];
            }
        }
        return minval;
    }

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
        std::string cloud_optics_file_lw = "./data/rrtmgp-cloud-optics-coeffs-lw.nc";
        OpticalProps2str clouds_sw = dummy_clouds_sw(cloud_optics_file_sw, scream::rrtmgp::k_dist_sw, p_lay, t_lay);
        OpticalProps1scl clouds_lw = dummy_clouds_lw(cloud_optics_file_lw, scream::rrtmgp::k_dist_lw, p_lay, t_lay);

        // Run RRTMGP code on dummy atmosphere; this might get ugly
        // Inputs should be atmosphere state, outputs should be fluxes
        // TODO: should absorption coefficients be an input, or should that be initialized
        // and kept in the scream::rrtmgp namespace?
        scream::rrtmgp::rrtmgp_main(
                p_lay, t_lay, p_lev, t_lev, gas_concs, col_dry, 
                sfc_alb_dir, sfc_alb_dif, mu0,
                clouds_sw, clouds_lw,
                fluxes_sw, fluxes_lw);
 
        // Check fluxes against reference; note that input file contains reference fluxes
        real2d sw_flux_up_ref ("sw_flux_up_ref " ,ncol,nlay+1);
        real2d sw_flux_dn_ref  ("sw_flux_dn_ref " ,ncol,nlay+1);
        real2d sw_flux_dn_dir_ref ("sw_flux_dn_dir_ref ",ncol,nlay+1);
        real2d lw_flux_up_ref ("lw_flux_up_ref " ,ncol,nlay+1);
        real2d lw_flux_dn_ref  ("lw_flux_dn_ref " ,ncol,nlay+1);
        read_fluxes(inputfile, sw_flux_up_ref, sw_flux_dn_ref, sw_flux_dn_dir_ref, lw_flux_up_ref, lw_flux_dn_ref );

        // Check values
        REQUIRE(all_equals(sw_flux_up_ref    , fluxes_sw.flux_up    ));
        REQUIRE(all_equals(sw_flux_dn_ref    , fluxes_sw.flux_dn    ));
        REQUIRE(all_equals(sw_flux_dn_dir_ref, fluxes_sw.flux_dn_dir));
        REQUIRE(all_equals(lw_flux_up_ref    , fluxes_lw.flux_up    ));
        REQUIRE(all_equals(lw_flux_dn_ref    , fluxes_lw.flux_dn    ));

        // ALTERNATIVELY: create a single or two-layer atmosphere to do a dummy calc
    }

    OpticalProps2str dummy_clouds_sw(
            std::string cloud_optics_file, GasOpticsRRTMGP &kdist, 
            real2d &p_lay, real2d &t_lay) {

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

        // Needed for consistency with all-sky example problem?
        cloud_optics.set_ice_roughness(2);

        // Generate some fake liquid and ice water data. We pick values to be midway between
        // the min and max of the valid lookup table values for effective radii
        real rel_val = 0.5 * (cloud_optics.get_min_radius_liq() + cloud_optics.get_max_radius_liq());
        real rei_val = 0.5 * (cloud_optics.get_min_radius_ice() + cloud_optics.get_max_radius_ice());

        // Restrict clouds to troposphere (> 100 hPa = 100*100 Pa) and not very close to the ground (< 900 hPa), and
        // put them in 2/3 of the columns since that's roughly the total cloudiness of earth.
        // Set sane values for liquid and ice water path.
        real2d rel("rel", ncol, nlay);
        real2d rei("rei", ncol, nlay);
        real2d lwp("lwp", ncol, nlay);
        real2d iwp("iwp", ncol, nlay);
        real2d cloud_mask("cloud_mask", ncol, nlay);
        parallel_for( Bounds<2>(nlay,ncol) , YAKL_LAMBDA (int ilay, int icol) {
            cloud_mask(icol,ilay) = p_lay(icol,ilay) > 100._wp * 100._wp && p_lay(icol,ilay) < 900._wp * 100._wp && mod(icol, 3) != 0;
            // Ice and liquid will overlap in a few layers
            lwp(icol,ilay) = merge(10._wp,  0._wp, cloud_mask(icol,ilay) && t_lay(icol,ilay) > 263._wp);
            iwp(icol,ilay) = merge(10._wp,  0._wp, cloud_mask(icol,ilay) && t_lay(icol,ilay) < 273._wp);
            rel(icol,ilay) = merge(rel_val, 0._wp, lwp(icol,ilay) > 0._wp);
            rei(icol,ilay) = merge(rei_val, 0._wp, iwp(icol,ilay) > 0._wp);
        });

        // Calculate cloud optics
        cloud_optics.cloud_optics(lwp, iwp, rel, rei, clouds);

        // Return optics
        return clouds;
    }

    OpticalProps1scl dummy_clouds_lw(
            std::string cloud_optics_file, GasOpticsRRTMGP &kdist, 
            real2d &p_lay, real2d &t_lay) {

        // Problem sizes
        int ncol = t_lay.dimension[0];
        int nlay = t_lay.dimension[1];

        REQUIRE(ncol == 128);

        // Initialize optics
        OpticalProps1scl clouds;
        clouds.init(kdist.get_band_lims_wavenumber());
        clouds.alloc_1scl(ncol, nlay);  // this is dumb, why do we need to init and alloc separately?!

        // Initialize working optics class...this is kind of strange, but there's a separate
        // class that does RRTMGP cloud optics. The actual cloud optical properties will then
        // get copied over to the "clouds" instance of OpticalProps2str we created above. Not
        // sure why the CloudOptics class didn't just inherit from OpticalProps2str so we could
        // have used it directly in the calls to the driver routines.
        CloudOptics cloud_optics;
        load_cld_lutcoeff(cloud_optics, cloud_optics_file);

        // Needed for consistency with all-sky example problem?
        cloud_optics.set_ice_roughness(2);

        // Generate some fake liquid and ice water data. We pick values to be midway between
        // the min and max of the valid lookup table values for effective radii
        real rel_val = 0.5 * (cloud_optics.get_min_radius_liq() + cloud_optics.get_max_radius_liq());
        real rei_val = 0.5 * (cloud_optics.get_min_radius_ice() + cloud_optics.get_max_radius_ice());

        // Restrict clouds to troposphere (> 100 hPa = 100*100 Pa) and not very close to the ground (< 900 hPa), and
        // put them in 2/3 of the columns since that's roughly the total cloudiness of earth.
        // Set sane values for liquid and ice water path.
        real2d rel("rel", ncol, nlay);
        real2d rei("rei", ncol, nlay);
        real2d lwp("lwp", ncol, nlay);
        real2d iwp("iwp", ncol, nlay);
        real2d cloud_mask("cloud_mask", ncol, nlay);
        parallel_for( Bounds<2>(nlay,ncol) , YAKL_LAMBDA (int ilay, int icol) {
            cloud_mask(icol,ilay) = p_lay(icol,ilay) > 100._wp * 100._wp && p_lay(icol,ilay) < 900._wp * 100._wp && mod(icol, 3) != 0;
            // Ice and liquid will overlap in a few layers
            lwp(icol,ilay) = merge(10._wp,  0._wp, cloud_mask(icol,ilay) && t_lay(icol,ilay) > 263._wp);
            iwp(icol,ilay) = merge(10._wp,  0._wp, cloud_mask(icol,ilay) && t_lay(icol,ilay) < 273._wp);
            rel(icol,ilay) = merge(rel_val, 0._wp, lwp(icol,ilay) > 0._wp);
            rei(icol,ilay) = merge(rei_val, 0._wp, iwp(icol,ilay) > 0._wp);
        });

        // Calculate cloud optics
        cloud_optics.cloud_optics(lwp, iwp, rel, rei, clouds);

        // Return optics
        return clouds;
    }

    // Function to read fluxes from input file so we can compare our answers against the reference
    void read_fluxes(
            std::string inputfile, 
            real2d &sw_flux_up, real2d &sw_flux_dn, real2d &sw_flux_dn_dir,
            real2d &lw_flux_up, real2d &lw_flux_dn) {

        // Initialize netcdf reader
        yakl::SimpleNetCDF io;
        io.open(inputfile, yakl::NETCDF_MODE_READ);

        // Initialize arrays to hold fluxes
        int nlev = io.getDimSize("lev");
        int ncol = io.getDimSize("col_flx");
        sw_flux_up = real2d("sw_flux_up", ncol, nlev);
        sw_flux_dn = real2d("sw_flux_dn", ncol, nlev);
        sw_flux_dn_dir = real2d("sw_flux_dn_dir", ncol, nlev);
        lw_flux_up = real2d("lw_flux_up", ncol, nlev);
        lw_flux_dn = real2d("lw_flux_dn", ncol, nlev);

        // Read data
        io.read(sw_flux_up, "sw_flux_up_result");
        io.read(sw_flux_dn, "sw_flux_dn_result");
        io.read(sw_flux_dn_dir, "sw_flux_dir_result");
        io.read(lw_flux_up, "lw_flux_up_result");
        io.read(lw_flux_dn, "lw_flux_dn_result");
    }
   
} // empty namespace
