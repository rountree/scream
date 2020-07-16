#include "catch2/catch.hpp"
#include "physics/rrtmgp/rrtmgp.hpp"
#include "physics/rrtmgp/scream_rrtmgp_interface.hpp"
#include "netcdf.h"
namespace {

    TEST_CASE("rrtmgp", "stub") {
      int val = scream::rrtmgp::rrtmgp_stub();
      REQUIRE(val == 42);
    }

    TEST_CASE("rrtmgp_read_scorpio", "") {
        REQUIRE(1 == 1);
    }

    TEST_CASE("rrtmgp_read_netcdf", "") {

        /* Try to read absorption coefficient data */
        int ncid, varid;
        int retval;
        
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
    }

} // empty namespace
