#ifndef SCREAM_RRTMGP_INTERFACE_HPP
#define SCREAM_RRTMGP_INTERFACE_HPP

#include "ekat/ekat_assert.hpp"
#include "mo_gas_optics_rrtmgp.h"

namespace scream {
    namespace rrtmgp {
        /* 
         * Objects containing k-distribution information need to be initialized
         * once and then persist throughout the life of the program, so we
         * declare them here within the rrtmgp namespace.
         */
        extern GasOpticsRRTMGP k_dist_sw;
        extern GasOpticsRRTMGP k_dist_lw;

        /*
         * Assuming we can jump directly to using a C++ API...
         */
        extern void rrtmgp_initialize();
        extern void rrtmgp_main();
        extern void rrtmgp_finalize();
    } // namespace rrtmgp
}  // namespace scream

#endif  // SCREAM_RRTMGP_INTERFACE_HPP
