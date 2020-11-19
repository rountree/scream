#ifndef SHOC_SFC_FLUXES_IMPL_HPP
#define SHOC_SFC_FLUXES_IMPL_HPP

#include "shoc_functions.hpp" // for ETI only but harmless for GPU

namespace scream {
namespace shoc {

/*
 * Implementation of shoc sfc_fluxes. Clients should NOT
 * #include this file, but include shoc_functions.hpp instead.
 */

template<typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>::sfc_fluxes(
  const MemberType&            team,
  const Int&                   num_tracer,
  const Scalar&                dtime,
  const Scalar&                rho_zi_sfc,
  const Scalar&                rdp_zt_sfc,
  const Scalar&                wthl_sfc,
  const Scalar&                wqw_sfc,
  const Scalar&                wtke_sfc,
  const uview_1d<const Spack>& wtracer_sfc,
  Scalar&                      thetal,
  Scalar&                      qw,
  Scalar&                      tke,
  const uview_1d<Spack>&       wtracer)
{
  const auto ggr = C::gravit;

  const auto cmnfac = dtime*(ggr*rho_zi_sfc*rdp_zt_sfc);
  thetal += cmnfac*wthl_sfc;
  qw += cmnfac*wqw_sfc;
  tke += cmnfac*wtke_sfc;

  const Int num_tracer_pack = ekat::npack<Spack>(num_tracer);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, num_tracer_pack), [&] (const Int& p) {
    wtracer(p) += cmnfac*wtracer_sfc(p);
  });
}

} // namespace shoc
} // namespace scream

#endif
