#ifndef P3_FUNCTIONS_ICE_COLLECTION_IMPL_HPP
#define P3_FUNCTIONS_ICE_COLLECTION_IMPL_HPP

#include "p3_functions.hpp" // for ETI only but harmless for GPU

namespace scream {
namespace p3 {

template<typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::ice_cldliq_collection(
  const Spack& rho, const Spack& temp,
  const Spack& rhofaci, const Spack& f1pr04,
  const Spack& qitot_incld, const Spack& qc_incld,
  const Spack& nitot_incld, const Spack& nc_incld,
  Spack& qccol, Spack& nccol, Spack& qcshd, Spack& ncshdc,
  const Smask& context)
{
  constexpr Scalar qsmall = C::QSMALL;
  constexpr Scalar tmelt  = C::Tmelt;

  // set up masks
  const auto t_is_negative        = temp < tmelt;
  const auto qitot_incld_gt_small = qitot_incld > qsmall;
  const auto qc_incld_gt_small    = qc_incld > qsmall;
  const auto both_gt_small        = qitot_incld_gt_small && qc_incld_gt_small && context;
  const auto both_gt_small_pos_t  = both_gt_small && !t_is_negative;

  constexpr auto eci = C::eci;
  constexpr auto inv_dropmass = C::ONE/C::dropmass;

  qccol.set(both_gt_small && t_is_negative,
            rhofaci*f1pr04*qc_incld*eci*rho*nitot_incld);
  nccol.set(both_gt_small, rhofaci*f1pr04*nc_incld*eci*rho*nitot_incld);

  // for T > 273.15, assume cloud water is collected and shed as rain drops
  // sink for cloud water mass and number, note qcshed is source for rain mass
  qcshd.set(both_gt_small_pos_t, rhofaci*f1pr04*qc_incld*eci*rho*nitot_incld);
  nccol.set(both_gt_small_pos_t, rhofaci*f1pr04*nc_incld*eci*rho*nitot_incld);
  // source for rain number, assume 1 mm drops are shed
  ncshdc.set(both_gt_small_pos_t, qcshd*inv_dropmass);
}

template<typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::ice_rain_collection(
  const Spack& rho, const Spack& temp,
  const Spack& rhofaci, const Spack& logn0r,
  const Spack& f1pr07, const Spack& f1pr08,
  const Spack& qitot_incld, const Spack& nitot_incld,
  const Spack& qr_incld,
  Spack& qrcol, Spack& nrcol,
  const Smask& context)
{
  constexpr Scalar qsmall = C::QSMALL;
  constexpr Scalar tmelt  = C::Tmelt;

  // Set up masks
  const auto t_is_negative        = temp <= tmelt;
  const auto qitot_incld_ge_small = qitot_incld >= qsmall;
  const auto qr_incld_ge_small    = qr_incld >= qsmall;
  const auto both_gt_small        = qitot_incld_ge_small && qr_incld_ge_small && context;
  const auto both_gt_small_neg_t  = both_gt_small && t_is_negative;

  constexpr Scalar ten = 10.0;
  constexpr auto eri = C::eri;

  // note: f1pr08 and logn0r are already calculated as log_10
  qrcol.set(both_gt_small_neg_t, pack::pow(ten, f1pr08+logn0r)*rho*rhofaci*eri*nitot_incld);
  nrcol.set(both_gt_small_neg_t, pack::pow(ten, f1pr07+logn0r)*rho*rhofaci*eri*nitot_incld);

  // rain number sink due to collection
  // for T > 273.15, assume collected rain number is shed as
  // 1 mm drops
  // note that melting of ice number is scaled to the loss
  // rate of ice mass due to melting
  // collection of rain above freezing does not impact total rain mass
  nrcol.set(both_gt_small && !t_is_negative,
            pack::pow(ten, f1pr07 + logn0r)*rho*rhofaci*eri*nitot_incld);
  // for now neglect shedding of ice collecting rain above freezing, since snow is
  // not expected to shed in these conditions (though more hevaily rimed ice would be
  // expected to lead to shedding)
}

template<typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::ice_self_collection(
  const Spack& rho, const Spack& rhofaci,
  const Spack& f1pr03, const Spack& eii,
  const Spack& qirim_incld, const Spack& qitot_incld,
  const Spack& nitot_incld, Spack& nislf,
  const Smask& context)
{
  constexpr Scalar qsmall = C::QSMALL;
  constexpr Scalar zero = C::ZERO;

  // Set up masks
  const auto qirim_incld_positive = qirim_incld > zero && context;
  const auto qitot_incld_gt_small = qitot_incld > qsmall && context;

  Spack tmp1{0.0};
  Spack Eii_fact{0.0};
  Smask tmp1_lt_six{0};
  Smask tmp1_ge_six{0};
  Smask tmp1_lt_nine{0};
  Smask tmp1_ge_nine{0};

  if (qitot_incld_gt_small.any()) {
    // Determine additional collection efficiency factor to be applied to ice-ice collection.
    // The computed values of qicol and nicol are multipiled by Eii_fact to gradually shut off collection
    // if ice is highly rimed.
    tmp1.set(qitot_incld_gt_small && qirim_incld_positive,
             qirim_incld/qitot_incld);   //rime mass fraction
    tmp1_lt_six  = tmp1 < sp(0.6);
    tmp1_ge_six  = tmp1 >= sp(0.6);
    tmp1_lt_nine = tmp1 < sp(0.9);
    tmp1_ge_nine = tmp1 >= sp(0.9);

    Eii_fact.set(tmp1_lt_six && qirim_incld_positive, 1);

    Eii_fact.set(tmp1_ge_six && tmp1_lt_nine && qirim_incld_positive,
                 1 - (tmp1-sp(0.6))/sp(0.3));

    Eii_fact.set(tmp1_ge_nine && qirim_incld_positive, 0);

    Eii_fact.set(!qirim_incld_positive && context, 1);

    nislf.set(qitot_incld_gt_small,
              f1pr03*rho*eii*Eii_fact*rhofaci*nitot_incld);
  }
}

} // namespace p3
} // namespace scream

#endif