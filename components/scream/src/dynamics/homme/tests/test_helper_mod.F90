module test_helper_mod

  implicit none

  public :: set_test_params_f90
  public :: cleanup_test_f90

contains

  subroutine set_test_params_f90 (ne_in) bind(c)
    use iso_c_binding,     only: c_int
    use dimensions_mod,    only: ne
    use control_mod,       only: topology, cubed_sphere_map, partmethod
    use params_mod,        only: SFCURVE
    use homme_context_mod, only: is_params_inited
    !
    ! Inputs
    !
    integer (kind=c_int), intent(in) :: ne_in

    ! Hard coded choices for cubed sphere
    topology = 'cube'
    cubed_sphere_map = 0
    partmethod = SFCURVE

    ! Set desired resolution
    ne = ne_in

    is_params_inited = .true.
  end subroutine set_test_params_f90

  subroutine cleanup_test_f90 () bind(c)
    use schedtype_mod,     only: schedule
    use parallel_mod,      only: rrequest, srequest, global_shared_buf, status
    use homme_context_mod, only: is_parallel_inited

    ! Cleanup the schedule structure
    deallocate(Schedule(1)%SendCycle)
    deallocate(Schedule(1)%RecvCycle)
    deallocate(Schedule(1)%MoveCycle)
    deallocate(Schedule(1)%pIndx)
    deallocate(Schedule(1)%gIndx)
    deallocate(Schedule(1)%Local2Global)
    deallocate(Schedule)

    deallocate(rrequest)
    deallocate(srequest)
    deallocate(status)
    deallocate(global_shared_buf)

    is_parallel_inited = .false.
  end subroutine cleanup_test_f90

end module test_helper_mod
