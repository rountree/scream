module scream_scorpio_interface_iso_c2f
  use iso_c_binding
  implicit none
     
#include "scream_config.f"
#ifdef SCREAM_DOUBLE_PRECISION
# define c_real c_double
#else
# define c_real c_float
#endif
!
! This file contains bridges from scream c++ to shoc fortran. 
!
contains
!=====================================================================!
  subroutine eam_init_pio_subsystem_c2f(mpicom,compid,local) bind(c)
    use scream_scorpio_interface, only : eam_init_pio_subsystem
    integer(kind=c_int), value, intent(in) :: mpicom
    integer(kind=c_int), value, intent(in) :: compid
    logical(kind=c_bool),value, intent(in) :: local

    call eam_init_pio_subsystem(mpicom,compid,LOGICAL(local))
  end subroutine eam_init_pio_subsystem_c2f
!=====================================================================!
  subroutine eam_pio_finalize_c2f() bind(c)
    use scream_scorpio_interface, only : eam_pio_finalize

    call eam_pio_finalize()
  end subroutine eam_pio_finalize_c2f
!=====================================================================!
  subroutine register_outfile_c2f(filename_in) bind(c)
    use scream_scorpio_interface, only : register_outfile
    type(c_ptr), intent(in) :: filename_in

    character(len=256)       :: filename

    call convert_c_string(filename_in,filename)
    call register_outfile(trim(filename))

  end subroutine register_outfile_c2f
!=====================================================================!
  subroutine register_infile_c2f(filename_in) bind(c)
    use scream_scorpio_interface, only : register_infile
    type(c_ptr), intent(in) :: filename_in

    character(len=256)       :: filename

    call convert_c_string(filename_in,filename)
    call register_infile(trim(filename))

  end subroutine register_infile_c2f
!=====================================================================!
  subroutine sync_outfile_c2f(filename_in) bind(c)
    use scream_scorpio_interface, only : eam_sync_piofile
    type(c_ptr), intent(in) :: filename_in

    character(len=256)       :: filename

    call convert_c_string(filename_in,filename)
    call eam_sync_piofile(trim(filename))

  end subroutine sync_outfile_c2f
!=====================================================================!
  subroutine set_decomp_c2f(filename_in) bind(c)
    use scream_scorpio_interface, only : set_decomp
    type(c_ptr), intent(in) :: filename_in

    character(len=256)       :: filename

    call convert_c_string(filename_in,filename)
    call set_decomp(trim(filename))
  end subroutine set_decomp_c2f
!=====================================================================!
  subroutine set_dof_c2f(filename_in,varname_in,dof_len,dof_vec) bind(c)
    use scream_scorpio_interface, only : set_dof
    type(c_ptr), intent(in)                             :: filename_in
    type(c_ptr), intent(in)                             :: varname_in
    integer(kind=c_int), value, intent(in)              :: dof_len
    integer(kind=c_int), intent(in), dimension(dof_len) :: dof_vec

    character(len=256)          :: filename
    character(len=256)          :: varname
    integer, dimension(dof_len) :: dof_vec_f90
    integer                     :: ii

    call convert_c_string(filename_in,filename)
    call convert_c_string(varname_in,varname)
    ! Need to add 1 to the dof_vec because C++ starts indices at 0 not 1:
    do ii = 1,dof_len
      dof_vec_f90(ii) = dof_vec(ii) + 1
    end do
    call set_dof(trim(filename),trim(varname),dof_len,dof_vec_f90)
  end subroutine set_dof_c2f
!=====================================================================!
  subroutine eam_pio_closefile_c2f(filename_in) bind(c)
    use scream_scorpio_interface, only : eam_pio_closefile
    type(c_ptr), intent(in) :: filename_in
    character(len=256)      :: filename

    call convert_c_string(filename_in,filename)
    call eam_pio_closefile(trim(filename))

  end subroutine eam_pio_closefile_c2f
!=====================================================================!
  subroutine pio_update_time_c2f(filename_in,time) bind(c)
    use scream_scorpio_interface, only : eam_update_time
    type(c_ptr), intent(in) :: filename_in
    real(kind=c_real), value, intent(in) :: time

    character(len=256)       :: filename

    call convert_c_string(filename_in,filename)
    call eam_update_time(trim(filename),time)

  end subroutine pio_update_time_c2f
!=====================================================================!
  subroutine get_variable_c2f(filename_in, shortname_in, longname_in, numdims, var_dimensions_in, dtype, pio_decomp_tag_in) bind(c)
    use scream_scorpio_interface, only : get_variable
    type(c_ptr), intent(in)                :: filename_in
    type(c_ptr), intent(in)                :: shortname_in
    type(c_ptr), intent(in)                :: longname_in
    integer(kind=c_int), value, intent(in) :: numdims
    type(c_ptr), intent(in)                :: var_dimensions_in(numdims)
    integer(kind=c_int), value, intent(in) :: dtype
    type(c_ptr), intent(in)                :: pio_decomp_tag_in
    
    character(len=256) :: filename
    character(len=256) :: shortname
    character(len=256) :: longname
    character(len=256) :: var_dimensions(numdims)
    character(len=256) :: pio_decomp_tag
    integer            :: ii
    
    call convert_c_string(filename_in,filename)
    call convert_c_string(shortname_in,shortname)
    call convert_c_string(longname_in,longname)
    call convert_c_string(pio_decomp_tag_in,pio_decomp_tag)
    do ii = 1,numdims
      call convert_c_string(var_dimensions_in(ii), var_dimensions(ii))
    end do
   
    call get_variable(filename,shortname,longname,numdims,var_dimensions,dtype,pio_decomp_tag)

  end subroutine get_variable_c2f
!=====================================================================!
  subroutine register_variable_c2f(filename_in, shortname_in, longname_in, numdims, var_dimensions_in, dtype, pio_decomp_tag_in) bind(c)
    use scream_scorpio_interface, only : register_variable
    type(c_ptr), intent(in)                :: filename_in
    type(c_ptr), intent(in)                :: shortname_in
    type(c_ptr), intent(in)                :: longname_in
    integer(kind=c_int), value, intent(in) :: numdims
    type(c_ptr), intent(in)                :: var_dimensions_in(numdims)
    integer(kind=c_int), value, intent(in) :: dtype
    type(c_ptr), intent(in)                :: pio_decomp_tag_in
    
    character(len=256) :: filename
    character(len=256) :: shortname
    character(len=256) :: longname
    character(len=256) :: var_dimensions(numdims)
    character(len=256) :: pio_decomp_tag
    integer            :: ii
    
    call convert_c_string(filename_in,filename)
    call convert_c_string(shortname_in,shortname)
    call convert_c_string(longname_in,longname)
    call convert_c_string(pio_decomp_tag_in,pio_decomp_tag)
    do ii = 1,numdims
      call convert_c_string(var_dimensions_in(ii), var_dimensions(ii))
    end do
   
    call register_variable(filename,shortname,longname,numdims,var_dimensions,dtype,pio_decomp_tag)

  end subroutine register_variable_c2f
!=====================================================================!
  subroutine register_dimension_c2f(filename_in, shortname_in, longname_in, length) bind(c)
    use scream_scorpio_interface, only : register_dimension
    type(c_ptr), intent(in)                :: filename_in
    type(c_ptr), intent(in)                :: shortname_in
    type(c_ptr), intent(in)                :: longname_in
    integer(kind=c_int), value, intent(in) :: length

    character(len=256) :: filename
    character(len=256) :: shortname 
    character(len=256) :: longname  

    call convert_c_string(filename_in,filename)
    call convert_c_string(shortname_in,shortname)
    call convert_c_string(longname_in,longname)
    call register_dimension(filename,shortname,longname,length)
    
  end subroutine register_dimension_c2f
!=====================================================================!
  subroutine eam_pio_enddef_c2f(filename_in) bind(c)
    use scream_scorpio_interface, only : eam_pio_enddef
    type(c_ptr), intent(in) :: filename_in

    character(len=256)      :: filename

    call convert_c_string(filename_in,filename)
    call eam_pio_enddef(filename)
  end subroutine eam_pio_enddef_c2f
!=====================================================================!
  subroutine count_pio_atm_file_c2f() bind(c)
    use scream_scorpio_interface, only : count_pio_atm_file

    call count_pio_atm_file()
  end subroutine count_pio_atm_file_c2f
!=====================================================================!
  subroutine convert_c_string(c_string_ptr,f_string)
  ! Purpose: To convert a c_string pointer to the proper fortran string format.
    type(c_ptr), intent(in) :: c_string_ptr
    character(len=256), intent(out) :: f_string
    character(len=256), pointer :: temp_string
    integer :: str_len

    call c_f_pointer(c_string_ptr,temp_string)
    str_len = index(temp_string, C_NULL_CHAR) - 1
    f_string = trim(temp_string(1:str_len))
    
    return
  end subroutine convert_c_string
!=====================================================================!
  subroutine grid_write_data_array_c2f_real_1d(filename_in,varname_in,dim1_length,hbuf_in) bind(c)
    use scream_scorpio_interface, only: grid_write_data_array

    type(c_ptr), intent(in)                :: filename_in
    type(c_ptr), intent(in)                :: varname_in
    integer(kind=c_int), value, intent(in) :: dim1_length
    real(kind=c_real), intent(in), dimension(dim1_length) :: hbuf_in

    character(len=256) :: filename
    character(len=256) :: varname

    call convert_c_string(filename_in,filename)
    call convert_c_string(varname_in,varname)
    call grid_write_data_array(filename,hbuf_in,varname)

  end subroutine grid_write_data_array_c2f_real_1d
!=====================================================================!
  subroutine grid_read_data_array_c2f_real(filename_in,varname_in,dim1_length,hbuf_out) bind(c)
    use scream_scorpio_interface, only: grid_read_data_array

    type(c_ptr), intent(in)                :: filename_in
    type(c_ptr), intent(in)                :: varname_in
    integer(kind=c_int), value, intent(in) :: dim1_length
    real(kind=c_real), intent(out), dimension(dim1_length) :: hbuf_out

    character(len=256) :: filename
    character(len=256) :: varname

    call convert_c_string(filename_in,filename)
    call convert_c_string(varname_in,varname)
    call grid_read_data_array(filename,hbuf_out,varname)

  end subroutine grid_read_data_array_c2f_real
!=====================================================================!
end module scream_scorpio_interface_iso_c2f
