module adjoint_capture
    implicit none
    real(8), allocatable :: material(:,:,:,:,:)
contains
    subroutine begin_material(nx,ny,nz)
        integer, intent(in) :: nx,ny,nz
        if (allocated(material)) deallocate(material)
        allocate(material(0:nx,0:ny,0:nz,10,4))
        material=0d0
    end subroutine

    subroutine record_material(i,j,k,l,sh,sv,cy,sy,cz,sz,uses_grad,gi,gj,gk)
        integer, intent(in) :: i,j,k,l,gi,gj,gk
        real(8), intent(in) :: sh,sv,cy,sy,cz,sz
        logical, intent(in) :: uses_grad
        material(i,j,k,:,l)=[sh,sv,cy,sy,cz,sz,merge(1d0,0d0,uses_grad), &
                             real(gi,8),real(gj,8),real(gk,8)]
    end subroutine

    subroutine save_material(rx)
        use buildgrid, only: f_RXs,f_RYs,f_RZs,f_nsub1,f_nsub2,f_nsub3, &
                              f_GridX,f_GridY,f_GridZ
        integer, intent(in) :: rx
        integer :: unit,i
        character(160) :: filename
        call execute_command_line('mkdir -p outputs/adjoint')
        write(filename,'(a,i0,a)') 'outputs/adjoint/material_rx',rx,'.bin'
        open(newunit=unit,file=trim(filename),access='stream',form='unformatted',status='replace')
        write(unit) int(shape(material),8)
        write(unit) material
        close(unit)
        write(filename,'(a,i0,a)') 'outputs/adjoint/grid_rx',rx,'.txt'
        open(newunit=unit,file=trim(filename),status='replace')
        write(unit,*) f_nsub1,f_nsub2,f_nsub3
        write(unit,*) size(f_RXs),size(f_RYs),size(f_RZs)
        write(unit,*) f_RXs
        write(unit,*) f_RYs
        write(unit,*) f_RZs
        write(unit,*) size(f_GridX),size(f_GridY),size(f_GridZ)
        write(unit,*) f_GridX
        write(unit,*) f_GridY
        write(unit,*) f_GridZ
        close(unit)
    end subroutine
end module

subroutine export_discrete_kernel(sqsig,field,ome,nx,ny,nz,nb,nf,sol)
    use jacobianprocessing, only: getHtensor,getcomponents
    use input, only: idtool,datflag
    use buildgrid, only: f_rxpoint
    implicit none
    integer, intent(in) :: nx,ny,nz,nb,nf
    real(8), intent(in) :: sqsig(0:nx,0:ny,0:nz,6,4),ome(nf)
    complex(8), intent(in) :: field(-1:nx+1,-1:ny+1,-1:nz+1,12,nb,nf),sol(nf,nb,nb)
    integer :: i,j,k,l,c,t,r,s,a,b,unit,tool,ncomp
    integer, parameter :: ca(6)=[1,2,3,1,2,1],cb(6)=[1,2,3,2,3,3]
    real(8), parameter :: pi=3.141592653589793d0
    real(8), allocatable :: kernel(:,:,:,:,:)
    complex(8) :: base(3,3),dh(3,3),dmat(6,6),factor
    character(160) :: filename
    ! datflag 3: the 8 in-plane geosignals; datflag 5 adds the 8 cross-plane (Y) ones.
    ! getcomponents linearizes either set (getdgeosig / getdgeosig3D) around H.
    if(datflag==3) then
        ncomp=8
    else if(datflag==5) then
        ncomp=16
    else
        error stop 'Discrete adjoint supports datflag=3 or 5'
    end if
    allocate(kernel(0:nx,0:ny,0:nz,24,ncomp))
    do t=1,nf
        call getHtensor(sol(t,:,:),base)
        ! H=(pi/10) q^T (C-i*omega*T^-2)^-1 q, T=sqrt(2.5/pi)*sigma^-1/2.
        factor=cmplx(0d0,ome(t)*pi*pi/25d0,8)
        kernel=0d0
        !$omp parallel do collapse(4) schedule(static) &
        !$omp default(shared) firstprivate(factor,base,t) &
        !$omp private(l,k,j,i,c,a,b,r,s,dmat,dh)
        do l=1,4
            do k=0,nz
                do j=0,ny
                    do i=0,nx
                        if (maxval(abs(sqsig(i,j,k,:,l)))==0d0) cycle
                        do c=1,6
                            a=l+4*(ca(c)-1); b=l+4*(cb(c)-1)
                            do s=1,nb
                                do r=1,nb
                                    dmat(r,s)=factor*field(i,j,k,a,r,t)*field(i,j,k,b,s,t)
                                    if(a/=b) dmat(r,s)=dmat(r,s)+ &
                                        factor*field(i,j,k,b,r,t)*field(i,j,k,a,s,t)
                                end do
                            end do
                            call getHtensor(dmat,dh)
                            call getcomponents(ncomp,base,dh,kernel(i,j,k,c+6*(l-1),:))
                        end do
                    end do
                end do
            end do
        end do
        !$omp end parallel do
        tool=idtool(f_rxpoint,t)
        write(filename,'(a,i0,a)') 'outputs/adjoint/kernel_tool',tool,'.bin'
        open(newunit=unit,file=trim(filename),access='stream',form='unformatted',status='replace')
        write(unit) int(shape(kernel),8)
        write(unit) int(f_rxpoint,8)
        write(unit) kernel
        close(unit)
    end do
    deallocate(kernel)
end subroutine
