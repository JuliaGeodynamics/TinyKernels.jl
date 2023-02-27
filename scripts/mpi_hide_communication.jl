using TinyKernels

using Plots

using MPI

using CUDA
@static if CUDA.functional()
    CUDA.allowscalar(false)
    using TinyKernels.CUDABackend
end

using AMDGPU
@static if AMDGPU.functional()
    AMDGPU.allowscalar(false)
    using TinyKernels.ROCBackend
end

include("mpi_utils.jl")

@tiny function kernel_init!(H)
    idx = @indices
    if checkbounds(Bool,H,idx...)
        r = 2.0.*(idx.-0.5)./size(H) .- 1.0
        @inbounds H[idx...] = exp(sum(-(x*10.0)^2 for x in r))
    end
    return
end

@tiny function kernel_flux!(qx,qy,qz,H,χ,dx,dy,dz)
    ix,iy,iz = @indices
    @inline isin(A) = checkbounds(Bool,A,ix,iy,iz)
    if isin(qx) @inbounds qx[ix,iy,iz] = -χ*(H[ix+1,iy,iz]-H[ix,iy,iz])/dx end
    if isin(qy) @inbounds qy[ix,iy,iz] = -χ*(H[ix,iy+1,iz]-H[ix,iy,iz])/dy end
    if isin(qz) @inbounds qz[ix,iy,iz] = -χ*(H[ix,iy,iz+1]-H[ix,iy,iz])/dz end
    return
end

@tiny function kernel_mass!(H,qx,qy,qz,dt,dx,dy,dz)
    ix,iy,iz = @indices
    if checkbounds(Bool,H,ix,iy,iz)
        @inbounds H[ix,iy,iz] -= dt*((qx[ix+1,iy,iz]-qx[ix,iy,iz])/dx+
                                     (qy[ix,iy+1,iz]-qy[ix,iy,iz])/dy+
                                     (qz[ix,iy,iz+1]-qz[ix,iy,iz])/dz)
    end
    return
end

@views inner_x(A) = A[2:end-1,:,:]
@views inner_y(A) = A[:,2:end-1,:]
@views inner_z(A) = A[:,:,2:end-1]

@views function main(;device,sz,dims)
    if !MPI.Initialized()
        MPI.Init()
    end

    comm_cart = MPI.Cart_create(MPI.COMM_WORLD,dims)
    me = MPI.Comm_rank(comm_cart)
    neighbors = ntuple(length(dims)) do idim
        MPI.Cart_shift(comm_cart,idim-1,1)
    end
    overlap = 2
    nx,ny,nz = sz
    dx,dy,dz = 1.0./sz
    χ        = 1.0
    dt       = min(dx,dy,dz)^2/6.1
    H  = device_array(Float64,device,nx  ,ny  ,nz  )
    qx = device_array(Float64,device,nx+1,ny  ,nz  ); fill!(qx,0.0)
    qy = device_array(Float64,device,nx  ,ny+1,nz  ); fill!(qy,0.0)
    qz = device_array(Float64,device,nx  ,ny  ,nz+1); fill!(qz,0.0)

    TinyKernels.device_synchronize(device)
    wait(Kernel(kernel_init!,device)(H;ndrange=axes(H)))

    flux! = Kernel(kernel_flux!,device)
    mass! = Kernel(kernel_mass!,device)

    boundary_width = (8,4,1)
    ranges = __split_ndrange(axes(H),boundary_width)

    for it in 1:maximum(sz)
        println(" step $it")
        ie,oe=hide_comm(ranges) do ndrange
            flux!(inner_x(qx),
                  inner_y(qy),
                  inner_z(qz),H,χ,dx,dy,dz;ndrange)
        end
        wait.(oe)
        exchange_halo!(qx,qy,qz;gridsize=sz,neighbors,overlap,comm=comm_cart)
        wait(ie)
        wait(mass!(H,qx,qy,qz,dt,dx,dy,dz;ndrange=axes(H)))
    end
    
    heatmap(Array(H[:,:,round(Int,nz/2)])';aspect_ratio=1,xlims=(1,nx),ylims=(1,ny))
    png("$me.png")

    if MPI.Initialized()
        MPI.Finalize()
    end
    return
end

@static if CUDA.functional()
    main(;device=CUDADevice(),sz=(127,127,127),dims=(2,2,2))
end