using Test
using Enzyme
using TinyKernels
using TinyKernels.KernelAD
using TinyKernels.CPUBackend

device = CPUDevice()
DAT = Float64
nx, ny = 10, 11

@tiny function kernel_test!(RUx, RUy, Ux, Uy)
    ix, iy = @indices()
    if ix ∈ axes(RUx, 1) && iy ∈ axes(RUx, 2)
        @inbounds RUx[ix, iy] = Ux[ix, iy]^2 - convert(eltype(Ux), 2.0) * Ux[ix+1, iy]^2 + Ux[ix+2, iy]^2 + convert(eltype(Ux), 0.5) * Uy[ix+1, iy]^2
    end
    if ix ∈ axes(RUy, 1) && iy ∈ axes(RUy, 2)
        @inbounds RUy[ix, iy] = Uy[ix, iy]^2 - convert(eltype(Uy), 2.0) * Uy[ix, iy+1]^2 + Uy[ix, iy+2]^2 + convert(eltype(Uy), 0.5) * Ux[ix, iy+1]^2
    end
    return
end

# Init array
Ux  = device_array(DAT, device, nx, ny); copyto!(Ux, rand(nx, ny))
Uy  = device_array(DAT, device, nx, ny); copyto!(Uy, rand(nx, ny))
RUx = device_array(DAT, device, nx-2, ny); fill!(RUx, 0.0)
RUy = device_array(DAT, device, nx, ny-2); fill!(RUy, 0.0)
# AD VJP
∂Rx_∂R = device_array(DAT, device, nx-2, ny); fill!(∂Rx_∂R, 1.0)
∂Ry_∂R = device_array(DAT, device, nx, ny-2); fill!(∂Ry_∂R, 1.0)
∂Ux_∂R = device_array(DAT, device, nx, ny); fill!(∂Ux_∂R, 0.0)
∂Uy_∂R = device_array(DAT, device, nx, ny); fill!(∂Uy_∂R, 0.0)
# Exact VJP
∂Ux_∂R_exact = device_array(DAT, device, nx, ny); fill!(∂Ux_∂R_exact, 0.0)
∂Ux_∂R_exact[3:end-2  ,2:end-1] .=              Ux[3:end-2  ,2:end-1]
∂Ux_∂R_exact[[1,end  ],2:end-1] .= DAT(3.0)  .* Ux[[1,end  ],2:end-1]
∂Ux_∂R_exact[[2,end-1],2:end-1] .=           .- Ux[[2,end-1],2:end-1]
∂Ux_∂R_exact[[1,end  ],[1,end]] .= DAT(2.0)  .* Ux[[1,end]  ,[1,end]]
∂Ux_∂R_exact[[2,end-1],[1,end]] .= DAT(-2.0) .* Ux[[2,end-1],[1,end]]

∂Uy_∂R_exact = device_array(DAT, device, nx, ny); fill!(∂Uy_∂R_exact, 0.0)
∂Uy_∂R_exact[2:end-1,3:end-2  ] .=              Uy[2:end-1,3:end-2  ]
∂Uy_∂R_exact[2:end-1,[1,end  ]] .= DAT(3.0)  .* Uy[2:end-1,[1,end  ]]
∂Uy_∂R_exact[2:end-1,[2,end-1]] .=           .- Uy[2:end-1,[2,end-1]]
∂Uy_∂R_exact[[1,end],[1,end  ]] .= DAT(2.0)  .* Uy[[1,end],[1,end  ]]
∂Uy_∂R_exact[[1,end],[2,end-1]] .= DAT(-2.0) .* Uy[[1,end],[2,end-1]]

@testset "auto-diff 2D" begin
    # Generate kernel
    test! = kernel_test!(device)
    # Generate kernel gradient
    grad_test_kernel! = Enzyme.autodiff(test!)
    # Evaluate forward problem
    TinyKernels.device_synchronize(device)
    wait(test!(RUx, RUy, Ux, Uy; ndrange=size(Ux)))
    # Compute VJP
    wait(grad_test_kernel!(DuplicatedNoNeed(RUx, ∂Rx_∂R),
                           DuplicatedNoNeed(RUy, ∂Ry_∂R),
                           DuplicatedNoNeed(Ux , ∂Ux_∂R),
                           DuplicatedNoNeed(Uy , ∂Uy_∂R); ndrange=size(Ux)))
    @test ∂Ux_∂R ≈ ∂Ux_∂R_exact
    @test ∂Uy_∂R ≈ ∂Uy_∂R_exact
end