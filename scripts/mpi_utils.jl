@inline __subrange(nr,bw,I,::Val{1}) = 1:bw[I]
@inline __subrange(nr,bw,I,::Val{2}) = (size(nr,I)-bw[I]+1):size(nr,I)
@inline __subrange(nr,bw,I,::Val{3}) = (bw[I]+1):(size(nr,I)-bw[I])

@inline __split_ndrange(ndrange,ndwidth) = __split_ndrange(CartesianIndices(ndrange),ndwidth)

function __split_ndrange(ndrange::CartesianIndices{N},ndwidth::NTuple{N,<:Integer}) where N
    @assert all(size(ndrange) .> ndwidth.*2)
    @inline ndsubrange(I,::Val{J}) where J = ntuple(Val(N)) do idim
        if idim < I
            1:size(ndrange,idim)
        elseif idim == I
            __subrange(ndrange,ndwidth,idim,Val(J))
        else
            __subrange(ndrange,ndwidth,idim,Val(3))
        end
    end
    ndinner = ntuple(idim -> __subrange(ndrange,ndwidth,idim,Val(3)), Val(N))
    return ntuple(Val(2N+1)) do i
        if i == 2N+1
            ndrange[ndinner...]
        else
            idim,idir = divrem(i-1,2) .+ 1
            ndrange[ndsubrange(idim,Val(idir))...]
        end
    end
end

function hide_comm(f,ranges)
    ie = f(ranges[end])
    oe = ntuple(i->f(ranges[i]), length(ranges)-1)
    return ie, oe
end

@inline function __get_index(nx,grid_nx,overlap,side)
    if side == 1
        overlap + nx - grid_nx
    else
        grid_nx - overlap + 1
    end
end

@generated function __get_halo_view(array,side,gridsize,overlap,::Val{Dim}) where Dim
    ex = Expr(:call,:view,:array)
    for idim in 1:ndims(array)
        push!(ex.args, idim == Dim ? :coord : :(:))
    end
    quote
        coord = __get_index(size(array,Dim),gridsize[Dim],overlap,side)
        return $ex
    end
end

@generated function __get_boundary_view(array,::Val{Side},::Val{Dim}) where {Side,Dim}
    ex = Expr(:call,:view,:array)
    coord = Side == 1 ? 1 : :(size(array,$Dim))
    for idim in 1:ndims(array)
        push!(ex.args, idim == Dim ? coord  : :(:))
    end
    return ex
end

function exchange_halo!(arrays...; gridsize,neighbors,overlap,comm)
    nreq = length(arrays)*2
    recv = MPI.MultiRequest(nreq)
    send = MPI.MultiRequest(nreq)
    # serialise loop over dimensions to avoid data races in corners
    for idim in eachindex(gridsize)
        # receive halo
        ireq = 1
        for (side,rank) in Iterators.reverse(enumerate(neighbors[idim]))
            for array in arrays
                if rank != MPI.PROC_NULL
                    arv = __get_halo_view(array,side,gridsize,overlap,Val(idim))
                    MPI.Irecv!(arv,comm,recv[ireq];source=rank)
                end
                ireq += 1
            end
        end
        # send boundary
        ireq = 1
        for (side,rank) in enumerate(neighbors[idim])
            for array in arrays
                if rank != MPI.PROC_NULL
                    arv = __get_boundary_view(array,Val(side),Val(idim))
                    MPI.Isend(arv,comm,send[ireq];dest=rank)
                end
                ireq += 1
            end
        end
        MPI.Waitall(recv)
        MPI.Waitall(send)
    end
    return
end
