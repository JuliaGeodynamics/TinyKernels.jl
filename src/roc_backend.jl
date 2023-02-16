module ROCBackend

export ROCDevice

import TinyKernels: Kernel

using AMDGPU

struct ROCDevice end

struct ROCEvent{T}
    signal::T
end

import Base: wait

wait(ev::ROCEvent) = wait(ev.signal)
wait(evs::AbstractArray{ROCEvent}) = wait.(evs)

mutable struct QueuePool
    next_queue_idx::Int
    queues::Vector{ROCQueue}
end

const MAX_QUEUES = 6
const QUEUES = Dict{Symbol,QueuePool}()

function get_queue(priority::Symbol)
    pool = get!(QUEUES, priority) do
        max_queues = MAX_QUEUES
        QueuePool(1, [ROCQueue(AMDGPU.default_device(); priority=priority) for _ in 1:max_queues])
    end
    return pick_queue(pool)
end

function pick_queue(pool::QueuePool)
    # round-robin queue selection
    pool.next_queue_idx += 1
    pool.next_queue_idx = ((pool.next_queue_idx - 1) % length(pool.queues)) + 1
    return pool.queues[pool.next_queue_idx]
end

function (k::Kernel{<:ROCDevice})(args...)
    events = ROCEvent[]
    for (range, priority) in zip(k.ranges, k.priorities)
        # compile ROC kernel
        roc_kernel = @roc launch = false k.fun(range, args...)
        # determine optimal launch parameters
        config = AMDGPU.launch_configuration(roc_kernel.fun)
        nthreads = (32, cld(config.groupsize, 32))
        ngrid = length.(range)
        # launch kernel
        queue = get_queue(priority)
        signal = @roc groupsize=nthreads gridsize=ngrid queue=queue k.fun(range, args...)
        # record event
        push!(events, ROCEvent(signal))
    end
    return events
end

end