# using Makie
# using Observables
using IterTools

WORKERS_BY_FLOOR = [100, 120, 60, 120, 80, 20]
ELEVATOR_CAPACITY = 10
N_ELEVATORS = 4

ELEVATOR_GROUNDFLOOR_TIME = 15
ELEVATOR_TRAVEL_TIME = 5
ELEVATOR_UNLOAD_TIME = 10

tripDuration(floors::Set) =
    ELEVATOR_GROUNDFLOOR_TIME +
    ELEVATOR_TRAVEL_TIME*2*maximum(floors) +
    ELEVATOR_UNLOAD_TIME*foldl((a, c) -> a+1, distinct(floors); init=0)


function simulateClusters(queues_by_lift::Array{Vector{Set{Int}}, N_ELEVATORS})
    maximum(map(q -> sum(s -> tripDuration(), q), queues_by_lift))
end

clusters = [repeat([Set([f])], ceil(Int64, n/ELEVATOR_CAPACITY)) for (f, n) in enumerate(WORKERS_BY_FLOOR)]; clusters = reduce(vcat, clusters)

# greedily coin-problem it
trip_durations = sort(map(tripDuration, clusters), rev=true)
tot_durations = zeros(N_ELEVATORS)
for duration in trip_durations
    tot_durations[argmin(tot_durations)] += duration
end
@show tot_durations
@show maximum(tot_durations), maximum(tot_durations)/60






for i in 1:6
    println("floor ", i, " -> ", tripDuration(Set([i])))
end
