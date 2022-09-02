# using Makie
# using Observables
using IterTools

WORKERS_BY_FLOOR = [100, 120, 60, 120, 80, 20]
ELEVATOR_CAPACITY = 10
N_ELEVATORS = 4

ELEVATOR_GROUNDFLOOR_TIME = 15
ELEVATOR_TRAVEL_TIME = 5
ELEVATOR_UNLOAD_TIME = 10

iterlen(iterable) = foldl((a, c) -> a+1, iterable; init=0)

tripDuration(floors::Set) =
    ELEVATOR_GROUNDFLOOR_TIME +
    ELEVATOR_TRAVEL_TIME*2*maximum(floors) +        # travel time per floor *2 bc up + down
    ELEVATOR_UNLOAD_TIME*iterlen(distinct(floors))  # unload once per distinct floor of ppl in the elevator

simulateClusters(queues_by_lift::Array{Vector{Set{Int}}, N_ELEVATORS}) =
    maximum(map(q -> sum(s -> tripDuration(), q), queues_by_lift))  # total time = max of sum of trip durations for each set of clusters sent to an elevator. this assumes no breaks

clusters = [repeat([Set([f])], ceil(Int64, n/ELEVATOR_CAPACITY)) for (f, n) in enumerate(WORKERS_BY_FLOOR)]; clusters = reduce(vcat, clusters)

# greedily coin-problem it
trip_durations = sort(map(tripDuration, clusters), rev=true)
tot_durations = zeros(N_ELEVATORS)
for duration in trip_durations
    argmn = argmin(tot_durations)
    tot_durations[argmin(tot_durations)] += duration
    println("added ", duration, " to ", argmn, ", now the times are ", tot_durations)
end
@show tot_durations
@show maximum(tot_durations), maximum(tot_durations)/60






for i in 1:6
    println("floor ", i, " -> ", tripDuration(Set([i])))
end
