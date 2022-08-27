using Distributions
using ResumableFunctions
using SimJulia
using ProgressMeter

using Random

const RUNS = 500
const N = 10
const S = 3
const SEED = 150
const LAMBDA = 100
const MU = 1

Random.seed!(SEED)
const F = Exponential(LAMBDA)
const G = Exponential(MU)

@resumable function machine(env::Environment, repair_facility::Resource, spares::Store{Process})
    while true
        try @yield timeout(env, Inf) finally
            @yield timeout(env, rand(F))
            get_spare = get(spares)
            @yield get_spare | timeout(env)
            if state(get_spare) != SimJulia.idle 
                @yield interrupt(value(get_spare))
            else
                throw(StopSimulation("No more spares!"))
            end
            @yield request(repair_facility)
            @yield timeout(env, rand(G))
            @yield release(repair_facility)
            @yield put(spares, active_process(env))
        end
    end
end

@resumable function start_sim(env::Environment, repair_facility::Resource, spares::Store{Process})
    for i in 1:N
        proc = @process machine(env, repair_facility, spares)
        @yield interrupt(proc)
    end
    for i in 1:S
        proc = @process machine(env, repair_facility, spares)
        @yield put(spares, proc) 
    end
end

function sim_repair()
    sim = Simulation()
    repair_facility = Resource(sim)
    spares = Store{Process}(sim)
    @process start_sim(sim, repair_facility, spares)
    msg = run(sim)
    stop_time = now(sim)
    # println("At time $stop_time: $msg")
    stop_time
end

results = Float64[]
@showprogress for i in 1:RUNS push!(results, sim_repair()) end
println("Average crash time: ", sum(results)/RUNS)


# plotting
using GLMakie
f = Figure()
ax = Axis(f[1, 1], title="Histogram of Arrive Time per Worker by strategy")
hist!(ax, results, bins=20)
display(f)  # run this script with julia -i to launch interactive and keep the window around

