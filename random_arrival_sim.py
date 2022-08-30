# fine i'll do it in python >:(

import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum
from queue import PriorityQueue
from operator import attrgetter as ag, itemgetter as ig

N_ELEVATORS = 4
N_FLOORS = 6
WORKERS_BY_FLOOR = [100, 120, 60, 120, 80, 20]

ELEVATOR_CAPACITY = 10
ELEVATOR_GROUNDFLOOR_TIME = 15
ELEVATOR_TRAVEL_TIME = 5
ELEVATOR_UNLOAD_TIME = 10


class Whappened(Enum):  # the events that will trigger a call to the strategy
    WORKER_ARRIVED = 1
    ELEVATOR_ARRIVED = 2

@dataclass
class ElevatorMoment:
    etas_by_elv: NDArray[np.float32]
    waiting_by_floor: NDArray[np.int32]
    arrival_etas_by_floor: NDArray[np.float32]

    whappened: Whappened

    processed_by_elv: List[List[Tuple[int, int]]]   # processed = List[(floor, count)]



# strategies take in the current state (ElevatorMoment) and return what clusters to send to each elevator in the form of ElevatorMoment.arrival_etas_by_floor "proccessed"
def strat_3a0_donothing(em):
    return [[]]*4
def strat_3a1_nochange(em):
    return [[]]*4
STRATS = {
    '3a1': strat_3a1_nochange,
}


@dataclass(order=True)
class ElevatorSimEvents:
    time: float
    whappened: Whappened=field(compare=False)
    data: int   # either elevator idx or floor idx

def get_trip_time_by_sent_subclusters(subclusters: List[Tuple[int, int]]):
    return ELEVATOR_GROUNDFLOOR_TIME + 2*ELEVATOR_TRAVEL_TIME*max(f for f, _ in subclusters) + ELEVATOR_UNLOAD_TIME*len(set(f for f, _ in subclusters))

def simulate(strategy, arrival_times_by_floor: List[NDArray[np.float32]], init_time=-3600, max_steps=sum(WORKERS_BY_FLOOR)*4):

    # internal state
    cur_time = init_time
    ready_time_by_elv = np.zeros(N_ELEVATORS)
    n_waiting_by_floor = np.zeros(N_FLOORS, dtype=np.int32)
    processed_by_elv = [[] for _ in range(N_ELEVATORS)] # use list comp not *N_ELEVATORS bc that would shallow copy

    # initialize all the statistics values
    idle_time_by_elevator = np.zeros(N_ELEVATORS)
    arrival_times_by_floor = [[] for _ in range(N_FLOORS)]

    # main sim loop
    event_pq = PriorityQueue(sum(WORKERS_BY_FLOOR))     # initialize with enough capacity for every worker's arrival event
    for f, times in arrival_times_by_floor:             # insert all the arrival events into the pq
        for t in times:
            event_pq.put(ElevatorSimEvents(t, Whappened.WORKER_ARRIVED, f))
    for _ in range(max_steps):
        if event_pq.empty(): break
        cur_time, whappened, data = ag('time', 'whappened', 'data')(event_pq.get())

        match whappened:
            case Whappened.WORKER_ARRIVED:
                n_waiting_by_floor[data] += 1
            case Whappened.ELEVATOR_ARRIVED:
                pass

        # figure out what the relative state (state from this point in time) is
        arrival_etas_by_floor = [arrival_time - cur_time for arrival_time in arrival_times_by_floor]                            # convert absolute times to arrival etas
        arrival_etas_by_floor = [arrival_time[np.searchsorted(arrival_time, 0):] for arrival_time in arrival_times_by_floor]    # slice the front off so only people who have yet to arrive are in the etas
        elevator_moment = ElevatorMoment(np.maximum(ready_time_by_elv - cur_time, 0), n_waiting_by_floor, arrival_etas_by_floor, whappened, processed_by_elv)

        send_to_elvs = strategy(elevator_moment)

        assert len(send_to_elvs) == N_ELEVATORS
        for elv, to_send in enumerate(send_to_elvs):
            if to_send is None or len(to_send) == 0: continue   # do nothing for this elevator if we weren't told to
            if ready_time_by_elv[elv] > cur_time:                    # do nothing since this elevator is not ready yet
                print(f"{cur_time:6.2f} WARNING: tried to send {to_send} to elevator {elv} but it's not due back for another {ready_time_by_elv[elv]-cur_time} seconds")
                continue
            if sum([count for _, count in to_send]) > ELEVATOR_CAPACITY:
                print(f"{cur_time:6.2f} WARNING: tried to send {'+'.join(count for _, count in to_send)} = {sum(count for _, count in to_send)} people but elevator cap is {ELEVATOR_CAPACITY}")
                continue
            for floor, count in to_send:
                if n_waiting_by_floor[floor] < count:
                    print(f"{cur_time:6.2f} WARNING: tried to send group of size {count} to floor {floor} but only {n_waiting_by_floor[floor]} are here")
                    break
            else: # we should be good to at least send someone, since the elevator is valid and the clusters are valid
                # stats update
                idle_time_by_elevator[elv] += cur_time - ready_time_by_elv[elv]
                trip_time, prev_floor = ELEVATOR_GROUNDFLOOR_TIME, 0
                for f, n in sorted(to_send, key=lambda scl: scl[0]):
                    trip_time += ELEVATOR_TRAVEL_TIME * f - prev_floor + ELEVATOR_UNLOAD_TIME   # DECISION: maybe trip time shouldn't count the unload time as ppl are getting out
                    prev_floor = f
                    arrival_times_by_floor[f] += [trip_time]*n

                # state update
                ready_time_by_elv[elv] = cur_time + get_trip_time_by_sent_subclusters(to_send)
                processed_by_elv[elv] += to_send
                for f, n in to_send: n_waiting_by_floor[f] -= n

                event_pq.put(ElevatorSimEvents(ready_time_by_elv[elv], Whappened.ELEVATOR_ARRIVED, elv))



if __name__ == '__main__':
    # TODO write random sample function
    # TODO: test everything
    print("hewo world")



# from matplotlib import pyplot as plt
#
# import numpy as np
# rng = np.random.default_rng(seed=42)
# vals = rng.standard_normal(int(1e6))
# plt.hist(vals, bins=30)
# plt.show()
#

# from dataclasses import dataclass, field
# from typing import Any
#
# @dataclass(order=True)
# class PrioritizedItem:
#     priority: int
#     item: Any=field(compare=False)
#
# from queue import PriorityQueue
# pq = PriorityQueue()
# pq.put(PrioritizedItem(3, 'going'))
# pq.put(PrioritizedItem(1, 'coming'))
#
# while not pq.empty():
#     got = pq.get()
#     print(got)
