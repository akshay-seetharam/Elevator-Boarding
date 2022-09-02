# fine i'll do it in python >:(

import numpy as np
from numpy.typing import NDArray
from tqdm import trange, tqdm
from pprint import pprint
from matplotlib import pyplot as plt

from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum
from queue import PriorityQueue
from operator import attrgetter as ag, itemgetter as ig
from functools import reduce

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

@dataclass(order=True)
class ElevatorSimEvents:
    time: float
    whappened: Whappened=field(compare=False)
    data: int   # either elevator idx or floor idx

@dataclass
class ElevatorMoment:
    etas_by_elv: NDArray[np.float32]
    waiting_by_floor: NDArray[np.int32]
    arrival_etas_by_floor: NDArray[np.float32]

    # whappened: Whappened
    sim_event: ElevatorSimEvents

    # processed_by_elv: List[List[Tuple[int, int]]]   # processed = List[(floor, count)]


# DEPRECATED strategies take in the current state (ElevatorMoment) and return what clusters to send to each elevator in the form of ElevatorMoment.arrival_etas_by_floor "proccessed"
# strategies take in the current state (ElevatorMoment) and return what clusters to send to each elevator in the form of ElevatorMoment.arrival_etas_by_floor "proccessed"
def strat_3a0_donothing(em):
    return [[]]*4
# def strat_3a1_nochange(em):
#     return [[]]*4
class strat_3a1_nochange:
    def __init__(self, arrival_times_by_floor):
        self.arrival_times_by_floor = arrival_times_by_floor
        self.proccessed_id_by_floor = np.zeros(N_ELEVATORS)
        self.sequential_arrival = []    # OPTIM: maybe make this a dequeue

    def step(self, em):
        match em.sim_event.whappened:
            case Whappened.WORKER_ARRIVED:
                self.sequential_arrival.append(em.sim_event.data)
                return [[]]*N_ELEVATORS
            case Whappened.ELEVATOR_ARRIVED:
                elev_to_send = em.sim_event.data
                ret = [[] for _ in range(N_ELEVATORS)]
                ret[elev_to_send] = self.sequential_arrival[:ELEVATOR_CAPACITY]
                self.sequential_arrival = self.sequential_arrival[ELEVATOR_CAPACITY:]
                return ret

STRATS = {
    '3a1': strat_3a1_nochange,
}


def get_trip_time_by_sent_subclusters(subclusters: List[Tuple[int, int]]):
    return ELEVATOR_GROUNDFLOOR_TIME + 2*ELEVATOR_TRAVEL_TIME*max(f+1 for f, _ in subclusters) + ELEVATOR_UNLOAD_TIME*len(set(f for f, _ in subclusters))

def simulate(strategy, arrival_times_by_floor: List[NDArray[np.float32]], init_time=-3600, max_steps=sum(WORKERS_BY_FLOOR)*4):

    # internal state
    cur_time = float(init_time)
    ready_time_by_elv = np.zeros(N_ELEVATORS)
    n_waiting_by_floor = np.zeros(N_FLOORS, dtype=np.int32)

    # initialize all the statistics values
    idle_time_by_elevator = np.zeros(N_ELEVATORS)
    trip_distinct_floors_by_elevator = [[] for _ in range(N_FLOORS)]
    final_times_by_floor = [[] for _ in range(N_FLOORS)]

    # main sim loop
    event_pq = PriorityQueue()     # initialize with enough capacity for every worker's arrival event
    for f, times in enumerate(tqdm(arrival_times_by_floor, desc='inserting init ppl')):             # insert all the arrival events into the pq
        for t in times:
            event_pq.put(ElevatorSimEvents(t, Whappened.WORKER_ARRIVED, f))
    for i in trange(N_ELEVATORS, desc='inserting init elevators'):
        print(event_pq.qsize())
        event_pq.put_nowait(ElevatorSimEvents(0, Whappened.ELEVATOR_ARRIVED, i))

    print("go time!")

    for _ in trange(max_steps):
        if event_pq.empty(): break
        cur_time, whappened, data = ag('time', 'whappened', 'data')(event_pq.get())

        match whappened:
            case Whappened.WORKER_ARRIVED:
                n_waiting_by_floor[data] += 1
            case Whappened.ELEVATOR_ARRIVED:
                pass

        # figure out what the relative state (state from this point in time) is
        arrival_etas_by_floor = [arrival_times - cur_time for arrival_times in arrival_times_by_floor]                            # convert absolute times to arrival etas
        arrival_etas_by_floor = [arrival_times[np.searchsorted(arrival_times, 0):] for arrival_times in arrival_times_by_floor]    # slice the front off so only people who have yet to arrive are in the etas
        elevator_moment = ElevatorMoment(
            etas_by_elv=np.maximum(ready_time_by_elv - cur_time, 0),
            waiting_by_floor=n_waiting_by_floor,
            arrival_etas_by_floor=arrival_etas_by_floor,
            sim_event=ElevatorSimEvents(cur_time, whappened, data),
            # processed_by_elv=processed_by_elv,
        )

        send_to_elvs = strategy(elevator_moment)

        assert len(send_to_elvs) == N_ELEVATORS
        for elv, to_send in enumerate(send_to_elvs):
            if to_send is None or len(to_send) == 0: continue   # do nothing for this elevator if we weren't told to
            if ready_time_by_elv[elv] > cur_time:                    # do nothing since this elevator is not ready yet
                print(f"{cur_time:6.2f} WARNING: tried to send {to_send} to elevator {elv} but it's not due back for another {ready_time_by_elv[elv]-cur_time} seconds")
                continue
            # for to_send = [floor, floor, floor, ...]
            if len(to_send) > ELEVATOR_CAPACITY:
                print(f"{cur_time:6.2f} WARNING: tried to send {'+'.join(count for _, count in to_send)} = {sum(count for _, count in to_send)} people but elevator cap is {ELEVATOR_CAPACITY}")
            pprint(to_send)
            for floor in to_send:
                if n_waiting_by_floor[floor] < 1:
                    print(f"{cur_time:6.2f} WARNING: tried to send group of size 1 to floor {floor} but only {n_waiting_by_floor[floor]} are here")
                    break
            # # for to_send = [(floor, count)]
            # if sum([count for _, count in to_send]) > ELEVATOR_CAPACITY:
            #     print(f"{cur_time:6.2f} WARNING: tried to send {'+'.join(count for _, count in to_send)} = {sum(count for _, count in to_send)} people but elevator cap is {ELEVATOR_CAPACITY}")
            #     continue
            # for floor, count in to_send:
            #     if n_waiting_by_floor[floor] < count:
            #         print(f"{cur_time:6.2f} WARNING: tried to send group of size {count} to floor {floor} but only {n_waiting_by_floor[floor]} are here")
            #         break
            else: # we should be good to at least send someone, since the elevator is valid and the clusters are valid
                to_send_bucketed = list({ f: sum(1 for tf in to_send if tf == f) for f in set(to_send) }.items())
                print(f"                          {to_send} -> {to_send_bucketed}")
                to_send = to_send_bucketed



                # stats update
                idle_time_by_elevator[elv] += cur_time - ready_time_by_elv[elv]
                trip_distinct_floors_by_elevator[elv].append(len(to_send))
                trip_time, prev_floor = ELEVATOR_GROUNDFLOOR_TIME, 0
                for f, n in sorted(to_send, key=lambda scl: scl[0]):
                    trip_time += ELEVATOR_TRAVEL_TIME * f - prev_floor + ELEVATOR_UNLOAD_TIME   # DECISION: maybe trip time shouldn't count the unload time as ppl are getting out
                    prev_floor = f
                    print(f, n)
                    final_times_by_floor[f] += [trip_time]*n

                # state update
                ready_time_by_elv[elv] = cur_time + get_trip_time_by_sent_subclusters(to_send)
                # processed_by_elv[elv] += to_send
                for f, n in to_send: n_waiting_by_floor[f] -= n

                event_pq.put(ElevatorSimEvents(ready_time_by_elv[elv], Whappened.ELEVATOR_ARRIVED, elv))

    return cur_time, idle_time_by_elevator, final_times_by_floor, trip_distinct_floors_by_elevator


def arrival_instantaneous():
    # return [-np.ones(count) for count in WORKERS_BY_FLOOR]
    return [np.random.uniform(-1, 0, [count]) for count in WORKERS_BY_FLOOR]


if __name__ == '__main__':
    arrival_times_by_floor = arrival_instantaneous()
    print(arrival_times_by_floor)
    cringe_strategy = strat_3a1_nochange(arrival_times_by_floor)
    tot_time, idle_time_by_elevator, arrival_times_by_floor, trip_distinct_floors_by_elevator = simulate(cringe_strategy.step, arrival_times_by_floor, init_time=0)
    print(tot_time)

    fig, ax = plt.subplots()
    pprint(trip_distinct_floors_by_elevator)
    ax.hist(reduce(lambda a, c: a + c, trip_distinct_floors_by_elevator))
    ax.set_xlabel('number of distinct floors per trip')
    ax.set_ylabel('number of trips')
    ax.set_title('histogram of floors visited per trip')
    plt.show()



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
