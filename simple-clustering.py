# import numpy as np # matrix & array manipulation
# from matplotlib import pyplot as plt # plotting

class Elevator:
    def __init__(self, number, status, seconds_left, cluster=None):
        self.number = number # indexing just to keep track
        self.status = status # either loading, unloading, moving up, or moving down
        self.seconds_left = seconds_left # how many seconds left until status can change, important for the Elevatorinator
        self.cluster = cluster # cluster that it is engaged with, if any

    def __str__(self):
        return f"Elevator {self.number} is {self.status} with {self.seconds_left} left \n Cluster: {self.cluster.number if self.cluster else None}"

class Cluster:
    def __init__(self, number, size, destination, status, seconds_left, elevator=None):
        self.number = number # indexing just to keep track
        self.size = size # number of workers in the cluster
        self.destination = destination
        self.status = status # either waiting, loading, moving up, unloading, or arrived
        self.seconds_left = seconds_left # how many seconds left until status can change
        self.elevator = elevator # elevator that it is engaged with, if any

    def __str__(self):
        return f"Cluster with destination Floor {self.destination} is {self.status}\n Elevator: {self.elevator.number}"

def is_done(clusters):
    for cluster in clusters:
        if cluster.status != "arrived":
            return False
    return True

if __name__=='__main__': # code runs from here
    """ PARAMETERS """
    floors = {1: 100, 2: 120, 3: 60, 4: 120, 5: 80, 6: 20}
    elapsed_time = 0 # will tick up as events happen
    elevators = []
    loading_time = 15
    moving_btwn_floors_time = 5
    unloading_time = 10
    reopening_time = 8 # shouldn't be an issue since we're assuming this never happens
    """ PARAMETERS """

    # initialize clusters - this is the heart of the method. TODO: make this a function instead of hard coding
    clusters = []
    count = 0
    for floor in floors:
        to_go = floors[floor] # number of workers that must be assigned to a cluster, ticks down as clusters are created
        while to_go > 0:
            clusters.append(Cluster(count, 10, floor, "waiting", loading_time))
            to_go -= 10
            count += 1

    # begin the loading process
    for i in range(4):
        elevators.append(Elevator(i, "loading", loading_time, clusters[i]))
        clusters[i].status = "loading"
        clusters[i].elevator = elevators[i]
        clusters[i].seconds_left = loading_time
        elevators[i].cluster = clusters[i]
        clusters[i].elevator = elevators[i]

    next_cluster = len(elevators) + 1 # index of the next cluster to be loaded

    # main loop to simulate the elevators. I shall call it, "The Elevatorinator!"
    while not is_done(clusters):
        for elevator in elevators:
            if elapsed_time % 5 == 0:
                print(elevator)
            if elevator.seconds_left == 0:
                # elevator is done with its current task, so it must do something else depending on what it can do
                if elevator.status == "loading":
                    elevator.status = "moving up"
                    elevator.seconds_left = moving_btwn_floors_time * elevator.cluster.destination
                    elevator.cluster.status = "moving up"
                    elevator.cluster.seconds_left = elevator.seconds_left
                elif elevator.status == "moving up":
                    elevator.status = "unloading"
                    elevator.seconds_left = unloading_time
                    elevator.cluster.status = "unloading"
                    elevator.cluster.seconds_left = elevator.seconds_left
                elif elevator.status == "unloading":
                    elevator.status = "moving down"
                    elevator.seconds_left = moving_btwn_floors_time * elevator.cluster.destination
                    elevator.cluster.status = "arrived"
                    elevator.cluster.elevator = None
                    elevator.cluster = None
                elif elevator.status == "moving down":
                    elevator.status = "loading"
                    elevator.seconds_left = loading_time
                    if next_cluster < len(clusters):
                        elevator.cluster = clusters[next_cluster]
                        elevator.cluster.status = "loading"
                        elevator.cluster.elevator = elevator
                        elevator.cluster.seconds_left = elevator.seconds_left
                        next_cluster += 1
                    else:
                        elevator.cluster = None
                else:
                    raise Exception('Elevator is in an invalid state')
            else:
                elevator.seconds_left -= 1
        elapsed_time += 1
        print(f'Elapsed time: {elapsed_time} seconds')

    print(f'All clusters have arrived at their destinations in {elapsed_time} seconds')