import numpy as np # matrix & array manipulation
# from matplotlib import pyplot as plt # plotting

class Elevator:
    def __init__(self, number, status):
        self.number = number # indexing just to keep track
        self.status = status # either loading, unloading, moving up, or moving down

    def __str__(self):
        return f"Elevator {self.number} is {self.status}"

class Cluster:
    def __init__(self, size, destination, status):
        self.size = size # number of workers in the cluster
        self.destination = destination
        self.status = status # either waiting, loading, moving up, unloading, or arrived

    def __str__(self):
        return f"Cluster with destination Floor {self.destination} is {self.status}"

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
    for i in range(4):
        elevators.append(Elevator(i, "loading"))
    """ PARAMETERS """

    # initialize clusters - this is the heart of the method. TODO: make this a function instead of hardcoding
    clusters = []
    for floor in floors:
        to_go = floors[floor] # number of workers that must be assigned to a cluster, ticks down as clusters are created
        while to_go > 0:
            clusters.append(Cluster(10, floor, "waiting"))
            to_go -= 10

    # main loop to simulate the elevators. I shall call it, "The Elevatorinator!"