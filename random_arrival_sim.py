# fine i'll do it in python >:(

from matplotlib import pyplot as plt

import numpy as np
rng = np.random.default_rng(seed=42)
vals = rng.standard_normal(int(1e6))
plt.hist(vals, bins=30)
plt.show()


from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

from queue import PriorityQueue
pq = PriorityQueue()
pq.put(PrioritizedItem(3, 'going'))
pq.put(PrioritizedItem(1, 'coming'))

while not pq.empty():
    got = pq.get()
    print(got)
