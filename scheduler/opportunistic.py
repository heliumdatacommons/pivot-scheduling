import numpy as np

from scheduler import GlobalSchedulerBase


class OpportunisticGlobalScheduler(GlobalSchedulerBase):

  def __init__(self, *args, **kwargs):
    super(OpportunisticGlobalScheduler, self).__init__(*args, **kwargs)

  def schedule(self, tasks):
    resc = self.resource_info
    for t in tasks:
      t_demand = np.array([t.cpus, t.mem, t.disk, t.gpus])
      qualified = [hid for hid, r in resc.items() if np.all(r >= t_demand)]
      if len(qualified) > 0:
        h = self.cluster.get_host(self.randomizer.choice(qualified))
        resc[h.id] -= t_demand
        t.placement = h.id
    return list(tasks)