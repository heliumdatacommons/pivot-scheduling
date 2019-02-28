import numpy as np
import numpy.linalg as la

from scheduler import GlobalSchedulerBase

class FirstFitGlobalScheduler(GlobalSchedulerBase):

  def __init__(self, *args, **kwargs):
    decreasing = str(kwargs.pop('decreasing', False))
    super(FirstFitGlobalScheduler, self).__init__(*args, **kwargs)
    self.__decreasing = decreasing

  def schedule(self, tasks):
    env, hosts = self.env, self.cluster.hosts
    resc = self.resource_info
    if self.__decreasing:
      tasks = self._sort_tasks(tasks)
    for t in tasks:
      t_demand = np.array([t.cpus, t.mem, t.disk, t.gpus])
      for h in hosts:
        r = resc[h.id]
        if np.all(r >= t_demand):
          t.placement = h.id
          r -= t_demand
          break
    return tasks

  def _sort_tasks(self, tasks):
    return sorted(tasks, key=lambda t: -la.norm(np.array([t.cpus, t.mem, t.disk, t.gpus]), 2))


class BestFitGlobalScheduler(GlobalSchedulerBase):

  def __init__(self, *args, **kwargs):
    decreasing = str(kwargs.pop('decreasing', False))
    super(BestFitGlobalScheduler, self).__init__(*args, **kwargs)
    self.__decreasing = decreasing

  def schedule(self, tasks):
    resc = self.resource_info
    if self.__decreasing:
      tasks = sorted(tasks, key=lambda t: -la.norm(np.array([t.cpus, t.mem, t.disk, t.gpus]), 2))
    for t in tasks:
      t_demand = np.array([t.cpus, t.mem, t.disk, t.gpus])
      qualified = [(hid, r) for hid, r in resc.items() if np.all(r > t_demand)]
      if qualified:
        _, hid, r = min([(la.norm(r - t_demand, 2), hid, r) for hid, r in qualified])
        t.placement = hid
        r -= t_demand
    return tasks