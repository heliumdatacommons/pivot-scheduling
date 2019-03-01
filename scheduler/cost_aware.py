import numpy as np
import numpy.linalg as la

import application

from collections import defaultdict, Counter

from scheduler import GlobalSchedulerBase


class CostAwareGlobalScheduler(GlobalSchedulerBase):

  def __init__(self, *args, **kwargs):
    bin_pack_algo = str(kwargs.pop('bin_pack_algo', 'first-fit'))
    sort_tasks = bool(kwargs.pop('sort_tasks', False))
    sort_hosts = bool(kwargs.pop('sort_hosts', False))
    realtime_bw = kwargs.pop('realtime_bw',False)
    host_decay = kwargs.pop('host_decay', False)
    super(CostAwareGlobalScheduler, self).__init__(*args, **kwargs)
    self.__bin_pack_algo = bin_pack_algo
    self.__sort_tasks = sort_tasks
    self.__sort_hosts = sort_hosts
    self.__realtime_bw = realtime_bw
    self.__host_decay = host_decay
    if host_decay:
      self.__n_container = None

  def schedule(self, tasks):
    bin_pack_algo = self.__bin_pack_algo
    if bin_pack_algo == 'first-fit':
      bin_pack_algo = self._first_fit
    elif bin_pack_algo == 'best-fit':
      bin_pack_algo = self._best_fit
    env, storage, hosts = self.env, self.cluster.storage, self.cluster.hosts
    resc = self.resource_info
    groups = self._group_tasks(tasks)
    for anchor, task_group in groups.items():
      if isinstance(anchor, application.Application):
        anchor = self.randomizer.choice(storage)
      if self.__sort_tasks:
        task_group = self._sort_tasks(task_group)
      bin_pack_algo(hosts, task_group, anchor, resc)
    return tasks

  def _group_tasks(self, tasks):
    cluster = self.cluster
    groups = defaultdict(list)
    for t in tasks:
      c, app = t.container, t.container.application
      preds = [t for p in app.get_predecessors(c.id) for t in p.tasks]
      if preds:
        placement, _ = max(Counter([t.placement for t in preds]).items(), key=lambda x: x[1])
        locality = cluster.get_host(placement).locality
        data_src = cluster.get_storage_by_locality(locality)
        groups[data_src] += t,
      else:
        groups[app] += t,
    return groups

  def _sort_tasks(self, tasks):
    return sorted(tasks, key=lambda t: -la.norm(np.array([t.cpus, t.mem, t.disk, t.gpus]), 2))

  def _best_fit(self, hosts, tasks, anchor, resc):
    env, cluster, meta = self.env, self.cluster, self.cluster.meta
    host_decay, rt_bw = self.__host_decay, self.__realtime_bw
    if host_decay:
      n_container = self.__n_container

    def host_score_func(item):
      h, t = item
      t_demand = np.array([t.cpus, t.mem, t.disk, t.gpus])
      r = la.norm(resc[h.id] - t_demand, 2)
      in_route = cluster.get_route(anchor.id, h.id)
      out_route = cluster.get_route(h.id, anchor.id)
      if round(in_route.bw, 3) != round(in_route.realtime_bw, 3):
        self.logger.debug('bw: %.3f, rt bw: %.3f'%(in_route.bw, in_route.realtime_bw))
      if round(out_route.bw, 3) != round(out_route.realtime_bw, 3):
        self.logger.debug('bw: %.3f, rt bw: %.3f'%(out_route.bw, out_route.realtime_bw))
      bw = (in_route.realtime_bw + out_route.realtime_bw) if rt_bw else (in_route.bw + out_route.bw)
      t = meta.cost[(anchor.locality, h.locality)] + meta.cost[(h.locality, anchor.locality)]
      decay = max(n_container[h.id] if host_decay else 0, 1)

      return t * r * decay/bw

    for t in tasks:
      t_demand = np.array([t.cpus, t.mem, t.disk, t.gpus])
      candidates = [(h, t) for h in hosts if np.all(resc[h.id] >= t_demand)]
      if len(candidates) == 0:
        self.logger.debug('[%.3f] Task %s is put into the waiting queue' % (env.now, t.id))
        self.logger.debug('Demand: %.1f cpus, %.1f mem, %.1f disk, %.1f gpus' % (t.cpus, t.mem, t.disk, t.gpus))
      else:
        h, _ = min(candidates, key=host_score_func)
        self.logger.debug('[%.3f] Container %s is placed on host %s' % (env.now, t.id, h.id))
        t.placement = h.id
        resc[h.id] -= t_demand
        if host_decay:
          n_container[h.id] += 1

  def _first_fit(self, hosts, tasks, data_src, resc):
    env, cluster, meta = self.env, self.cluster, self.cluster.meta
    sort_hosts, rt_bw = self.__sort_hosts, self.__realtime_bw
    host_decay = self.__host_decay

    def host_score_func(h):
      r = la.norm(resc[h.id], 2)
      in_route = cluster.get_route(data_src.id, h.id)
      out_route = cluster.get_route(h.id, data_src.id)
      if round(in_route.bw, 3) != round(in_route.realtime_bw, 3):
        self.logger.debug('bw: %.3f, rt bw: %.3f'%(in_route.bw, in_route.realtime_bw))
      if round(out_route.bw, 3) != round(out_route.realtime_bw, 3):
        self.logger.debug('bw: %.3f, rt bw: %.3f'%(out_route.bw, out_route.realtime_bw))
      bw = (in_route.realtime_bw + out_route.realtime_bw) if rt_bw else (in_route.bw + out_route.bw)

      c = meta.cost[(data_src.locality, h.locality)] + meta.cost[(h.locality, data_src.locality)]
      df = max(len(h.tasks) if host_decay else 0, 1)
      return c * df/(r * bw)

    if sort_hosts:
      hosts = sorted(hosts, key=host_score_func)
    for t in tasks:
      t_demand = np.array([t.cpus, t.mem, t.disk, t.gpus])
      for i, h in enumerate(hosts):
        r = resc[h.id]
        if np.all(r > t_demand):
          t.placement = h.id
          r -= t_demand
          break
