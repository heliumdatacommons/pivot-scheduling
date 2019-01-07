import simpy
import numpy as np
import numpy.random as rnd

from resource import Cluster, Host, Storage, ResourceMetadata
from resource.network import NetworkRoute
from util import Loggable


class RandomClusterGenerator(Loggable):

  def __init__(self, env, dispatch_q, notify_q,
               cpus_lo, cpus_hi, mem_lo, mem_hi, disk_lo, disk_hi, gpus_lo, gpus_hi, seed=None):
    assert isinstance(env, simpy.Environment)
    assert isinstance(dispatch_q, simpy.Store)
    assert isinstance(notify_q, simpy.Store)
    assert 0 < cpus_lo <= cpus_hi
    assert 0 < mem_lo <= mem_hi
    assert 0 < disk_lo <= disk_hi
    assert 0 <= gpus_lo <= gpus_hi
    self.__env = env
    self.__dispatch_q = dispatch_q
    self.__notify_q = notify_q
    self.__cpus_lo, self.__cpus_hi = cpus_lo, cpus_hi
    self.__mem_lo, self.__mem_hi = mem_lo, mem_hi
    self.__disk_lo, self.__disk_hi = disk_lo, disk_hi
    self.__gpus_lo, self.__gpus_hi = gpus_lo, gpus_hi
    self.__meta = ResourceMetadata()
    rnd.seed(seed)

  def generate(self, n_host, uniform=True):
    hosts = self._generate_hosts(n_host, uniform)
    storage = self._generate_storage(hosts)
    routes = self._generate_routes(hosts, storage)
    return Cluster(self.__env, self.__dispatch_q, self.__notify_q,
                   hosts=hosts, storage=storage, routes=routes, meta=self.__meta)

  def _generate_hosts(self, n_host, uniform=True):
    assert isinstance(n_host, int) and n_host > 0
    meta = self.__meta
    if uniform:
      cpus = int(rnd.choice(np.arange(self.__cpus_lo, self.__cpus_hi + 2, 2)))
      mem = int(rnd.choice(np.arange(self.__mem_lo, self.__mem_hi + 1024, 1024)))
      disk = int(rnd.choice(np.arange(self.__disk_lo, self.__disk_hi + 1024, 1024)))
      gpus = int(rnd.randint(self.__gpus_lo, self.__gpus_hi + 1))
      return [Host(self.__env, str(i), cpus, mem, disk, gpus, locality=rnd.choice(meta.zones))
              for i in range(n_host)]
    else:
      return [Host(self.__env, str(i),
                   int(rnd.choice(np.arange(self.__cpus_lo, self.__cpus_hi + 2, 2))),
                   int(rnd.choice(np.arange(self.__mem_lo, self.__mem_hi + 1024, 1024))),
                   int(rnd.choice(np.arange(self.__disk_lo, self.__disk_hi + 1024, 1024))),
                   int(rnd.randint(self.__gpus_lo, self.__gpus_hi + 1)),
                   locality=rnd.choice(meta.zones))
              for i in range(n_host)]

  def _generate_storage(self, hosts):
    return [Storage(self.__env, 's-%d'%i, locality=l)
            for i, l in enumerate(set([h.locality for h in hosts]))]

  def _generate_routes(self, hosts, storage):
    env, meta = self.__env, self.__meta
    routes = []
    for src in hosts:
      for dst in hosts:
        if src == dst:
          continue
        routes += NetworkRoute(env, src, dst, meta.bw[(src.locality, dst.locality)]),
    for h in hosts:
      for s in storage:
        routes += [NetworkRoute(env, h, s, meta.bw[(h.locality, s.locality)]),
                   NetworkRoute(env, s, h, meta.bw[(s.locality, h.locality)])]
    return routes








