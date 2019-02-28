import os
import json
import simpy
import numpy as np

import resources

from collections import defaultdict

from util import Loggable, floor, ceil


class Meter(Loggable):

  def __init__(self, env):
    assert isinstance(env, simpy.Environment)
    self.__meta = resources.ResourceMetadata()
    self.__env = env
    self.__hosts = defaultdict(list)
    self.__routes = defaultdict(dict)
    self.__usage = defaultdict(dict)
    self.__data_transfers = []
    self.__sched_turnovers = []
    self.__n_sched_ops = 0

  @property
  def runtime(self):
    return self.__env.now

  @property
  def cumulative_instance_hours(self):
    return sum([sum([v[1] - v[0] for v in vals]) for h, vals in self.__hosts.items()])/3600

  @property
  def total_network_traffic_cost(self):
    routes, meta = self.__routes, self.__meta
    cost = 0
    for r, pkts in routes.items():
      nbytes = sum([sum(nbytes for _, _, nbytes in trans) for trans in pkts.values()])
      cost += meta.calc_network_traffic_cost(r.src, r.dst, nbytes)
    return cost

  @property
  def average_congestion_delay(self):
    delay, n_pkts = 0, 0
    for pkts in self.__routes.values():
      n_pkts += len(pkts)
      for p in pkts.values():
        for i, (start, end, _) in enumerate(p):
          if i > 0:
            delay += start - p[i - 1][1]
    return n_pkts and delay/n_pkts

  def initialize(self):
    self.__hosts = defaultdict(list)
    self.__routes = defaultdict(dict)
    self.__meta = resources.ResourceMetadata()

  def host_check_in(self, h):
    env, hosts = self.__env, self.__hosts
    last_chkpt = hosts[h][-1] if hosts[h] else None
    self._track_resource_usage(h)
    if last_chkpt is None:
      hosts[h] += [env.now],
    elif len(last_chkpt) == 2:
      if env.now > last_chkpt[-1]:
        hosts[h] += [env.now],
      else:
        last_chkpt.pop()

  def host_check_out(self, h):
    env, hosts = self.__env, self.__hosts
    last_chkpt = hosts[h][-1] if hosts[h] else None
    self._track_resource_usage(h)
    if last_chkpt is None:
      raise Exception('Check-out occurs before any check-in')
    if len(last_chkpt) == 1:
      last_chkpt += env.now,
    elif len(last_chkpt) == 2:
      if env.now > last_chkpt[-1]:
        last_chkpt[-1] = env.now

  def route_check_in(self, r, pkt_id):
    self.__routes[r].setdefault(pkt_id, []).append([self.__env.now])

  def route_check_out(self, r, pkt_id, nbytes):
    self.__routes[r][pkt_id][-1] += [self.__env.now, nbytes]

  def add_data_transfer(self, timepoint, srcs, dst, data_amt, duration, prop_delay, avg_bw, avg_egress_cost):
    data_transfers = self.__data_transfers
    data_transfers += {
      'timestamp': timepoint,
      'from': [[s.cloud.value, s.region.value, s.zone.value] for s in srcs],
      'to': [dst.cloud.value, dst.region.value, dst.zone.value],
      'data_amt': data_amt,
      'total_delay': duration,
      'propagation_delay': prop_delay,
      'avg_bw': avg_bw,
      'avg_egress_cost': avg_egress_cost
    },

  def add_scheduling_turnover(self, timepoint):
    self.__sched_turnovers += timepoint,

  def increment_scheduling_ops(self, n_ops):
    self.__n_sched_ops += n_ops

  def save(self, data_dir):
    if not os.path.exists(data_dir):
      os.makedirs(data_dir, exist_ok=True)
    # host_usage = open('%s/host_usage.json' % data_dir, 'w')
    # cpu_usage = open('%s/cpu_usage.json' % data_dir, 'w')
    # mem_usage = open('%s/mem_usage.json' % data_dir, 'w')
    # disk_usage = open('%s/disk_usage.json' % data_dir, 'w')
    # gpu_usage = open('%s/gpu_usage.json' % data_dir, 'w')
    with open('%s/general.json'%data_dir, 'w') as f:
      json.dump({
        'egress_cost': self.total_network_traffic_cost,
        'cum_instance_hours': self.cumulative_instance_hours
      }, f)
    with open('%s/transfers.json'%data_dir, 'w') as f:
      json.dump([t for t in self.__data_transfers], f)
    with open('%s/scheduler.json'%data_dir, 'w') as f:
      json.dump({
        'turnovers': self.__sched_turnovers,
        'total_scheduling_ops': self.__n_sched_ops
      }, f)
    with open('%s/host_usage.json'%data_dir, 'w') as f:
      x, y = self.plot_host_usage()
      json.dump({
        'timestamps': x,
        'n_hosts': y
      }, f)

  def plot_host_usage(self, sample_size=100):
    counter = {}
    for h, durations in self.__hosts.items():
      for start, end in durations:
        start, end = floor(start, sample_size), ceil(end, sample_size)
        cur_end = min(start + sample_size, end)
        while cur_end < end:
          timerange = (cur_end - sample_size, cur_end)
          if h not in counter.get(timerange, set()):
            counter.setdefault(timerange, set()).add(h)
          cur_end += sample_size
    x = list(sorted(counter.keys()))
    y = [len(counter[k]) for k in x]
    return x, y

  def plot_resource_usage(self, resource, sample_size=100):
    counter, usage = {}, self.__usage[resource]
    for h, recs in usage.items():
      for timepoint, amt in recs:
        counter.setdefault(floor(timepoint, sample_size), {}).setdefault(h, []).append(amt)
    for timepoint, hosts in dict(counter).items():
      counter[timepoint] = np.mean([np.mean(vals) for vals in hosts.values()])
    x = list(sorted(counter.keys()))
    y = [counter[k] for k in x]
    return x, y

  def get_avg_host_usage(self, sample_size=100):
    _, n_hosts = self.plot_host_usage(sample_size=sample_size)
    return np.mean(n_hosts)

  def get_avg_cpu_usage(self, sample_size=100):
    _, n_cpus = self.plot_resource_usage('cpus', sample_size=sample_size)
    return np.mean(n_cpus)

  def get_avg_mem_usage(self, sample_size=100):
    _, mem = self.plot_resource_usage('mem', sample_size=sample_size)
    return np.mean(mem)

  def get_avg_disk_usage(self, sample_size=100):
    _, disks = self.plot_resource_usage('disk', sample_size=sample_size)
    return np.mean(disks)

  def get_avg_gpus_usage(self, sample_size=100):
    _, n_gpus = self.plot_resource_usage('gpus', sample_size=sample_size)
    return np.mean(n_gpus)

  def _track_resource_usage(self, h):
    assert isinstance(h, resources.Host)
    usage, now, resc = self.__usage, self.__env.now, h.resource
    usage['cpus'].setdefault(h, []).append((now, resc.cpus_used/resc.total_cpus))
    usage['mem'].setdefault(h, []).append((now, resc.mem_used/resc.total_mem))
    usage['disk'].setdefault(h, []).append((now, resc.disk_used/resc.total_disk))
    usage['gpus'].setdefault(h, []).append((now, resc.gpus_used/resc.total_gpus))






