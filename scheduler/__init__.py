import simpy
import numpy as np
import numpy.random as rnd

import appliance
import resources

from collections import Iterable, OrderedDict, defaultdict

from resources.meter import Meter
from util import Loggable


class GlobalSchedulerBase(Loggable):

  def __init__(self, env, cluster, interval=5, seed=None, meter=None, *args, **kwargs):
    assert isinstance(env, simpy.Environment)
    self.__env = env
    assert isinstance(cluster.dispatch_q, simpy.Store)
    self.__dispatch_q = cluster.dispatch_q
    assert isinstance(cluster.notify_q, simpy.Store)
    self.__notify_q = cluster.notify_q
    assert isinstance(cluster, resources.Cluster)
    self.__cluster = cluster
    assert isinstance(interval, int)
    self.__interval = interval
    self.__local_schedulers = {}
    self.__resource_info = {}
    self.__submit_q = simpy.Store(env)
    self.__wait_q = OrderedDict()
    self.__randomizer = rnd.RandomState(seed)
    assert isinstance(meter, Meter)
    self.__meter = meter

  @property
  def env(self):
    return self.__env

  @property
  def cluster(self):
    return self.__cluster

  @property
  def resource_info(self):
    return dict(self.__resource_info)

  @property
  def randomizer(self):
    return self.__randomizer

  def start(self):
    env = self.__env
    env.process(self._dispatch())
    env.process(self._listen())

  def submit(self, app):
    """

    :param app: app.Appliance
    :param sched_class: local scheduler class

    """
    assert isinstance(app, appliance.Appliance)
    if app.id in self.__local_schedulers:
      self.logger.error('Appliance %s already exists'%app.id)
      return
    scheduler = LocalScheduler(self.__env, app, self.__cluster, self.__submit_q)
    self.__local_schedulers[app.id] = scheduler
    self.__env.process(scheduler.start())

  def get_scheduler(self, app_id):
    return self.__local_schedulers.get(app_id)

  def schedule(self, containers):
    raise NotImplemented

  def _update_resource_info(self):
    self.__resource_info = {h.id: dict(cpus=h.resource.cpus_available,
                                       mem=h.resource.mem_available,
                                       disk=h.resource.disk_available,
                                       gpus=h.resource.gpus_available)
                            for h in self.cluster.hosts}

  def _dispatch(self):
    env, dispatch_q, interval = self.__env, self.__dispatch_q, self.__interval
    submit_q, wait_q = self.__submit_q, self.__wait_q
    meter = self.__meter
    while any([not lc.appliance.is_finished for lc in self.__local_schedulers.values()]):
      ready_q = []
      while wait_q:
        c, _ = wait_q.popitem()
        ready_q += c,
      n_items = len(submit_q.items)
      while len(ready_q) < n_items:
        c = yield submit_q.get()
        ready_q += c,
      self._update_resource_info()
      if meter:
        meter.increment_scheduling_ops(len(ready_q))
      for c in self.schedule(ready_q):
        if c.is_nascent:
          if c.placement is None:
            self.logger.debug('[%.3f] Container %s is put into the waiting queue' % (env.now, c.id))
            self.logger.debug('Demand: %.1f cpus, %.1f mem, %.1f disk, %.1f gpus' % (c.cpus, c.mem, c.disk, c.gpus))
            wait_q[c] = c
            if meter:
              meter.add_scheduling_turnover(env.now)
          else:
            self.logger.debug('[%.3f] Container %s is placed on host %s'%(env.now, c.id, c.placement))
            self.logger.debug('[%d] Dispatched container %s, runtime: %.3f'%(env.now, c.id, c.runtime))
            yield dispatch_q.put(c)
            c.set_submitted()
        else:
          self.logger.error('[%d] Container state is not nascent: %s'%(env.now, c.id))
      yield env.timeout(interval)

  def _listen(self):
    local_schedulers = self.__local_schedulers
    while any([not lc.appliance.is_finished for lc in self.__local_schedulers.values()]):
      success, c = yield self.__notify_q.get()
      self.logger.debug('[%.3f] Container %s finished: %s'%(self.__env.now, c.id, success))
      if not c.appliance:
        self.logger.error('Appliance of container %s is not set'%c.id)
        continue
      local_sched = local_schedulers.get(c.appliance.id)
      if local_sched is None:
        self.logger.error('Appliance %s does not exist'%c.appliance.id)
        continue
      if success:
        c.set_finished()
        local_sched.notify(c)
      else:
        c.set_nascent()
        c.placement = None
        yield self.__submit_q.put(c)
      if c.appliance.is_finished:
        app = c.appliance
        app.end_time = self.__env.now
        self.logger.debug('Appliance end time: %d'%app.end_time)
        self.logger.info('Appliance %s finished in %.3f seconds'%(app.id, app.end_time - app.start_time))
        local_schedulers.pop(app, None)


class LocalScheduler(Loggable):

  def __init__(self, env, app, cluster, submit_q, interval=5):
    assert isinstance(env, simpy.Environment)
    self.__env = env
    assert isinstance(app, appliance.Appliance)
    self.__appliance = app
    assert isinstance(cluster, resources.Cluster)
    self.__cluster = cluster
    assert isinstance(submit_q, simpy.Store)
    self.__submit_q = submit_q
    self.__resource_info = {}
    self.__interval = interval
    self.__ready_q = OrderedDict()

  @property
  def env(self):
    return self.__env

  @property
  def appliance(self):
    return self.__appliance

  @property
  def cluster(self):
    return self.__cluster

  @property
  def resource_info(self):
    return dict(self.__resource_info)

  @property
  def interval(self):
    return self.__interval

  def start(self):
    env, app = self.__env, self.appliance
    self.init()
    while not app.is_finished:
      while len(self.__ready_q) > 0:
        _, c = self.__ready_q.popitem()
        if c.is_nascent:
          # self.logger.debug('[%d] Submit %s to the global scheduler'%(env.now, c.id))
          yield self.__submit_q.put(c)
      yield env.timeout(self.__interval)

  def update_resource_info(self):
    self.__resource_info = {h.id: dict(cpus=h.resource.cpus_available,
                                       mem=h.resource.mem_available,
                                       disk=h.resource.disk_available,
                                       gpus=h.resource.gpus_available)
                            for h in self.cluster.hosts}

  def add_to_ready_queue(self, c):
    assert isinstance(c, appliance.Container)
    self.__ready_q[c] = c

  def init(self):
    app = self.appliance
    app.start_time = self.env.now
    self.logger.debug('Appliance start time: %d'%app.start_time)
    for c in app.get_sources():
      self.add_to_ready_queue(c)

  def notify(self, c):
    assert isinstance(c, appliance.Container)
    app, env = self.appliance, self.env
    containers = app.get_ready_successors(c.id) if c.is_finished else [c]
    self.logger.debug('[%.3f] local scheduler gets notified, next batch of containers: %s'%(env.now, containers))
    for c in containers:
      self.add_to_ready_queue(c)


class OpportunisticGlobalScheduler(GlobalSchedulerBase):

  def __init__(self, *args, **kwargs):
    super(OpportunisticGlobalScheduler, self).__init__(*args, **kwargs)

  def schedule(self, containers):
    assert isinstance(containers, Iterable) and all([isinstance(c, appliance.Container) for c in containers])
    resc = self.resource_info
    for c in containers:
      qualified = [hid for hid, r in resc.items()
                   if r['cpus'] >= c.cpus and r['mem'] >= c.mem and r['disk'] >= c.disk and r['gpus'] >= c.gpus]
      if len(qualified) > 0:
        h = self.cluster.get_host(self.randomizer.choice(qualified))
        r = resc[h.id]
        c.placement = h.id
        r['cpus'] -= c.cpus
        r['mem'] -= c.mem
        r['disk'] -= c.disk
        r['gpus'] -= c.gpus
    return list(containers)


class FirstFitGlobalScheduler(GlobalSchedulerBase):

  def __init__(self, *args, **kwargs):
    decreasing = str(kwargs.pop('decreasing', False))
    super(FirstFitGlobalScheduler, self).__init__(*args, **kwargs)
    self.__decreasing = decreasing

  def schedule(self, containers):
    env, hosts = self.env, self.cluster.hosts
    resc = self.resource_info
    if self.__decreasing:
      containers = self._sort_containers(containers)
    for c in containers:
      for h in hosts:
        r = resc[h.id]
        if r['cpus'] >= c.cpus and r['mem'] >= c.mem and r['disk'] >= c.disk and r['gpus'] >= c.gpus:
          c.placement = h.id
          r['cpus'] -= c.cpus
          r['mem'] -= c.mem
          r['disk'] -= c.disk
          r['gpus'] -= c.gpus
          break
    return containers

  def _sort_containers(self, contrs):
    def container_score_func(c):
      assert isinstance(c, appliance.Container)
      return -self._calc_euclidean_dist(c.cpus, c.mem, c.disk, c.gpus)

    return sorted(contrs, key=container_score_func)

  def _calc_euclidean_dist(self, h_cpus, h_mem, h_disk, h_gpus,
                           c_cpus=0, c_mem=0, c_disk=0, c_gpus=0):
    return np.sqrt((h_cpus - c_cpus) ** 2 + (h_mem - c_mem) ** 2 + (h_disk - c_disk) ** 2 + (h_gpus - c_gpus) ** 2)


class BestFitGlobalScheduler(GlobalSchedulerBase):

  def __init__(self, *args, **kwargs):
    decreasing = str(kwargs.pop('decreasing', False))
    super(BestFitGlobalScheduler, self).__init__(*args, **kwargs)
    self.__decreasing = decreasing

  def schedule(self, containers):
    resc, calc_dist = self.resource_info, self._calc_euclidean_dist
    if self.__decreasing:
      containers = self._sort_containers(containers)
    for c in containers:
      qualified = [(hid, r) for hid, r in resc.items()
                   if r['cpus'] >= c.cpus and r['mem'] >= c.mem and r['disk'] >= c.disk and r['gpus'] >= c.gpus]
      if qualified:
        _, hid, r = min([(calc_dist(r['cpus'], r['mem'], r['disk'], r['gpus'], c.cpus, c.mem, c.disk, c.gpus),
                          hid, r) for hid, r in qualified])
        c.placement = hid
        r['cpus'] -= c.cpus
        r['mem'] -= c.mem
        r['disk'] -= c.disk
        r['gpus'] -= c.gpus
    return containers

  def _sort_containers(self, contrs):
    def container_score_func(c):
      assert isinstance(c, appliance.Container)
      return -self._calc_euclidean_dist(c.cpus, c.mem, c.disk, c.gpus)

    return sorted(contrs, key=container_score_func)

  def _calc_euclidean_dist(self, h_cpus, h_mem, h_disk, h_gpus,
                           c_cpus=0, c_mem=0, c_disk=0, c_gpus=0):
    return np.sqrt((h_cpus - c_cpus) ** 2 + (h_mem - c_mem) ** 2 + (h_disk - c_disk) ** 2 + (h_gpus - c_gpus) ** 2)


class CostAwareGlobalScheduler(GlobalSchedulerBase):

  def __init__(self, *args, **kwargs):
    bin_pack_algo = str(kwargs.pop('bin_pack_algo', 'first-fit'))
    sort_containers = bool(kwargs.pop('sort_containers', False))
    sort_hosts = bool(kwargs.pop('sort_hosts', False))
    realtime_bw = kwargs.pop('realtime_bw',False)
    host_decay = kwargs.pop('host_decay', False)
    super(CostAwareGlobalScheduler, self).__init__(*args, **kwargs)
    self.__bin_pack_algo = bin_pack_algo
    self.__sort_containers = sort_containers
    self.__sort_hosts = sort_hosts
    self.__realtime_bw = realtime_bw
    self.__host_decay = host_decay
    self.__last_locality = None

  def schedule(self, containers):
    bin_pack_algo = self.__bin_pack_algo
    if bin_pack_algo == 'first-fit':
      bin_pack_algo = self._first_fit
    elif bin_pack_algo == 'best-fit':
      bin_pack_algo = self._best_fit
    env, storage, hosts = self.env, self.cluster.storage, self.cluster.hosts
    groups = self._group_containers(containers)
    for anchor, contrs in groups.items():
      if isinstance(anchor, appliance.Appliance):
        anchor = self.randomizer.choice(storage)
      if self.__sort_containers:
        contrs = self._sort_containers(contrs)
      bin_pack_algo(hosts, contrs, anchor)
    return containers

  def _group_containers(self, contrs):
    cluster = self.cluster
    groups = defaultdict(list)
    for c in contrs:
      app = c.appliance
      preds = c.appliance.get_predecessors(c.id)
      if preds:
        placement = self.randomizer.choice(preds).placement
        locality = cluster.get_host(placement).locality
        data_src = cluster.get_storage_by_locality(locality)
        groups[data_src] += c,
      else:
        groups[app] += c,
    return groups

  def _sort_containers(self, contrs):

    def container_score_func(c):
      assert isinstance(c, appliance.Container)
      return -self._calc_euclidean_dist(c.cpus, c.mem, c.disk, c.gpus)

    return sorted(contrs, key=container_score_func)

  def _best_fit(self, hosts, contrs, anchor):
    env, resc, cluster, meta = self.env, self.resource_info, self.cluster, self.cluster.meta
    host_decay, rt_bw = self.__host_decay, self.__realtime_bw

    def host_score_func(item):
      h, c = item
      assert isinstance(h, resources.Host)
      assert isinstance(c, appliance.Container)
      r = self._calc_euclidean_dist(resc[h.id]['cpus'], resc[h.id]['mem'], resc[h.id]['disk'], resc[h.id]['gpus'],
                                    c.cpus, c.mem, c.disk, c.gpus)
      in_route = cluster.get_route(anchor.id, h.id)
      out_route = cluster.get_route(h.id, anchor.id)
      if round(in_route.bw, 3) != round(in_route.realtime_bw, 3):
        self.logger.debug('bw: %.3f, rt bw: %.3f'%(in_route.bw, in_route.realtime_bw))
      if round(out_route.bw, 3) != round(out_route.realtime_bw, 3):
        self.logger.debug('bw: %.3f, rt bw: %.3f'%(out_route.bw, out_route.realtime_bw))
      bw = (in_route.realtime_bw + out_route.realtime_bw) if rt_bw else (in_route.bw + out_route.bw)
      c = meta.cost[(anchor.locality, h.locality)] + meta.cost[(h.locality, anchor.locality)]
      decay = max(len(h.containers) if host_decay else 0, 1)

      return c * r * decay/bw

    for c in contrs:
      candidates = [(h, c) for h in hosts
                    if resc[h.id]['cpus'] >= c.cpus
                    and resc[h.id]['mem'] >= c.mem
                    and resc[h.id]['disk'] >= c.disk
                    and resc[h.id]['gpus'] >= c.gpus]
      if len(candidates) == 0:
        self.logger.debug('[%.3f] Container %s is put into the waiting queue' % (env.now, c.id))
        self.logger.debug('Demand: %.1f cpus, %.1f mem, %.1f disk, %.1f gpus' % (c.cpus, c.mem, c.disk, c.gpus))
      else:
        h, _ = min(candidates, key=host_score_func)
        self.logger.debug('[%.3f] Container %s is placed on host %s' % (env.now, c.id, h.id))
        c.placement = h.id
        resc[h.id]['cpus'] -= c.cpus
        resc[h.id]['mem'] -= c.mem
        resc[h.id]['disk'] -= c.disk
        resc[h.id]['gpus'] -= c.gpus

  def _first_fit(self, hosts, contrs, data_src):
    env, resc, cluster, meta = self.env, self.resource_info, self.cluster, self.cluster.meta
    sort_hosts, rt_bw = self.__sort_hosts, self.__realtime_bw
    host_decay = self.__host_decay

    def host_score_func(h):
      assert isinstance(h, resources.Host)
      h_resc = resc[h.id]
      r = self._calc_euclidean_dist(h_resc['cpus'], h_resc['mem'], h_resc['disk'], h_resc['gpus'])
      in_route = cluster.get_route(data_src.id, h.id)
      out_route = cluster.get_route(h.id, data_src.id)
      if round(in_route.bw, 3) != round(in_route.realtime_bw, 3):
        self.logger.debug('bw: %.3f, rt bw: %.3f'%(in_route.bw, in_route.realtime_bw))
      if round(out_route.bw, 3) != round(out_route.realtime_bw, 3):
        self.logger.debug('bw: %.3f, rt bw: %.3f'%(out_route.bw, out_route.realtime_bw))
      bw = (in_route.realtime_bw + out_route.realtime_bw) if rt_bw else (in_route.bw + out_route.bw)

      c = meta.cost[(data_src.locality, h.locality)] + meta.cost[(h.locality, data_src.locality)]
      df = max(len(h.containers) if host_decay else 0, 1)
      return c * df/(r * bw)

    if sort_hosts:
      hosts = sorted(hosts, key=host_score_func)
    for c in contrs:
      for i, h in enumerate(hosts):
        r = resc[h.id]
        if r['cpus'] >= c.cpus and r['mem'] >= c.mem and r['disk'] >= c.disk and r['gpus'] >= c.gpus:
          c.placement = h.id
          r['cpus'] -= c.cpus
          r['mem'] -= c.mem
          r['disk'] -= c.disk
          r['gpus'] -= c.gpus
          break

  def _calc_euclidean_dist(self, h_cpus, h_mem, h_disk, h_gpus,
                           c_cpus=0, c_mem=0, c_disk=0, c_gpus=0):
    return np.sqrt((h_cpus - c_cpus) ** 2 + (h_mem - c_mem) ** 2 + (h_disk - c_disk) ** 2 + (h_gpus - c_gpus) ** 2)






