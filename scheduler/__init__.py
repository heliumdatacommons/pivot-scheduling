import simpy
import inspect
import numpy as np
import numpy.random as rnd

import appliance
import resource

from collections import Iterable, OrderedDict, defaultdict

from util import Loggable


class GlobalScheduler(Loggable):

  def __init__(self, env, dispatch_q, notify_q, cluster):
    assert isinstance(env, simpy.Environment)
    self.__env = env
    assert isinstance(dispatch_q, simpy.Store)
    self.__dispatch_q = dispatch_q
    assert isinstance(notify_q, simpy.Store)
    self.__notify_q = notify_q
    assert isinstance(cluster, resource.Cluster)
    self.__cluster = cluster
    self.__local_schedulers = {}

  def start(self):
    self.__env.process(self._listen())

  def submit(self, app, sched_class, **kwargs):
    """

    :param app: app.Appliance
    :param sched_class: local scheduler class

    """
    assert isinstance(app, appliance.Appliance)
    assert inspect.isclass(sched_class)
    if app.id in self.__local_schedulers:
      self.logger.error('Appliance %s already exists'%app.id)
      return
    for kw in ('env', 'dispatch_q', ):
      kwargs.pop(kw, None)
    scheduler = sched_class(self.__env, app, self.__dispatch_q, self.__cluster, **kwargs)
    self.__local_schedulers[app.id] = scheduler
    self.__env.process(scheduler.start())

  def get_scheduler(self, app_id):
    return self.__local_schedulers.get(app_id)

  def _listen(self):
    while True:
      success, c = yield self.__notify_q.get()
      self.logger.info('[%.3f] Container %s finished: %s'%(self.__env.now, c.id, success))
      if not c.appliance:
        self.logger.error('Appliance of container %s is not set'%c.id)
        continue
      local_sched = self.__local_schedulers.get(c.appliance.id)
      if not local_sched:
        self.logger.error('Appliance %s does not exist'%c.appliance.id)
        continue
      local_sched.notify(success, c)
      if c.appliance.is_finished:
        self.__local_schedulers.pop(c.appliance.id, None)


class LocalSchedulerBase(Loggable):

  def __init__(self, env, app, dispatch_q, cluster, schedule_interval=5, *args, **kwargs):
    assert isinstance(env, simpy.Environment)
    self.__env = env
    assert isinstance(app, appliance.Appliance)
    self.__appliance = app
    assert isinstance(dispatch_q, simpy.Store)
    self.__dispatch_q = dispatch_q
    assert isinstance(cluster, resource.Cluster)
    self.__cluster = cluster
    self.__resource_info = {}
    self.__schedule_interval = schedule_interval
    self.__ready_q = OrderedDict()
    self.__wait_q = OrderedDict()

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
  def schedule_interval(self):
    return self.__schedule_interval

  def start(self):
    env = self.__env
    self.init()
    while not self.appliance.is_finished:
      while len(self.__ready_q) > 0:
        _, c = self.__ready_q.popitem()
        if c.is_nascent:
          self.logger.info(
            '[%d] Dispatched container %s, runtime: %.3f'%(self.__env.now, c.id, c.runtime))
          yield self.__dispatch_q.put(c)
          c.set_submitted()
      yield env.timeout(self.__schedule_interval)

  def update_resource_info(self):
    self.__resource_info = {h.id: dict(cpus=h.resource.cpus_available,
                                       mem=h.resource.mem_available,
                                       disk=h.resource.disk_available,
                                       gpus=h.resource.gpus_available)
                            for h in self.cluster.hosts}

  def add_to_ready_queue(self, c):
    assert isinstance(c, appliance.Container)
    assert c.placement is not None
    self.__ready_q[c.id] = c

  def add_to_wait_queue(self, c):
    assert isinstance(c, appliance.Container)
    assert c.placement is None
    self.__wait_q[c.id] = c

  def clear_wait_queue(self):
    wait_q = list(self.__wait_q.values())
    self.__wait_q.clear()
    return wait_q

  def init(self):
    raise NotImplemented

  def notify(self, success, c):
    """

    :param success:
    :param c:
    :return:
    """
    assert isinstance(success, bool)
    assert isinstance(c, appliance.Container)

  def schedule(self, containers):
    """

    :param containers: a sequence of appliance.Container
    :return: a sequence of containers with placement
    """
    assert isinstance(containers, Iterable)
    assert all([isinstance(c, appliance.Container) for c in containers])

  def allocate_resource(self, contr, host):
    resc = self.resource_info
    hid = host.id
    contr.placement = hid
    resc[hid].update(cpus=resc[hid]['cpus'] - contr.cpus,
                     mem=resc[hid]['mem'] - contr.mem,
                     disk=resc[hid]['disk'] - contr.disk,
                     gpus=resc[hid]['gpus'] - contr.gpus)



class OpportunisticLocalScheduler(LocalSchedulerBase):

  def __init__(self, *args, **kwargs):
    super(OpportunisticLocalScheduler, self).__init__(*args, **kwargs)

  def init(self):
    self.schedule(self.appliance.get_sources())

  def notify(self, success, c):
    """

    :param success:
    :param c:
    :return:
    """
    assert isinstance(success, bool)
    assert isinstance(c, appliance.Container)
    app, env = self.appliance, self.env
    if not success:
      c.set_nascent()
      c.placement = None
    containers = self.clear_wait_queue() + (app.get_ready_successors(c.id) if success else [c])
    self.logger.info('[%.3f] local scheduler gets notified, next batch of containers: %s'%(env.now, containers))
    self.schedule(containers)

  def schedule(self, containers):
    """

    :param containers: a sequence of appliance.Container
    """
    super(OpportunisticLocalScheduler, self).schedule(containers)
    env = self.env
    self.update_resource_info()
    for c in containers:
      host = self._select_hosts(c)
      if host is None:
        h = self.cluster.hosts[0].resource
        self.logger.info('[%.3f] Container %s is put into the waiting queue'%(env.now, c.id))
        self.logger.info('Demand: %.1f cpus, %.1f mem, %.1f disk, %.1f gpus'%(c.cpus, c.mem, c.disk, c.gpus))
        self.logger.info('Available: %.1f cpus, %.1f mem, %.1f disk, %.1f gpus'%(h.cpus_available,
                                                                                 h.mem_available,
                                                                                 h.disk_available,
                                                                                 h.gpus_available))
        self.add_to_wait_queue(c)
      else:
        self.logger.info('[%.3f] Container %s is placed on host %s'%(env.now, c.id, host.id))
        self.allocate_resource(c, host)
        self.add_to_ready_queue(c)

  def _select_hosts(self, c):
    resc = self.resource_info
    qualified = [hid for hid, r in resc.items()
                 if r['cpus'] >= c.cpus and r['mem'] >= c.mem
                 and r['disk'] >= c.disk and r['gpus'] >= c.gpus]
    if not qualified:
      return None
    return self.cluster.get_host(rnd.choice(qualified))


class CostAwareLocalScheduler(LocalSchedulerBase):

  def __init__(self, *args, **kwargs):
    super(CostAwareLocalScheduler, self).__init__(*args, **kwargs)

  def init(self):
    self.schedule(self.appliance.get_sources())

  def notify(self, success, c):
    assert isinstance(success, bool)
    assert isinstance(c, appliance.Container)
    app, env = self.appliance, self.env
    if not success:
      c.set_nascent()
      c.placement = None
    containers = self.clear_wait_queue() + (app.get_ready_successors(c.id) if c.is_finished else [c])
    self.logger.info('[%.3f] local scheduler gets notified, next batch of containers: %s'%(env.now, containers))
    self.schedule(containers)

  def schedule(self, containers):
    self.update_resource_info()
    env = self.env
    meta = self.cluster.meta
    groups = self._group_containers(containers)
    for base, contrs in groups.items():
      if not base:
        base = rnd.choice(meta.zones)
      hosts = self._sort_hosts(self.cluster.hosts, base)
      contrs = self._sort_containers(contrs)
      for c, h in self._first_fit(hosts, contrs):
        if h is None:
          self.logger.info('[%.3f] Container %s is put into the waiting queue'%(env.now, c.id))
          self.logger.info('Demand: %.1f cpus, %.1f mem, %.1f disk, %.1f gpus'%(c.cpus, c.mem, c.disk, c.gpus))
          self.add_to_wait_queue(c)
        else:
          self.logger.info('[%.3f] Container %s is placed on host %s'%(env.now, c.id, h.id))
          self.allocate_resource(c, h)
          self.add_to_ready_queue(c)

  def _group_containers(self, contrs):
    cluster, app = self.cluster, self.appliance
    groups = defaultdict(list)
    for c in contrs:
      preds = app.get_predecessors(c.id)
      if preds:
        placement = rnd.choice(preds).placement
        zone = cluster.get_host(placement).locality
        groups[zone] += c,
      else:
        groups[None] += c,
    return groups

  def _sort_hosts(self, hosts, base):
    resc, meta = self.resource_info, self.cluster.meta

    def host_score_func(h):
      assert isinstance(h, resource.Host)
      r = self._calc_euclidean_dist(resc[h.id]['cpus'], resc[h.id]['mem'],
                                    resc[h.id]['disk'], resc[h.id]['gpus'])
      bw = meta.bw[(base, h.locality)] + meta.bw[(h.locality, base)]

      c = meta.cost[(base, h.locality)] + meta.cost[(h.locality, base)]
      return c/(r * bw)

    return sorted(hosts, key=host_score_func)

  def _sort_containers(self, contrs):

    def container_score_func(c):
      assert isinstance(c, appliance.Container)
      return -self._calc_euclidean_dist(c.cpus, c.mem, c.disk, c.gpus)

    return sorted(contrs, key=container_score_func)

  def _first_fit(self, hosts, contrs):
    resc = self.resource_info
    placements = []
    for c in contrs:
      placed = False
      for h in hosts:
        if resc[h.id]['cpus'] >= c.cpus \
            and resc[h.id]['mem'] >= c.mem \
            and resc[h.id]['disk'] >= c.disk \
            and resc[h.id]['gpus'] >= c.gpus:
          placements += (c, h),
          placed = True
          self._subtract_resource_usage(h.id, c.cpus, c.mem, c.disk, c.gpus)
          break
      if not placed:
        placements += (c, None),
    return placements

  def _best_fit(self, hosts, contrs):
    euclidean_dist = self._calc_euclidean_dist
    resc = self.resource_info
    placements = []
    for i, c in enumerate(contrs):
      min_cap, min_host = np.iinfo(np.int32).max, 0
      for j, h in enumerate(hosts):
        diff = euclidean_dist(resc[h.id]['cpus'], resc[h.id]['mem'],
                              resc[h.id]['disk'], resc[h.id]['gpus'],
                              c.cpus, c.mem, c.disk, c.gpus)
        if resc[h.id]['cpus'] >= c.cpus \
            and resc[h.id]['mem'] >= c.mem \
            and resc[h.id]['disk'] >= c.disk \
            and resc[h.id]['gpus'] >= c.gpus \
            and  diff < min_cap:
          min_cap, min_host = diff, h
      if min_cap == np.iinfo(np.int32).max:
        placements += (c, None),
      else:
        placements += (c, h)
        self._subtract_resource_usage(h.id, c.cpus, c.mem, c.disk, c.gpus)
    return placements

  def _calc_euclidean_dist(self, h_cpus, h_mem, h_disk, h_gpus,
                           c_cpus=0, c_mem=0, c_disk=0, c_gpus=0):
    return np.sqrt((h_cpus - c_cpus) ** 2
                   + (h_mem - c_mem) ** 2
                   + (h_disk - c_disk) ** 2
                   + (h_gpus - c_gpus) ** 2)

  def _subtract_resource_usage(self, hid, cpus, mem, disk, gpus):
    resc = self.resource_info
    resc[hid]['cpus'] -= cpus
    resc[hid]['mem'] -= mem
    resc[hid]['disk'] -= disk
    resc[hid]['gpus'] -= gpus






