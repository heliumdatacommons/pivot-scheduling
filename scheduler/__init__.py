import simpy
import numpy as np
import numpy.random as rnd

import application
import resources

from collections import OrderedDict

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
    assert meter is None or isinstance(meter, Meter)
    self.__meter = meter
    self.__is_stopped = False

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

  def stop(self):
    self.__is_stopped = True

  def submit(self, app):
    """

    :param app: app.Application
    :param sched_class: local scheduler class

    """
    assert isinstance(app, application.Application)
    env, local_schedulers = self.__env, self.__local_schedulers
    if app.id in local_schedulers:
      self.logger.error('Application %s already exists'%app.id)
      return
    scheduler = LocalScheduler(env, app, self.__cluster, self.__submit_q)
    local_schedulers[app.id] = scheduler
    env.process(scheduler.start())

  def get_scheduler(self, app_id):
    return self.__local_schedulers.get(app_id)

  def schedule(self, tasks):
    raise NotImplemented

  def _update_resource_info(self):
    self.__resource_info = {h.id: np.array([h.resource.cpus_available, h.resource.mem_available,
                                            h.resource.disk_available, h.resource.gpus_available])
                            for h in self.cluster.hosts}

  def _dispatch(self):
    env, dispatch_q, interval = self.__env, self.__dispatch_q, self.__interval
    submit_q, wait_q = self.__submit_q, self.__wait_q
    meter, local_schedulers = self.__meter, self.__local_schedulers
    while not self.__is_stopped or any([not lc.application.is_finished for lc in local_schedulers.values()]):
      ready_q = []
      while wait_q:
        t, _ = wait_q.popitem()
        ready_q += t,
      n_items = len(submit_q.items)
      while len(ready_q) < n_items:
        t = yield submit_q.get()
        ready_q += t,
      self._update_resource_info()
      if meter:
        meter.increment_scheduling_ops(len(ready_q))
      for t in self.schedule(ready_q):
        if t.is_nascent:
          if t.placement is None:
            self.logger.debug('[%.3f] Task %s is put into the waiting queue' % (env.now, t.id))
            self.logger.debug('Demand: %.1f cpus, %.1f mem, %.1f disk, %.1f gpus' % (t.cpus, t.mem, t.disk, t.gpus))
            wait_q[t] = t
          else:
            self.logger.debug('[%.3f] Task %s is placed on host %s'%(env.now, t.id, t.placement))
            self.logger.debug('[%d] Dispatched task %s, runtime: %.3f'%(env.now, t.id, t.runtime))
            yield dispatch_q.put(t)
            t.set_submitted()
        else:
          self.logger.error('[%d] Task state is not nascent: %s'%(env.now, t.id))
      yield env.timeout(interval)
    if not self.__is_stopped:
      self.logger.error('Scheduler quit')

  def _listen(self):
    env, local_schedulers, submit_q = self.__env, self.__local_schedulers, self.__submit_q
    while not self.__is_stopped or any([not lc.application.is_finished for lc in local_schedulers.values()]):
      success, t = yield self.__notify_q.get()
      # self.logger.debug('[%.3f] Task %s finished: %s'%(self.__env.now, t.id, success))
      app = t.container.application
      if not app:
        self.logger.error('Application of task %s is not set'%t.id)
        continue
      local_sched = local_schedulers.get(app.id)
      if local_sched is None:
        self.logger.error('Application %s does not exist'%t.app.id)
        continue
      if success:
        t.set_finished()
        local_sched.notify(t)
      else:
        t.set_nascent()
        t.placement = None
        yield submit_q.put(t)
      if app.is_finished:
        app.end_time = self.__env.now
        start_time, end_time = app.start_time, app.end_time
        self.logger.debug('Application end time: %d'%end_time)
        self.logger.info('[%.3f] Application %s finished in %.3f seconds'%(env.now, app.id, end_time - start_time))
        local_schedulers.pop(app, None)
    if not self.__is_stopped:
      self.logger.error('Scheduler quit')


class LocalScheduler(Loggable):

  def __init__(self, env, app, cluster, submit_q, interval=5):
    assert isinstance(env, simpy.Environment)
    self.__env = env
    assert isinstance(app, application.Application)
    self.__application = app
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
  def application(self):
    return self.__application

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
    env, app = self.__env, self.application
    self.init()
    while not app.is_finished:
      while len(self.__ready_q) > 0:
        _, t = self.__ready_q.popitem()
        if t.is_nascent:
          self.logger.debug('[%d] Submit %s to the global scheduler'%(env.now, t.id))
          yield self.__submit_q.put(t)
      yield env.timeout(self.__interval)

  def update_resource_info(self):
    self.__resource_info = {h.id: np.array([h.resource.cpus_available, h.resource.mem_available,
                                            h.resource.disk_available,h.resource.gpus_available])
                            for h in self.cluster.hosts}

  def add_to_ready_queue(self, t):
    self.__ready_q[t] = t

  def init(self):
    app = self.application
    app.start_time = self.env.now
    for c in app.get_sources():
      for t in c.generate_tasks():
        self.add_to_ready_queue(t)

  def notify(self, t):
    assert isinstance(t, application.Task)
    app, env = self.application, self.env
    if t.is_finished:
      cur_c = t.container
      if cur_c.is_finished:
        successors = app.get_ready_successors(cur_c.id)
        for c in successors:
          for t in c.generate_tasks():
            self.add_to_ready_queue(t)
    else:
      self.add_to_ready_queue(t)








