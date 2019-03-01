import json
import yaml
import numpy as np

from simpy import Environment
from multiprocessing import Process

from application import Application, Container
from resources.meter import Meter
from util import Loggable


class ExperimentRun(Process, Loggable):

  def __init__(self, label, cluster, scheduler, load_f, data_dir, output_size_scale_factor,
               n_apps=None, **sched_config):
    super(ExperimentRun, self).__init__()
    self.__label = label
    self.__cluster = cluster
    self.__scheduler = scheduler
    self.__load_f = load_f
    self.__output_size_scale_factor = output_size_scale_factor
    self.__n_apps = n_apps
    self.__sched_config = sched_config
    self.__data_dir = data_dir

  def run(self):
    data_dir = self.__data_dir
    label, cluster, load_f, n_apps = self.__label, self.__cluster, self.__load_f, self.__n_apps
    output_size_scale_factor = self.__output_size_scale_factor
    scheduler, sched_config = self.__scheduler, self.__sched_config
    env = Environment()
    meter = Meter(env)
    cluster = cluster.clone(env, meter)
    scheduler = scheduler(env, cluster, meter=meter, **sched_config)
    load_gen = TraceBasedApplicationGenerator(env, load_f, scheduler, output_size_scale_factor, n_apps)

    cluster.start()
    scheduler.start()
    load_gen.start()

    self.logger.info('Testing %s'%label)

    env.run()
    avg_runtime = np.mean([a.end_time - a.start_time for a in load_gen.apps])
    meter.save('%s/%s'%(data_dir, label))
    with open('%s/%s/general.json'%(data_dir, label), 'r+') as f:
      general = json.load(f)
      general.update(avg_runtime=avg_runtime)
      f.seek(0)
      json.dump(general, f)
    self.logger.info('Finish testing %s'%label)


class TraceBasedApplicationGenerator(Loggable):
  """
  Generate batch jobs as applications runnable on PIVOT from the
  sampled data of the Alibaba cluster trace.

  `MEM_SCALE_FACTOR`: The CPU values in the trace are absolute while the memory
  ones are normalized to 100. Considering every machine in the trace consistently
  has 96 cores, we assume it equivalent to the `r5d.24xlarge` machine in AWS,
  which has 96 cores and 768GB. Therefore, we scale the memory demands of the
  tasks by the `MEM_SCALE_FACTOR` to make them absolute values (in GBs), so that
  they can be tested on machines with varying resource configurations.

  """

  MEM_SCALE_FACTOR = 7.68 * 1024

  def __init__(self, env, trace_f, scheduler, output_size_scale_factor, n_apps=None):
    self.__env = env
    self.__apps = []
    self.__scheduler = scheduler
    self.__output_size_scale_factor = output_size_scale_factor
    self.__n_apps = n_apps
    self._load_data(trace_f)

  @property
  def apps(self):
    return [a for _, apps in self.__apps for a in apps]

  def start(self):
    self.__env.process(self._submit_applications())

  def _load_data(self, trace_f):
    env = self.__env
    with open(trace_f) as f:
      jobs = yaml.load(f)
      # self.logger.info('Processing job %s'%j['id'])
      for j in jobs:
        contrs = []
        for t in j['tasks']:
          task_id, runtime, n_inst = str(t['id']), t['runtime'], t['n_instances']
          cpus, mem = t['cpus'], t['mem']
          deps = [str(d) for d in t['dependencies']]
          contrs += Container(env, task_id, cpus=cpus,
                              mem=mem * self.MEM_SCALE_FACTOR,
                              output_size=mem * self.__output_size_scale_factor,
                              runtime=runtime, instances=n_inst, dependencies=deps),
        app = Application(env, j['id'], contrs)
        self._bin_insert(j['submit_time'], app)

  def _submit_applications(self):
    env, scheduler, n_apps = self.__env, self.__scheduler, self.__n_apps
    last_ts, counter = None, 0
    for ts, apps in self.__apps:
      if last_ts:
        yield env.timeout(ts - last_ts)
      for a in apps:
        self.logger.info('[%.3f] Application %s submitted'%(env.now, a.id))
        scheduler.submit(a)
        counter += 1
        if n_apps and counter == n_apps:
          break
      if n_apps and counter == n_apps:
        break
      last_ts = ts
    scheduler.stop()

  def _bin_insert(self, ts, app):
    apps = self.__apps
    if len(apps) == 0:
      apps += (ts, [app]),
      return
    lo, hi = 0, len(apps) - 1
    while lo <= hi:
      mid = (lo + hi) // 2
      if apps[mid][0] > ts:
        hi = mid - 1
      elif apps[mid][0] < ts:
        lo = mid + 1
      else:
        apps[mid][1].append(app),
        return
    apps.insert(lo, (ts, [app]))
