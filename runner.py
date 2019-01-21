import json
import numpy as np

from simpy import Environment
from multiprocessing import Process

from resources.meter import Meter
from util import Loggable


class ExperimentRun(Process, Loggable):

  def __init__(self, label, cluster, apps, scheduler, data_dir, **sched_config):
    super(ExperimentRun, self).__init__()
    self.__label = label
    self.__env = env = Environment()
    self.__meter = meter = Meter(env)
    self.__cluster = cluster = cluster.clone(env, meter)
    self.__apps = [a.clone() for a in apps]
    self.__scheduler = scheduler(env, cluster, meter=meter, **sched_config)
    self.__data_dir = data_dir

  def run(self):
    label, data_dir, meter = self.__label, self.__data_dir, self.__meter
    env, apps, cluster, scheduler = self.__env, self.__apps, self.__cluster, self.__scheduler
    cluster.start()
    scheduler.start()
    self.logger.info('Testing %s'%label)
    for a in apps:
      scheduler.submit(a)
      self.logger.info('%s, ID: %s, # of containers: %d, '
                       'est. local runtime: %.2f, avg. data size: %.2fGB'%(label, a.id, len(a.containers),
                                                                           a.estimate_local_runtime(),
                                                                           a.avg_data_size/8000))
    env.run()
    avg_runtime = np.mean([a.end_time - a.start_time for a in apps])
    avg_local_runtime = np.mean([a.estimate_local_runtime() for a in apps])
    meter.save('%s/%s'%(data_dir, label))
    with open('%s/%s/general.json'%(data_dir, label), 'r+') as f:
      general = json.load(f)
      general.update(avg_runtime=avg_runtime, avg_local_runtime=avg_local_runtime)
      f.seek(0)
      json.dump(general, f)
    self.logger.info('Finish testing %s'%label)

