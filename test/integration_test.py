import simpy
import unittest
import numpy.random as rnd

from appliance import Appliance, Container
from appliance.gen import DataParallelApplianceGenerator
from resource import Cluster, Host
from resource.gen import RandomClusterGenerator
from scheduler import GlobalScheduler


class OpportunisticSchedulerTest(unittest.TestCase):

  def setUp(self):
    rnd.seed(123)
    env = simpy.Environment()
    dispatch_q, notify_q = simpy.Store(env), simpy.Store(env)
    self.cluster = RandomClusterGenerator(env, dispatch_q, notify_q,
                                          cpus_lo=16, cpus_hi=16,
                                          mem_lo=1024 * 20, mem_hi=1024 * 20,
                                          disk_lo=1024 * 20, disk_hi=1024 * 20,
                                          gpus_lo=16, gpus_hi=16, seed=123).generate(100)
    self.env, self.dispatch_q, self.notify_q = env, dispatch_q, notify_q

  def test_one_app(self):
    """


    """
    from scheduler import OpportunisticLocalScheduler, CostAwareLocalScheduler
    env, cluster = self.env, self.cluster
    app = DataParallelApplianceGenerator(env, min_cpus=1, max_cpus=4,
                                         min_mem=1024, max_mem=1024 * 4,
                                         min_disk=1024, max_disk=1024 * 4,
                                         min_gpus=0, max_gpus=1,
                                         min_runtime=60, max_runtime=60 * 6,
                                         min_output_nbytes=10 ** 4, max_output_nbytes=10 ** 5,
                                         min_seq_steps=0, max_seq_steps=1,
                                         min_parallel_steps=1, max_parallel_steps=2,
                                         min_parallel_level=100, max_parallel_level=101,
                                         seed=123).generate()
    dispatch_q, notify_q = self.dispatch_q, self.notify_q
    global_scheduler = GlobalScheduler(env, dispatch_q, notify_q, cluster)
    cluster.start()
    global_scheduler.start()
    print(set([h.locality for h in cluster.hosts]))
    # global_scheduler.submit(app, CostAwareLocalScheduler)
    global_scheduler.submit(app, OpportunisticLocalScheduler)
    local_scheduler = global_scheduler.get_scheduler(app.id)
    env.run()
    self.assertTrue(app.is_finished)
    # self.assertTrue(all([c.is_finished for c in app.containers]))
    # self.assertAlmostEqual(env.now, max([c.runtime for c in app.containers]),
    #                        delta=local_scheduler.schedule_interval)
