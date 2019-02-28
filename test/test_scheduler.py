import simpy
import unittest
import numpy as np
import numpy.random as rnd

from application import Application, Container
from resources import Cluster, Host, Locality, Cloud, Region, Zone
from scheduler import GlobalSchedulerBase


class OpportunisticSchedulerTest(unittest.TestCase):

  def setUp(self):
    self.env = simpy.Environment()

  def test_one_app_wo_dep(self):
    """
    Test the scheduler with an application without dependencies.

    In this test, resources in the cluster are sufficient for running all containers in
    parallel.

    The test examines:

    1. Whether the application finishes end-to-end
    2. Whether all the containers are run and finished
    3. The maximum runtime differs from the simulator clock within a schedule interval (, which
       indicates all the containers are run in parallel)

    """
    from scheduler.opportunistic import OpportunisticGlobalScheduler
    env = self.env
    contrs = [Container(env, str(cid), 1, 1024, 1024, 1, rnd.uniform(2, 100)) for cid in range(16)]
    app = Application(env, 'test', contrs)
    dispatch_q, notify_q = simpy.Store(env), simpy.Store(env)
    hosts = [Host(env, str(i), 16, 1024 * 16, 1024 * 16, 16) for i in range(1)]
    cluster = Cluster(env, dispatch_q, notify_q, hosts)
    global_scheduler = OpportunisticGlobalScheduler(env, dispatch_q, notify_q, cluster)
    cluster.start()
    global_scheduler.start()
    global_scheduler.submit(app)
    local_scheduler = global_scheduler.get_scheduler(app.id)
    env.run()
    self.assertTrue(app.is_finished)
    self.assertTrue(all([c.is_finished for c in contrs]))
    self.assertAlmostEqual(env.now, max([c.runtime for c in contrs]),
                           delta=local_scheduler.interval)

  def test_one_app_w_dep(self):
    """
    Test the scheduler with an application with dependencies.

    In this test, resources in the cluster are sufficient for running all containers in
    parallel.

    The test examines:

    1. Whether the application finishes end-to-end
    2. Whether all the containers are run and finished
    3. The maximum runtime differs from the simulator clock within a schedule interval (, which
       indicates all the containers are run in parallel)

    """
    from scheduler.opportunistic import OpportunisticGlobalScheduler
    env = self.env
    contrs = [Container(env, str(cid), 1, 1024, 1024, 1, rnd.uniform(2, 100)) for cid in range(16)]
    app = Application(env, 'test', contrs)
    for c in app.containers:
      c.add_dependencies(*[str(i) for i in range(int(c.id))])
    dispatch_q, notify_q = simpy.Store(env), simpy.Store(env)
    hosts = [Host(env, str(i), 1, 1024, 1024, 1) for i in range(1)]
    cluster = Cluster(env, dispatch_q, notify_q, hosts)
    global_scheduler = OpportunisticGlobalScheduler(env, dispatch_q, notify_q, cluster)
    cluster.start()
    global_scheduler.start()
    global_scheduler.submit(app)
    local_scheduler = global_scheduler.get_scheduler(app.id)
    env.run()
    self.assertTrue(app.is_finished)
    self.assertTrue(all([c.is_finished for c in contrs]))
    self.assertAlmostEqual(env.now, sum([c.runtime for c in contrs]),
                           delta=(local_scheduler.interval - 1) * 16)