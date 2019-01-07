import simpy
import unittest
import numpy as np
import numpy.random as rnd

from appliance import Appliance, Container
from resource import Cluster, Host, Locality, Cloud, Region, Zone
from scheduler import GlobalScheduler


class OpportunisticSchedulerTest(unittest.TestCase):

  def setUp(self):
    self.env = simpy.Environment()

  def test_one_app_wo_dep(self):
    """
    Test the scheduler with an appliance without dependencies.

    In this test, resources in the cluster are sufficient for running all containers in
    parallel.

    The test examines:

    1. Whether the appliance finishes end-to-end
    2. Whether all the containers are run and finished
    3. The maximum runtime differs from the simulator clock within a schedule interval (, which
       indicates all the containers are run in parallel)

    """
    from scheduler import OpportunisticLocalScheduler
    env = self.env
    contrs = [Container(env, str(cid), 1, 1024, 1024, 1, rnd.uniform(2, 100)) for cid in range(16)]
    app = Appliance(env, 'test', contrs)
    dispatch_q, notify_q = simpy.Store(env), simpy.Store(env)
    hosts = [Host(env, str(i), 16, 1024 * 16, 1024 * 16, 16) for i in range(1)]
    cluster = Cluster(env, dispatch_q, notify_q, hosts)
    global_scheduler = GlobalScheduler(env, dispatch_q, notify_q, hosts)
    cluster.start()
    global_scheduler.start()
    global_scheduler.submit(app, OpportunisticLocalScheduler)
    local_scheduler = global_scheduler.get_scheduler(app.id)
    env.run()
    self.assertTrue(app.is_finished)
    self.assertTrue(all([c.is_finished for c in contrs]))
    self.assertAlmostEqual(env.now, max([c.runtime for c in contrs]),
                           delta=local_scheduler.schedule_interval)

  def test_one_app_w_dep(self):
    """
    Test the scheduler with an appliance with dependencies.

    In this test, resources in the cluster are sufficient for running all containers in
    parallel.

    The test examines:

    1. Whether the appliance finishes end-to-end
    2. Whether all the containers are run and finished
    3. The maximum runtime differs from the simulator clock within a schedule interval (, which
       indicates all the containers are run in parallel)

    """
    from scheduler import OpportunisticLocalScheduler
    env = self.env
    contrs = [Container(env, str(cid), 1, 1024, 1024, 1, rnd.uniform(2, 100)) for cid in range(16)]
    app = Appliance(env, 'test', contrs)
    for c in app.containers:
      c.add_dependencies(*[str(i) for i in range(int(c.id))])
    dispatch_q, notify_q = simpy.Store(env), simpy.Store(env)
    hosts = [Host(env, str(i), 1, 1024, 1024, 1) for i in range(1)]
    cluster = Cluster(env, dispatch_q, notify_q, hosts)
    global_scheduler = GlobalScheduler(env, dispatch_q, notify_q, hosts)
    cluster.start()
    global_scheduler.start()
    global_scheduler.submit(app, OpportunisticLocalScheduler)
    local_scheduler = global_scheduler.get_scheduler(app.id)
    env.run()
    self.assertTrue(app.is_finished)
    self.assertTrue(all([c.is_finished for c in contrs]))
    self.assertAlmostEqual(env.now, sum([c.runtime for c in contrs]),
                           delta=(local_scheduler.schedule_interval - 1) * 16)

  def test_insufficient_resource_one_app(self):
    """
    Test the scheduler with insufficient resources in the cluster.
    (resource_demand = 2 * resource_available)

    The test examines:

    1. Whether the appliance finishes end-to-end
    2. Whether all the containers in the appliance are run and finished eventually
    3. Whether the containers are run and finished in time
       (max(runtime_c) <= runtime <= 2 * max(runtime_c))

    """
    from scheduler import OpportunisticLocalScheduler
    env = self.env
    contrs = [Container(env, str(cid), 1, 1024, 1024, 1, rnd.uniform(2, 100)) for cid in range(16)]
    app = Appliance(env, 'test', contrs)
    dispatch_q, notify_q = simpy.Store(env), simpy.Store(env)
    hosts = [Host(env, str(i), 2, 1024 * 2, 1024 * 2, 2) for i in range(4)]
    cluster = Cluster(env, dispatch_q, notify_q, hosts)
    scheduler = GlobalScheduler(env, dispatch_q, notify_q, hosts)
    cluster.start()
    scheduler.start()
    scheduler.submit(app, OpportunisticLocalScheduler)
    env.run()
    self.assertTrue(app.is_finished)
    self.assertTrue(all([c.is_finished for c in contrs]))
    self.assertGreaterEqual(env.now, max([c.runtime for c in contrs]))
    self.assertLessEqual(env.now, 2 * max([c.runtime for c in contrs]))