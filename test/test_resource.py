import simpy
import unittest
import numpy as np

from collections import defaultdict, Counter

from appliance import Container
from resources import Cluster, Host
from resources.network import Packet, NetworkRoute
from util import Loggable


class MockScheduler(Loggable):

  def __init__(self, env, dispatch_q, notify_q):
    self.__env = env
    self.__dispatch_q = dispatch_q
    self.__notify_q = notify_q

  def start(self):
    self.__env.process(self._submit())
    self.__env.process(self._listen())

  def _submit(self):
    counter = 0
    rnd = np.random
    while True:
      c = Container(self.__env, str(counter), rnd.randint(1, 4), rnd.randint(1024, 4096),
                    rnd.randint(1024, 10240), rnd.randint(1, 4), rnd.randint(10, 20),
                    placement=str(rnd.randint(0, 10)))
      yield self.__dispatch_q.put(c)
      self.logger.info('[%d] Submitted container %s: cpus: %.1f, mem: %d, disk: %d, gpus: %d, '
                       'runtime: %d, placement: %s'%(self.__env.now, c.id, c.cpus, c.mem, c.disk,
                                                     c.gpus, c.runtime, c.placement))
      yield self.__env.timeout(rnd.randint(1, 20))
      counter += 1

  def _listen(self):
    while True:
      success, c = yield self.__notify_q.get()
      self.logger.info('Success in executing container %s on %s: %s'%(c.id, c.placement, success))


class TestResource(unittest.TestCase):

  def setUp(self):
    env = simpy.Environment()
    dispatch_q, notify_q = simpy.Store(env), simpy.Store(env)
    rnd = np.random
    hosts = [Host(env, str(i), rnd.randint(1, 10), rnd.randint(4096, 10240),
                  rnd.randint(10240, 102400), rnd.randint(1, 10))
             for i in range(10)]
    self.env = env
    self.cluster = Cluster(env, dispatch_q, notify_q, hosts)
    self.scheduler = MockScheduler(env, dispatch_q, notify_q)

  def test_run(self):
    self.cluster.start()
    self.scheduler.start()
    self.env.run(until=1000)


class TestRandomClusterGenerator(unittest.TestCase):

  def setUp(self):
    env = simpy.Environment()
    dispatch_q, notify_q = simpy.Store(env), simpy.Store(env)
    from resources.gen import RandomClusterGenerator
    self.gen = RandomClusterGenerator(env, dispatch_q, notify_q,
                                      cpus_lo=1, cpus_hi=16,
                                      mem_lo=1024 * 4, mem_hi=1024 * 16,
                                      disk_lo=1024 * 10, disk_hi=1024 * 100,
                                      gpus_lo=1, gpus_hi=4)

  def test_generating_uniform_hosts(self):
    n_hosts = 100
    cluster = self.gen.generate(n_hosts, uniform=True)
    self.assertEqual(len(set(h.resource.total_cpus for h in cluster.hosts)), 1)
    self.assertEqual(len(set(h.resource.total_mem for h in cluster.hosts)), 1)
    self.assertEqual(len(set(h.resource.total_disk for h in cluster.hosts)), 1)
    self.assertEqual(len(set(h.resource.total_gpus for h in cluster.hosts)), 1)

  def test_generating_nonuniform_hosts(self):
    n_hosts = 100
    cluster = self.gen.generate(n_hosts, uniform=False)
    self.assertGreater(len(set(h.resource.total_cpus for h in cluster.hosts)), 1)
    self.assertGreater(len(set(h.resource.total_mem for h in cluster.hosts)), 1)
    self.assertGreater(len(set(h.resource.total_disk for h in cluster.hosts)), 1)
    self.assertGreater(len(set(h.resource.total_gpus for h in cluster.hosts)), 1)

  def test_host_distribution(self):
    n_hosts = 100
    cluster = self.gen.generate(n_hosts)
    host_dist = Counter([h.locality for h in cluster.hosts]).items()
    self.assertAlmostEqual(np.mean([c for _, c in host_dist]), n_hosts/len(self.gen.LOCALITY),
                           delta=.5)


class NetworkRouteTest(unittest.TestCase):

  def setUp(self):
    self.env = simpy.Environment()

  def test_send_one_packet(self):
    env = self.env
    src = Host()
    route = NetworkRoute()