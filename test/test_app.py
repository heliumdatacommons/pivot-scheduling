import simpy
import unittest
import networkx as nx
import numpy.random as rnd

from application import Container, Application


class ApplicationTest(unittest.TestCase):

  def setUp(self):
    self.env = simpy.Environment()
    env = self.env
    contrs = [Container(env, str(cid),rnd.uniform(0, 10), rnd.uniform(1024, 10240),
                        rnd.uniform(1024, 10240), rnd.uniform(0, 10), rnd.uniform(0, 100))
              for cid in range(10)]
    for c in contrs:
      c.add_dependencies(*[str(i) for i in range(int(c.id))])
    self.app = Application(env, 'test', contrs)

  def test_empty_application(self):
    try:
      Application(self.env, 'empty', [])
    except:
      self.fail('Failed to create empty application')

  def test_application_with_one_container(self):
    try:
      Application(self.env, 'one-container', [self.app.containers[0]])
    except:
      self.fail('Failed to create an application with only one container')

  def test_directed_acyclic_graph(self):
    self.assertTrue(nx.is_directed_acyclic_graph(self.app.dag))

  def test_predecessors(self):
    expected = set([str(i) for i in range(5)])
    preds = self.app.get_predecessors('5')
    for p in preds:
      self.assertIsInstance(p, Container)
    self.assertEqual(set([p.id for p in preds]), expected)
    self.assertEqual(self.app.get_predecessors('0'), [])
    self.assertEqual(self.app.get_predecessors('non-existent'), [])

  def test_successors(self):
    expected = set([str(i) for i in range(6, 10)])
    succs = self.app.get_successors('5')
    for s in succs:
      self.assertIsInstance(s, Container)
    self.assertEqual(set([s.id for s in succs]), expected)
    self.assertEqual(self.app.get_successors('9'), [])
    self.assertEqual(self.app.get_successors('non-existent'), [])

  def test_application_with_cycle(self):
    contrs = sorted(self.app.containers, key=lambda c: int(c.id))[:3]
    contrs[0].add_dependencies('1', '2')
    with self.assertRaises(ValueError):
      Application(self.env, 'app-with-cycle', contrs)

  def test_application_with_no_dependencies(self):
    env = self.env
    app = Application(env, 'test',
                      [Container(env, str(cid),rnd.uniform(0, 10), rnd.uniform(1024, 10240),
                               rnd.uniform(1024, 10240), rnd.uniform(0, 10), rnd.uniform(0, 100))
                     for cid in range(10)])
    self.assertEqual(len(app.get_sources()), 10)


class RandomDataflowGeneratorTest(unittest.TestCase):

  def setUp(self):
    from application.gen import RandomDAGGenerator
    self.dag = RandomDAGGenerator(100, 100, .1, .1).generate()

  # def test_dataflow_validity(self):
  #   from application.gen import RandomDataflowGenerator
  #   dag = self.dag
  #   gen = RandomDataflowGenerator(self.dag, 100, 500, 1.)
  #   dataflows = gen.generate()
  #   for src, dfs in dataflows.items():
  #     print(src, dfs)
  #
  #   print(sum([len(dfs) for dfs in dataflows]))


class RandomApplicationGeneratorTest(unittest.TestCase):

  def setUp(self):
    from application.gen import RandomApplicationGenerator
    env = simpy.Environment()
    self.gen = RandomApplicationGenerator(env, n_nodes_lo=2, n_nodes_hi=20,
                                          edge_den_lo=.3, edge_den_hi=1.,
                                          cpus_lo=.1, cpus_hi=4,
                                          mem_lo=128, mem_hi=1024 * 4,
                                          disk_lo=1024, disk_hi=1024 * 10,
                                          gpus_lo=0, gpus_hi=4,
                                          runtime_lo=60, runtime_hi=60 * 60,
                                          output_size_lo=0, output_size_hi=10000)

  def test_application_generation(self):
    app = self.gen.generate()
    self.assertIsInstance(app, Application)


