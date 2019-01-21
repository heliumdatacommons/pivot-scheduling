import simpy
import uuid
import networkx as nx
import numpy.random as rnd

from collections import deque, defaultdict

from appliance import Appliance, Container, Dataflow
from util import Loggable


class RandomDAGGenerator(Loggable):

  def __init__(self, n_nodes_lo, n_nodes_hi, edge_den_lo, edge_den_hi, seed=None):
    """

    :param n_nodes_lo: lower bound of number of nodes
    :param n_nodes_hi: upper bound of number of nodes
    :param edge_den_lo: lower bound of edge density
    :param edge_den_hi: upper bound of edge density
    :param seed: random seed
    """
    assert 1 < n_nodes_lo <= n_nodes_hi
    assert 0 < edge_den_lo <= edge_den_hi <= 1
    self.__n_nodes_lo = n_nodes_lo
    self.__n_nodes_hi = n_nodes_hi
    self.__edge_den_lo = edge_den_lo
    self.__edge_den_hi = edge_den_hi
    self.__seed = seed
    rnd.seed(seed)

  def generate(self):
    n_nodes = int(rnd.uniform(self.__n_nodes_lo, self.__n_nodes_hi))
    edge_den = rnd.uniform(self.__edge_den_lo, self.__edge_den_hi)
    g = nx.fast_gnp_random_graph(n_nodes, edge_den, seed=self.__seed, directed=True)
    return nx.DiGraph([(u, v) for u, v in g.edges() if u < v])


# class RandomDataflowGenerator(Loggable):
#
#   def __init__(self, dag, nbytes_lo, nbytes_hi, density=.5, seed=None):
#     assert isinstance(dag, nx.DiGraph)
#     assert 1 < nbytes_lo <= nbytes_hi
#     self.__dag = dag
#     self.__nbytes_lo = nbytes_lo
#     self.__nbytes_hi = nbytes_hi
#     self.__density = density
#     rnd.seed(seed)
#
#   def generate(self):
#     dag, dataflows = self.__dag, defaultdict(list)
#     lo, hi = self.__nbytes_lo, self.__nbytes_hi
#     sources = [str(n) for n in dag.nodes if dag.in_degree(n) == 0]
#     q = deque([(s, [ss for ss in sources if ss != s]) for s in sources])
#     while q:
#       src, dsts = q.popleft()
#       src_preds = set(dag.predecessors(int(src)))
#       for dst in dsts:
#         dst_preds = set(dag.predecessors(int(dst)))
#         if src_preds == dst_preds and rnd.uniform(0, 1) <= self.__density:
#           dataflows[src] += Dataflow(src, dst, rnd.randint(lo, hi)),
#       successors = [str(s) for s in dag.successors(int(src))]
#       for s in successors:
#         q += (s, [ss for ss in successors if ss != s]),
#     return dataflows


class RandomApplianceGenerator(Loggable):

  def __init__(self, env, n_nodes_lo, n_nodes_hi, edge_den_lo, edge_den_hi,
               cpus_lo, cpus_hi, mem_lo, mem_hi, disk_lo, disk_hi, gpus_lo, gpus_hi,
               runtime_lo, runtime_hi, output_nbytes_lo, output_nbytes_hi, seed=None):
    assert isinstance(env, simpy.Environment)
    assert 0 < cpus_lo <= cpus_hi
    assert 0 < mem_lo <= mem_hi
    assert 0 <= disk_lo <= disk_hi
    assert 0 <= gpus_lo <= gpus_hi
    assert 0 < runtime_lo <= runtime_hi
    assert 0 <= output_nbytes_lo <= output_nbytes_hi
    self.__env = env
    self.__dag_gen = RandomDAGGenerator(n_nodes_lo, n_nodes_hi, edge_den_lo, edge_den_hi, seed)
    self.__cpus_lo, self.__cpus_hi = cpus_lo, cpus_hi
    self.__mem_lo, self.__mem_hi = mem_lo, mem_hi
    self.__disk_lo, self.__disk_hi = disk_lo, disk_hi
    self.__gpus_lo, self.__gpus_hi = gpus_lo, gpus_hi
    self.__runtime_lo, self.__runtime_hi = runtime_lo, runtime_hi
    self.__output_nbytes_lo, self.__output_nbytes_hi = output_nbytes_lo, output_nbytes_hi
    rnd.seed(seed)

  def generate(self):
    dag = self.__dag_gen.generate()
    containers = {}
    for n in dag.nodes:
      containers[n] = Container(self.__env, str(n),
                                cpus=rnd.uniform(self.__cpus_lo, self.__cpus_hi),
                                mem=rnd.randint(self.__mem_lo, self.__mem_hi),
                                disk=rnd.randint(self.__disk_lo, self.__disk_hi),
                                gpus=rnd.randint(self.__gpus_lo, self.__gpus_hi),
                                runtime=rnd.uniform(self.__runtime_lo, self.__runtime_hi),
                                output_nbytes=rnd.randint(self.__output_nbytes_lo,
                                                          self.__output_nbytes_hi))
    for u, v in dag.edges:
      containers[v].add_dependencies(str(u))
    app = Appliance(self.__env, str(uuid.uuid4()), containers.values())
    # app.visualize()
    return app


class SequentialApplianceGenerator(Loggable):

  def __init__(self, env, n_nodes_lo, n_nodes_hi,
               cpus_lo, cpus_hi, mem_lo, mem_hi, disk_lo, disk_hi, gpus_lo, gpus_hi,
               runtime_lo, runtime_hi, output_nbytes_lo, output_nbytes_hi, seed=None):
    assert isinstance(env, simpy.Environment)
    assert 0 < cpus_lo <= cpus_hi
    assert 0 < mem_lo <= mem_hi
    assert 0 <= disk_lo <= disk_hi
    assert 0 <= gpus_lo <= gpus_hi
    assert 0 < runtime_lo <= runtime_hi
    assert 0 <= output_nbytes_lo <= output_nbytes_hi
    self.__env = env
    self.__n_nodes_lo, self.__n_nodes_hi = n_nodes_lo, n_nodes_hi
    self.__cpus_lo, self.__cpus_hi = cpus_lo, cpus_hi
    self.__mem_lo, self.__mem_hi = mem_lo, mem_hi
    self.__disk_lo, self.__disk_hi = disk_lo, disk_hi
    self.__gpus_lo, self.__gpus_hi = gpus_lo, gpus_hi
    self.__runtime_lo, self.__runtime_hi = runtime_lo, runtime_hi
    self.__output_nbytes_lo, self.__output_nbytes_hi = output_nbytes_lo, output_nbytes_hi
    rnd.seed(seed)

  def generate(self):
    dag = self._generate_dag()
    containers = {}
    for n in dag.nodes:
      containers[n] = Container(self.__env, str(n),
                                cpus=rnd.uniform(self.__cpus_lo, self.__cpus_hi),
                                mem=rnd.randint(self.__mem_lo, self.__mem_hi),
                                disk=rnd.randint(self.__disk_lo, self.__disk_hi),
                                gpus=rnd.randint(self.__gpus_lo, self.__gpus_hi),
                                runtime=rnd.uniform(self.__runtime_lo, self.__runtime_hi),
                                output_nbytes=rnd.randint(self.__output_nbytes_lo,
                                                          self.__output_nbytes_hi))
    for u, v in dag.edges:
      containers[v].add_dependencies(str(u))
    app = Appliance(self.__env, str(uuid.uuid4()), containers.values())
    app.visualize()
    return app

  def _generate_dag(self):
    n_nodes = rnd.randint(self.__n_nodes_lo, self.__n_nodes_hi)
    return nx.DiGraph([(i - 1, i) for i in range(1, n_nodes)])


class DataParallelApplianceGenerator(Loggable):

  def __init__(self, env, min_cpus, max_cpus, min_mem, max_mem, min_disk, max_disk,
               min_gpus, max_gpus, min_seq_steps, max_seq_steps,
               min_parallel_steps, max_parallel_steps, min_parallel_level, max_parallel_level,
               min_runtime, max_runtime, min_output_nbytes, max_output_nbytes, seed):
    assert isinstance(env, simpy.Environment)
    assert 0 < min_cpus <= max_cpus
    assert 0 < min_mem <= max_mem
    assert 0 <= min_disk <= max_disk
    assert isinstance(min_gpus, int) and isinstance(max_gpus, int) and 0 <= min_gpus <= max_gpus
    assert 0 <= min_seq_steps <= max_seq_steps
    assert 0 <= min_parallel_steps <= max_parallel_steps
    assert 1 < min_parallel_level <= max_parallel_level
    assert 0 < min_runtime <= max_runtime
    assert 0 <= min_output_nbytes <= max_output_nbytes
    self.__env = env
    self.__min_cpus, self.__max_cpus = min_cpus, max_cpus
    self.__min_mem, self.__max_mem = min_mem, max_mem
    self.__min_disk, self.__max_disk = min_disk, max_disk
    self.__min_gpus, self.__max_gpus = min_gpus, max_gpus
    self.__min_seq_steps, self.__max_seq_steps = min_seq_steps, max_seq_steps + 1
    self.__min_parallel_steps, self.__max_parallel_steps = min_parallel_steps, max_parallel_steps + 1
    self.__min_parallel_level, self.__max_parallel_level = min_parallel_level, max_parallel_level + 1
    self.__min_runtime, self.__max_runtime = min_runtime, max_runtime
    self.__min_output_nbytes, self.__max_output_nbytes = min_output_nbytes, max_output_nbytes
    rnd.seed(seed)

  def generate(self):
    n_seq_steps = rnd.randint(self.__min_seq_steps, self.__max_seq_steps)
    n_parallel_steps = rnd.randint(self.__min_parallel_steps, self.__max_parallel_steps)
    total_steps = n_seq_steps + n_parallel_steps
    assert total_steps > 0
    p_seq_step = n_seq_steps/total_steps
    n_nodes = 0
    containers, last_step = [], []
    for is_seq in rnd.choice(a=[True, False], size=total_steps, p=[p_seq_step, 1 - p_seq_step]):
      cpus = rnd.uniform(self.__min_cpus, self.__max_cpus)
      mem = rnd.randint(self.__min_mem, self.__max_mem)
      disk = rnd.randint(self.__min_disk, self.__max_disk)
      gpus = rnd.randint(self.__min_gpus, self.__max_gpus)
      unit_output_nbytes = rnd.randint(self.__min_output_nbytes, self.__max_output_nbytes)
      if is_seq:
        cid = n_nodes + 1
        runtime = rnd.uniform(self.__min_runtime, self.__max_runtime)
        output_nbytes = unit_output_nbytes * runtime
        c = Container(self.__env, str(cid), cpus=cpus, mem=mem, disk=disk, gpus=gpus,
                      runtime=runtime, output_nbytes=output_nbytes)
        for prev in last_step:
          c.add_dependencies(prev)
        containers += c,
        last_step = [str(cid)]
        n_nodes += 1
      else:
        parallel_level = rnd.randint(self.__min_parallel_level, self.__max_parallel_level) \
          if len(last_step) < 2 else len(last_step)
        for i, cid in enumerate(range(n_nodes + 1, n_nodes + parallel_level + 1)):
          runtime = rnd.uniform(self.__min_runtime, self.__max_runtime)
          output_nbytes = unit_output_nbytes * runtime
          c = Container(self.__env, str(cid), cpus=cpus, mem=mem, disk=disk,gpus=gpus,
                        runtime=runtime, output_nbytes=output_nbytes)
          cur = i % parallel_level
          while cur < len(last_step):
            c.add_dependencies(last_step[cur])
            cur += parallel_level
          containers += c,
        last_step = [str(i) for i in range(n_nodes + 1, n_nodes + parallel_level + 1)]
        n_nodes += parallel_level
    app = Appliance(self.__env, str(uuid.uuid4()), containers)
    # app.visualize()
    return app








