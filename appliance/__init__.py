import uuid
import simpy
import numpy as np
import networkx as nx

from enum import Enum
from collections import Iterable
from numbers import Number
from collections import deque


from util import Loggable


class Appliance(Loggable):

  def __init__(self, env, id, containers=[]):
    assert isinstance(env, simpy.Environment)
    assert id is not None
    self.__id = str(id)
    self.__env = env
    self.__containers = {c.id: c for c in containers}
    for c in containers:
      c.appliance = self
    self.__container_indices = {idx: c for idx, c in enumerate(self.containers)}
    self.__container_reverse_indices = {c.id: idx for idx, c in self.__container_indices.items()}
    self.__dag = self._create_dag()
    self.__start_time, self.__end_time = 0, 0

  @property
  def env(self):
    return self.__env

  @property
  def id(self):
    return self.__id

  @property
  def containers(self):
    return list(self.__containers.values())

  @property
  def avg_data_size(self):
    return np.mean([c.output_nbytes for c in self.containers])

  @property
  def dag(self):
    return nx.DiGraph(self.__dag)

  @property
  def start_time(self):
    return self.__start_time

  @property
  def end_time(self):
    return self.__end_time

  @start_time.setter
  def start_time(self, st):
    self.__start_time = st

  @end_time.setter
  def end_time(self, et):
    self.__end_time = et

  @property
  def is_finished(self):
    return all([s.state == ContainerState.FINISHED for s in self.get_sinks()])

  @env.setter
  def env(self, env):
    assert isinstance(env, simpy.Environment)
    self.__env = env

  def clone(self):
    return Appliance(self.__env, str(uuid.uuid4()),
                     containers=[Container(self.__env, c.id, c.cpus, c.mem, c.disk, c.gpus,
                                           c.runtime, c.output_nbytes, c.dependencies)
                                 for c in self.containers])

  def get_container_by_id(self, id):
    return self.__containers.get(id)

  def get_container_by_index(self, idx):
    return self.__container_indices.get(idx)

  def get_predecessors(self, id):
    idx = self.__container_reverse_indices.get(id)
    if idx is None:
      raise ValueError('Unrecognized container: %s'%id)
    return [self.__container_indices[p_idx] for p_idx in self.__dag.predecessors(idx)
            if p_idx in self.__container_indices]

  def get_successors(self, id):
    idx = self.__container_reverse_indices.get(id)
    if idx is None:
      raise ValueError('Unrecognized container: %s'%id)
    return [self.__container_indices[s_idx] for s_idx in self.__dag.successors(idx)
            if s_idx in self.__container_indices]

  def get_unfinished_predecessors(self, id):
    return [p for p in self.get_predecessors(id) if p.state != ContainerState.FINISHED]

  def get_ready_successors(self, id):
    return [s for s in self.get_successors(id) if len(self.get_unfinished_predecessors(s.id)) == 0]

  def get_sources(self):
    return [self.__container_indices[n] for n in self.__dag.nodes
            if n in self.__container_indices and self.__dag.in_degree(n) == 0]

  def get_sinks(self):
    return [self.__container_indices[n] for n in self.__dag.nodes
            if n in self.__container_indices and self.__dag.out_degree(n) == 0]

  def estimate_local_runtime(self):
    runtime, finished = 0, set()
    deadlines = deque(sorted([(c.runtime, c) for c in self.get_sources()]))
    while deadlines:
      deadline, c = deadlines.popleft()
      runtime = max(deadline, runtime)
      finished.add(c.id)
      qualified = deque([(runtime + s.runtime, s) for s in self.get_successors(c.id)
                         if finished.issuperset([p.id for p in self.get_predecessors(s.id)])])
      if qualified:
        deadlines = deque(sorted(deadlines + qualified))
    return runtime

  def visualize(self):
    import matplotlib.pyplot as plt
    nx.draw(self.dag)
    plt.show()
    # dag = self.dag
    # level = set([n for n in dag.nodes if dag.in_degree(n) == 0])
    # max_depth, max_width = 0, 0
    # shown = set()
    # res = []
    # while level:
    #   max_width = max(max_width, len(level))
    #   shown |= level
    #   res += level,
    #   new_level = set()
    #   for n in level:
    #     new_level.update([s for s in dag.successors(n) if set(dag.predecessors(s)).issubset(shown)])
    #   max_depth += 1
    #   level = new_level
    #
    # _, ax = plt.subplots()
    # ax.set_xlim(0, 1.25)
    # ax.set_ylim(0, 1.25)
    # radius = 1/(8 * max(max_depth, max_width))
    # pos_map = {}
    # for depth, row in enumerate(res):
    #   print(list(row))
    #   for i, n in enumerate(row):
    #     pos = (depth * 1/max_depth + radius, i * 1/max_width + radius)
    #     pos_map[n] = pos
    #     c = plt.Circle(pos, radius)
    #     ax.add_artist(c)
    #     print(n, list(dag.predecessors(n)))
    #     for pre in dag.predecessors(n):
    #       cur_x, cur_y = pos
    #       pre_x, pre_y = pos_map[pre]
    #       ax.arrow(pre_x + radius, pre_y + radius,
    #                cur_x - radius, cur_y - radius,
    #                head_width=radius, head_length=radius)
    #
    # plt.show()


  def _create_dag(self):
    contr_idx = self.__container_reverse_indices
    dag = nx.DiGraph()
    for c in self.containers:
      dag.add_node(contr_idx[c.id])
    for c in self.containers:
      for d in c.dependencies:
        dag.add_edge(contr_idx[d], contr_idx[c.id])
    if not nx.is_directed_acyclic_graph(dag):
      raise ValueError('Container(s) in the appliance cannot create a DAG')
    return dag

  def __repr__(self):
    return self.id

  def __hash__(self):
    return hash(self.id)

  def __eq__(self, other):
    return isinstance(other, Appliance) and self.id == other.id


class ContainerState(Enum):

  NASCENT = 'nascent'
  SUBMITTED = 'submitted'
  RUNNING = 'running'
  FINISHED = 'finished'


class Container(Loggable):

  def __init__(self, env, id, cpus, mem, disk, gpus=0, runtime=0, output_nbytes=0,
               dependencies=[], appliance=None, placement=None, state=ContainerState.NASCENT):
    """

    :param env
    :param id:
    :param cpus:
    :param mem:
    :param disk:
    :param gpus:
    :param runtime: the container runs forever if runtime is 0
    :param dependencies: list, dependencies of the container in an appliance
    :param dataflows: list, dataflows
    :param appliance
    :param placement: host ID
    :param state:
    """
    assert isinstance(env, simpy.Environment)
    self.__env = env
    self.__id = id
    self.__appliance = appliance
    self.__cpus = cpus
    self.__mem = mem
    self.__disk = disk
    self.__gpus = gpus
    assert isinstance(runtime, Number)
    self.__runtime = runtime
    assert isinstance(output_nbytes, Number)
    self.__output_nbytes = output_nbytes
    assert isinstance(dependencies, Iterable) and all([isinstance(d, str)for d in dependencies])
    self.__dependencies = list(dependencies)
    assert appliance is None or isinstance(appliance, Appliance)
    self.__appliance = appliance
    assert placement is None or isinstance(placement, str)
    self.__placement = placement
    assert state is None or isinstance(state, ContainerState)
    self.__state = state

  @property
  def id(self):
    return self.__id

  @property
  def cpus(self):
    return self.__cpus

  @property
  def mem(self):
    return self.__mem

  @property
  def disk(self):
    return self.__disk

  @property
  def gpus(self):
    return self.__gpus

  @property
  def state(self):
    return self.__state

  @property
  def runtime(self):
    return self.__runtime

  @property
  def dependencies(self):
    return list(self.__dependencies)

  @property
  def output_nbytes(self):
    return self.__output_nbytes

  @property
  def appliance(self):
    return self.__appliance

  @property
  def placement(self):
    return self.__placement

  @property
  def state(self):
    return self.__state

  @property
  def is_nascent(self):
    return self.__state == ContainerState.NASCENT

  @property
  def is_finished(self):
    return self.__state == ContainerState.FINISHED

  @property
  def is_running(self):
    return self.__state == ContainerState.RUNNING

  @property
  def is_submitted(self):
    return self.__state == ContainerState.SUBMITTED

  @appliance.setter
  def appliance(self, app):
    assert app is None or isinstance(app, Appliance)
    self.__appliance = app

  @placement.setter
  def placement(self, p):
    assert p is None or isinstance(p, str)
    self.__placement = p

  def set_nascent(self):
    self.__state = ContainerState.NASCENT

  def set_submitted(self):
    self.__state = ContainerState.SUBMITTED

  def set_running(self):
    self.__state = ContainerState.RUNNING

  def set_finished(self):
    self.__state = ContainerState.FINISHED

  def add_dependencies(self, *c):
    self.__dependencies = list(set(self.__dependencies + list(c)))

  def __repr__(self):
    return self.id

  def __hash__(self):
    return hash((self.id, self.appliance))

  def __eq__(self, other):
    return isinstance(other, Container) \
            and self.id == other.id \
            and self.appliance == other.appliance


class Dataflow:

  def __init__(self, src, dst, nbytes):
    assert isinstance(src, str)
    assert isinstance(dst, str)
    assert isinstance(nbytes, int) and nbytes > 0
    self.__src = src
    self.__dst = dst
    self.__nbytes = nbytes

  @property
  def src(self):
    return self.__src

  @property
  def dst(self):
    return self.__dst

  @property
  def nbytes(self):
    return self.__nbytes

  def __repr__(self):
    return '%s -> %s: %d bytes'%(self.src, self.dst, self.nbytes)


