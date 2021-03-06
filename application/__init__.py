import uuid
import simpy
import numpy as np
import networkx as nx

from enum import Enum
from collections import Iterable
from numbers import Number
from collections import deque


from util import Loggable


class Application(Loggable):

  def __init__(self, env, id, containers=[]):
    assert isinstance(env, simpy.Environment)
    assert id is not None
    self.__id = str(id)
    self.__env = env
    self.__containers = {c.id: c for c in containers}
    for c in containers:
      c.application = self
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
    return np.mean([c.output_size for c in self.containers])

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
    return all([s.is_finished for s in self.get_sinks()])

  @env.setter
  def env(self, env):
    assert isinstance(env, simpy.Environment)
    self.__env = env

  def clone(self):
    return Application(self.__env, str(uuid.uuid4()),
                       containers=[Container(self.__env, c.id, c.cpus, c.mem, c.disk, c.gpus,
                                             c.runtime, c.output_size, c.instances, c.dependencies)
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
    return [p for p in self.get_predecessors(id) if not p.is_finished]

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

  def _create_dag(self):
    try:
      contr_idx = self.__container_reverse_indices
      dag = nx.DiGraph()
      for c in self.containers:
        dag.add_node(contr_idx[c.id])
      for c in self.containers:
        for d in c.dependencies:
          dag.add_edge(contr_idx[d], contr_idx[c.id])
      if not nx.is_directed_acyclic_graph(dag):
        raise ValueError('Container(s) in the application cannot create a DAG')
      return dag
    except KeyError as e:
      print(self.id)
      raise e

  def __repr__(self):
    return self.id

  def __hash__(self):
    return hash(self.id)

  def __eq__(self, other):
    return isinstance(other, Application) and self.id == other.id


class TaskState(Enum):

  NASCENT = 'nascent'
  SUBMITTED = 'submitted'
  RUNNING = 'running'
  FINISHED = 'finished'


class Task:

  def __init__(self, id, container, cpus, mem, disk=0, gpus=0, runtime=0, output_size=0,
               placement=None, state=TaskState.NASCENT):
    self.__id = id
    self.container = container
    self.cpus = cpus
    self.mem = mem
    self.disk = disk
    self.gpus = gpus
    self.runtime = runtime
    self.output_size = output_size
    self.placement = placement
    self.state = state

  @property
  def id(self):
    return '%s/%s'%(self.container.id, self.__id)

  @property
  def is_nascent(self):
    return self.state == TaskState.NASCENT

  @property
  def is_submitted(self):
    return self.state == TaskState.SUBMITTED

  @property
  def is_running(self):
    return self.state == TaskState.RUNNING

  @property
  def is_finished(self):
    return self.state == TaskState.FINISHED

  def set_nascent(self):
    self.state == TaskState.NASCENT

  def set_submitted(self):
    self.state = TaskState.SUBMITTED

  def set_running(self):
    self.state = TaskState.RUNNING

  def set_finished(self):
    self.state = TaskState.FINISHED


class Container(Loggable):

  def __init__(self, env, id, cpus, mem, disk=0, gpus=0, runtime=0, output_size=0, instances=1,
               dependencies=[], application=None):
    """

    :param env
    :param id:
    :param cpus:
    :param mem:
    :param disk:
    :param gpus:
    :param runtime: the container runs forever if runtime is 0
    :param dependencies: list, dependencies of the container in an application
    :param dataflows: list, dataflows
    :param application
    """
    assert isinstance(env, simpy.Environment)
    self.__env = env
    self.__id = id
    self.__application = application
    self.__cpus = cpus
    self.__mem = mem
    self.__disk = disk
    self.__gpus = gpus
    assert isinstance(runtime, Number)
    self.__runtime = runtime
    assert isinstance(output_size, Number)
    self.__output_size = output_size
    assert isinstance(instances, int) and instances > 0
    self.__instances = instances
    assert isinstance(dependencies, Iterable) and all([isinstance(d, str)for d in dependencies])
    self.__dependencies = list(dependencies)
    assert application is None or isinstance(application, Application)
    self.__application = application
    self.__tasks = []

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
  def runtime(self):
    return self.__runtime

  @property
  def dependencies(self):
    return list(self.__dependencies)

  @property
  def output_size(self):
    return self.__output_size

  @property
  def instances(self):
    return self.__instances

  @property
  def application(self):
    return self.__application

  @property
  def tasks(self):
    return list(self.__tasks)

  @property
  def is_finished(self):
    tasks = self.__tasks
    return len(tasks) > 0 and all([t.state == TaskState.FINISHED for t in tasks])

  @application.setter
  def application(self, app):
    assert app is None or isinstance(app, Application)
    self.__application = app

  def add_dependencies(self, *c):
    self.__dependencies = list(set(self.__dependencies + list(c)))

  def generate_tasks(self):
    tasks = self.__tasks
    while len(tasks) < self.instances:
      cpus, mem, disk, gpus = self.__cpus, self.__mem, self.__disk, self.__gpus
      runtime, output_size = self.__runtime, self.__output_size
      tasks += Task(len(tasks), self, cpus, mem, disk, gpus, runtime, output_size),
      yield tasks[-1]

  def __repr__(self):
    return self.id

  def __hash__(self):
    return hash((self.id, self.application))

  def __eq__(self, other):
    return isinstance(other, Container) \
            and self.id == other.id \
            and self.application == other.application


class Dataflow:

  def __init__(self, src, dst, data_size):
    assert isinstance(src, str)
    assert isinstance(dst, str)
    assert isinstance(data_size, int) and data_size > 0
    self.__src = src
    self.__dst = dst
    self.__data_size = data_size

  @property
  def src(self):
    return self.__src

  @property
  def dst(self):
    return self.__dst

  @property
  def data_size(self):
    return self.__data_size

  def __repr__(self):
    return '%s -> %s: %d bytes'%(self.src, self.dst, self.data_size)


