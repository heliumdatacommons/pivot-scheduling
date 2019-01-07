import os
import yaml
import numpy.random as rnd

import appliance

from enum import Enum
from numbers import Number
from collections import Iterable
from simpy import Environment, Container, Resource, Store

from resource.network import NetworkRoute
from util import Loggable, Singleton


class Cluster(Loggable):

  def __init__(self, env, dispatch_q, notify_q, hosts=[], storage=[], routes=[], meta=None):
    assert isinstance(env, Environment)
    self.__env = env
    assert isinstance(dispatch_q, Store)
    self.__dispatch_q = dispatch_q
    assert isinstance(notify_q, Store)
    self.__notify_q = notify_q
    assert isinstance(hosts, Iterable) and all([isinstance(h, Host) for h in hosts])
    self.__hosts = {}
    for h in hosts:
      self.add_host(h)
    assert isinstance(storage, Iterable) and all([isinstance(s, Storage) for s in storage])
    self.__storage = {}
    for s in storage:
      self.add_storage(s)
    self.__storage_by_locality = {s.locality: s for s in storage}
    assert isinstance(routes, Iterable) and all([isinstance(r, NetworkRoute) for r in routes])
    self.__routes = {}
    for r in routes:
      self.add_route(r)
    assert meta is None or isinstance(meta, ResourceMetadata)
    self.__meta = meta

  @property
  def hosts(self):
    return list(self.__hosts.values())

  @property
  def storage(self):
    return list(self.__storage.values())

  @property
  def routes(self):
    return list(self.__routes.values())

  @property
  def meta(self):
    return self.__meta

  @property
  def dispatch_q(self):
    return self.__dispatch_q

  @property
  def notify_q(self):
    return self.__notify_q

  def start(self):
    env = self.__env
    env.process(self._start())

  def get_host(self, id):
    return self.__hosts.get(id)

  def get_storage(self, id):
    return self.__storage.get(id)

  def get_storage_by_locality(self, l):
    assert isinstance(l, Locality)
    return self.__storage_by_locality.get(l)

  def get_route(self, src_id, dst_id):
    return self.__routes.get((str(src_id), str(dst_id)))

  def add_host(self, h):
    assert isinstance(h, Host)
    if h.id in self.__hosts:
      raise ValueError('Host %s already exists')
    h.cluster = self
    self.__hosts[h.id] = h

  def add_storage(self, s):
    assert isinstance(s, Storage)
    s.cluster = self
    self.__storage[s.id] = s

  def add_route(self, r):
    assert isinstance(r, NetworkRoute)
    r.cluster = self
    self.__routes[(r.src.id, r.dst.id)] = r

  def _start(self):
    env = self.__env
    while True:
      c = yield self.__dispatch_q.get()
      if not isinstance(c, appliance.Container):
        self.logger.error('Received non-container item: %s'%type(c))
        continue
      if c.placement not in self.__hosts:
        self.logger.error('Unrecognized host: %s'%c.placement)
        continue
      self.__env.process(self._execute_container(c, self.__hosts.get(c.placement)))

  def _execute_container(self, c, host):
    env = self.__env
    # Execute the job
    self.logger.info('[%.3f] Container %s starts running on host %s, '
                     'etc: %s' % (env.now, c.id, host.id, c.runtime))
    success = yield env.process(host.execute(c))

    yield self.__notify_q.put((success, c))

  def _pull_data(self, c, p, event_q):
    env = self.__env
    p_host, c_host = self.get_host(p.placement), self.get_host(c.placement)
    storage = self.get_storage_by_locality(p_host.locality)
    route = self.get_route(storage.id, c_host.id)
    evt = env.event()
    self.logger.info('[%.3f] Container %s starts pulling %d bytes data from %s, '
                     'bw: %.3f, etc: %.3f'%(env.now, c.id, p.output_nbytes, p.id, route.bw,
                                             p.output_nbytes/route.bw))
    yield env.process(route.send(p.output_nbytes, evt))
    yield evt
    self.logger.info('[%.3f] Container %s finished pulling data from %s'%(env.now, c.id, p.id))
    yield event_q.put(p.id)

  def _dump_data(self, c, host):
    env = self.__env
    local_storage = self.get_storage_by_locality(host.locality)
    route = self.get_route(host.id, local_storage.id)
    transfer_evt = env.event()
    self.logger.info('[%.3f] Container %s starts dumping %d bytes data, '
                     'bw: %.3f, etc: %.2f'%(env.now, c.id, c.output_nbytes, route.bw,
                                            c.output_nbytes/route.bw))
    yield env.process(route.send(c.output_nbytes, transfer_evt))
    yield transfer_evt
    self.logger.info('[%.3f] Container %s finished dumping of %d bytes '
                     'data'%(env.now, c.id, c.output_nbytes))


class Node(Loggable):

  def __init__(self, env, id, cluster=None, locality=None):
    assert isinstance(env, Environment)
    self.__env = env
    assert id is not None
    self.__id = str(id)
    assert cluster is None or isinstance(cluster, Cluster)
    self.__cluster = cluster
    assert locality is None or isinstance(locality, Locality)
    self.__locality = locality

  @property
  def env(self):
    return self.__env

  @property
  def id(self):
    return self.__id

  @property
  def cluster(self):
    return self.__cluster

  @property
  def locality(self):
    return self.__locality

  @cluster.setter
  def cluster(self, cluster):
    self.__cluster = cluster

  def __repr__(self):
    return self.id

  def __hash__(self):
    return hash(self.id, self)

  def __eq__(self, other):
    return isinstance(other, Node) and self.id == other.id


class Host(Node):

  def __init__(self, env, id, cpus, mem, disk, gpus, cluster=None, locality=None):
    """

    :param env: simpy.Environment
    :param id:
    :param cpus:
    :param mem:
    :param disk:
    :param gpus:
    :param cluster
    :param locality:
    """
    super(Host, self).__init__(env, id, cluster, locality)
    self.__resource = HostResource(env, cpus, mem, disk, gpus)
    self.__containers = set()

  @property
  def resource(self):
    return self.__resource

  @property
  def in_use(self):
    return len(self.__containers) > 0

  def execute(self, c):
    assert isinstance(c, appliance.Container)
    env, resc = self.env, self.__resource
    self.logger.debug('[%d] Container: %s, cpus: %.1f, mem: %d, disk: %d, gpus: %d' % (
      env.now, c.id, self.resource.cpus_available, self.resource.mem_available,
      self.resource.disk_available, self.resource.gpus_available))
    success = yield env.process(resc.subscribe(c.cpus, c.mem, c.disk, c.gpus))
    if success:
      self.__containers.add(c)
      self.logger.debug('[%d] Container: %s, cpus: %.1f, mem: %d, disk: %d, gpus: %d' % (
        env.now, c.id, self.resource.cpus_available, self.resource.mem_available,
        self.resource.disk_available, self.resource.gpus_available))
      c.set_running()

      # Pull data from the predecessors
      preds = [p for p in c.appliance.get_predecessors(c.id) if p.output_nbytes > 0]
      self.logger.info('Container %s pulls data from predecessors: %s' % (c.id, preds))
      if len(preds) > 0:
        event_q, finished = Store(env), set()
        for p in preds:
          env.process(self._pull_data(c, p, event_q))
        preds = set([p.id for p in preds])
        while finished != preds:
          finished_p = yield event_q.get()
          finished.add(finished_p)

      # Execute the container
      yield env.timeout(c.runtime)

      # Dumping output data to the local storage
      if c.output_nbytes > 0:
        yield env.process(self._dump_data(c))

      yield env.process(resc.unsubscribe(c.cpus, c.mem, c.disk, c.gpus))
      self.logger.debug('[%d] Container: %s, cpus: %.1f, mem: %d, disk: %d, gpus: %d' % (
        env.now, c.id, self.resource.cpus_available, self.resource.mem_available,
        self.resource.disk_available, self.resource.gpus_available))
      self.__containers.remove(c)
      c.set_finished()
    else:
      resc = self.resource
      if c.cpus > resc.cpus_available:
        self.logger.info(
          '[%.3f] CPU demand: %.3f, available: %.3f' % (env.now, c.cpus, resc.cpus_available))
      if c.mem > resc.mem_available:
        self.logger.info(
          '[%.3f] Memory demand: %.3f, available: %.3f' % (env.now, c.mem, resc.mem_available))
      if c.disk > resc.disk_available:
        self.logger.info(
          '[%.3f] Disk demand: %.3f, available: %.3f' % (env.now, c.disk, resc.disk_available))
      if c.gpus > resc.gpus_available:
        self.logger.info(
          '[%.3f] GPU demand: %.3f, available: %.3f' % (env.now, c.gpus, resc.gpus_available))

    return success

  def _pull_data(self, c, p, event_q):
    env, cluster = self.env, self.cluster
    p_host, c_host = cluster.get_host(p.placement), cluster.get_host(c.placement)
    storage = cluster.get_storage_by_locality(p_host.locality)
    route = cluster.get_route(storage.id, c_host.id)
    evt = env.event()
    self.logger.info('[%.3f] Container %s starts pulling %d bytes data from %s, '
                     'bw: %.3f, etc: %.3f'%(env.now, c.id, p.output_nbytes, p.id, route.bw,
                                            p.output_nbytes/route.bw))
    yield env.process(route.send(p.output_nbytes, evt))
    yield evt
    self.logger.info('[%.3f] Container %s finished pulling data from %s'%(env.now, c.id, p.id))
    yield event_q.put(p.id)

  def _dump_data(self, c):
    env, cluster = self.env, self.cluster
    local_storage = cluster.get_storage_by_locality(self.locality)
    route = cluster.get_route(self.id, local_storage.id)
    transfer_evt = env.event()
    self.logger.info('[%.3f] Container %s starts dumping %d bytes data, '
                     'bw: %.3f, etc: %.2f'%(env.now, c.id, c.output_nbytes, route.bw,
                                            c.output_nbytes/route.bw))
    yield env.process(route.send(c.output_nbytes, transfer_evt))
    yield transfer_evt
    self.logger.info('[%.3f] Container %s finished dumping of %d bytes '
                     'data'%(env.now, c.id, c.output_nbytes))

  def __repr__(self):
    return repr((self.id, self.locality))

  def __eq__(self, other):
    return isinstance(other, Host) and super(Host, self).__eq__(other)


class HostResource(Loggable):

  def __init__(self, env, cpus, mem, disk, gpus):
    assert isinstance(env, Environment)
    self.__env = env
    assert isinstance(cpus, Number)
    self.__cpus = Container(env, cpus, init=cpus)
    assert isinstance(mem, int)
    self.__mem = Container(env, mem, init=mem)
    assert isinstance(disk, int)
    self.__disk = Container(env, disk, init=disk)
    assert isinstance(gpus, int)
    self.__gpus = Container(env, gpus, init=gpus)
    self.__lock = Resource(env)

  @property
  def total_cpus(self):
    return self.__cpus.capacity

  @property
  def total_mem(self):
    return self.__mem.capacity

  @property
  def total_disk(self):
    return self.__disk.capacity

  @property
  def total_gpus(self):
    return self.__gpus.capacity

  @property
  def cpus_used(self):
    return self.__cpus.capacity - self.__cpus.level

  @property
  def mem_used(self):
    return self.__mem.capacity - self.__mem.level

  @property
  def disk_used(self):
    return self.__disk.capacity - self.__disk.level

  @property
  def gpus_used(self):
    return self.__gpus.capacity - self.__gpus.level

  @property
  def cpus_available(self):
    return self.__cpus.level

  @property
  def mem_available(self):
    return self.__mem.level

  @property
  def disk_available(self):
    return self.__disk.level

  @property
  def gpus_available(self):
    return self.__gpus.level

  def subscribe(self, cpus, mem, disk, gpus):
    with self.__lock.request() as lock:
      yield lock
      if cpus > self.cpus_available or cpus < 0 \
          or mem > self.mem_available or mem < 0 \
          or disk > self.disk_available or disk < 0 \
          or gpus > self.gpus_available or gpus < 0:
        return False
      if cpus > 0:
        yield self.__cpus.get(cpus)
      if mem > 0:
        yield self.__mem.get(mem)
      if disk > 0:
        yield self.__disk.get(disk)
      if gpus > 0:
        yield self.__gpus.get(gpus)
    return True

  def unsubscribe(self, cpus, mem, disk, gpus):
    with self.__lock.request() as lock:
      yield lock
      if 0 < cpus <= self.cpus_used:
        yield self.__cpus.put(cpus)
      if 0 < mem <= self.mem_used:
        yield self.__mem.put(mem)
      if 0 < disk <= self.disk_used:
        yield self.__disk.put(disk)
      if 0 < gpus <= self.gpus_used:
        yield self.__gpus.put(gpus)


class Storage(Node):

  def __init__(self, env, id, cluster=None, locality=None):
    super(Storage, self).__init__(env, id, cluster, locality)

  def __eq__(self, other):
    return isinstance(other, Storage) and super(Storage, self).__eq__(other)


class Cloud(Enum):

  AWS = 'aws'
  GCP = 'gcp'


class Region(Enum):

  # AWS
  US_EAST_1 = 'us-east-1'
  US_EAST_2 = 'us-east-2'
  US_WEST_1 = 'us-west-1'
  US_WEST_2 = 'us-west-2'
  CA_CENTRAL_1 = 'ca-central-1'

  # GCP
  US_EAST1 = 'us-east1'
  US_EAST4 = 'us-east4'
  US_WEST1 = 'us-west1'
  US_WEST2 = 'us-west2'
  US_CENTRAL1 = 'us-central1'
  NORTHAMERICA_NORTHEAST1 = 'northamerica-northeast1'


class Zone(Enum):

  A = 'a'
  B = 'b'
  C = 'c'
  D = 'd'


class Locality:

  def __init__(self, cloud, region, zone):
    assert isinstance(cloud, Cloud)
    assert isinstance(region, Region)
    assert isinstance(zone, Zone)
    self.__cloud = cloud
    self.__region = region
    self.__zone = zone

  @property
  def cloud(self):
    return self.__cloud

  @property
  def region(self):
    return self.__region

  @property
  def zone(self):
    return self.__zone

  def __repr__(self):
    return '(%s, %s, %s)'%(self.cloud.value, self.region.value, self.zone.value)

  def __hash__(self):
    return hash((self.cloud, self.region, self.zone))

  def __eq__(self, other):
    return isinstance(other, Locality) \
           and self.cloud == other.cloud \
           and self.region == other.region \
           and self.zone == other.zone


class ResourceMetadata(metaclass=Singleton):

  def __init__(self):
    self.__zones = []
    self.__cost, self.__bw = {}, {}
    self._load_locality_info_from_file()

  @property
  def zones(self):
    return list(self.__zones)

  @property
  def cost(self):
    return dict(self.__cost)

  @property
  def bw(self):
    return dict(self.__bw)

  def _load_locality_info_from_file(self):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root_dir, 'locality.yml')) as f:
      locality_f = yaml.load(f)
      locality, meta = locality_f['locality'], locality_f['meta']
      for cloud, regions in locality.items():
        for region, zones in regions.items():
          for zone in zones:
            self.__zones += Locality(Cloud(cloud), Region(region), Zone(zone)),
      for key, vals in meta.items():
        src, dst = key.split('--')
        src_cloud, src_region = src.split('_')
        dst_cloud, dst_region = dst.split('_')
        for sz in locality[src_cloud][src_region]:
          for dz in locality[dst_cloud][dst_region]:
            src_locality = Locality(Cloud(src_cloud), Region(src_region), Zone(sz))
            dst_locality = Locality(Cloud(dst_cloud), Region(dst_region), Zone(dz))
            self.__cost[(src_locality, dst_locality)] = vals['cost']
            multi_factor = 10 ** 3 if src_locality == dst_locality else rnd.uniform(.95, 1.05)
            self.__bw[(src_locality, dst_locality)] = vals['bw'] * multi_factor



