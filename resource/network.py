import simpy
import resource

from numbers import Number

from util import Loggable


class Packet:

  SIZE = 1000

  def __init__(self, nbytes, transfer_event):
    assert isinstance(nbytes, int) and nbytes > 0
    assert isinstance(transfer_event, simpy.Event)
    self.__nbytes = nbytes
    self.__transfer_event = transfer_event

  @property
  def nbytes(self):
    return self.__nbytes

  @property
  def is_transferred(self):
    return self.__nbytes == 0

  def decrement(self):
    nbytes = self.nbytes
    self.__nbytes = nbytes - min(nbytes, Packet.SIZE)
    if self.__nbytes == 0:
      self.__transfer_event.succeed()


class NetworkRoute(Loggable):

  def __init__(self, env, src, dst, bw, cluster=None):
    assert isinstance(env, simpy.Environment)
    self.__env = env
    assert isinstance(src, resource.Node)
    self.__src = src
    assert isinstance(dst, resource.Node)
    self.__dst = dst
    assert isinstance(bw, Number) and bw > 0
    self.__bw = bw
    assert cluster is None or isinstance(cluster, resource.Cluster)
    self.__cluster = cluster
    self.__pkts = simpy.Store(env)
    self.__env.process(self._transfer())

  @property
  def src(self):
    return self.__src

  @property
  def dst(self):
    return self.__dst

  @property
  def bw(self):
    return self.__bw

  @property
  def cluster(self):
    return self.__cluster

  @cluster.setter
  def cluster(self, cluster):
    self.__cluster = cluster

  def send(self, nbytes, transfer_event):
    yield self.__pkts.put(Packet(nbytes, transfer_event))

  def _transfer(self):
    env = self.__env
    while True:
      pkt = yield self.__pkts.get()
      yield env.timeout(Packet.SIZE/self.__bw)
      self.logger.debug('[%s] processing packet %s'%(env.now, pkt))
      pkt.decrement()
      if not pkt.is_transferred:
        yield self.__pkts.put(pkt)

  def _fluctuate(self):
    pass

  def __repr__(self):
    return '%s -> %s'%(self.src.id, self.dst.id)

  def __hash__(self):
    return hash((self.src.id, self.dst.id))

  def __eq__(self, other):
    return isinstance(other, NetworkRoute) \
           and self.src.id == other.src.id \
           and self.dst.id == other.dst.id