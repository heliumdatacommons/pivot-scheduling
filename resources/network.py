import uuid
import simpy
import resources

from numbers import Number

from util import Loggable


class Packet:

  PACKET_SIZE = 1000

  def __init__(self, num_mbs, transfer_event):
    self.__id = str(uuid.uuid4())
    assert isinstance(num_mbs, Number) and num_mbs > 0
    assert isinstance(transfer_event, simpy.Event)
    self.__num_mbs = num_mbs
    self.__transfer_event = transfer_event

  @property
  def id(self):
    return self.__id

  @property
  def num_mbs(self):
    return self.__num_mbs

  @property
  def is_transferred(self):
    return self.__num_mbs == 0

  def decrement(self):
    total_mbs = self.num_mbs
    self.__num_mbs = total_mbs - min(total_mbs, Packet.PACKET_SIZE)
    if self.__num_mbs == 0:
      self.__transfer_event.succeed()


class NetworkRoute(Loggable):

  def __init__(self, env, src, dst, bw, cluster=None, meter=None):
    assert isinstance(env, simpy.Environment)
    self.__env = env
    assert isinstance(src, resources.Node)
    self.__src = src
    assert isinstance(dst, resources.Node)
    self.__dst = dst
    assert isinstance(bw, Number)
    self.__bw = bw
    assert cluster is None or isinstance(cluster, resources.Cluster)
    self.__cluster = cluster
    assert meter is None or isinstance(meter, resources.Meter)
    self.__meter = meter
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
  def realtime_bw(self):
    est = (sum([p.num_mbs for p in self.__pkts.items]) + 1) / self.__bw
    return 1/est if est else self.__bw

  @property
  def cluster(self):
    return self.__cluster

  @cluster.setter
  def cluster(self, cluster):
    self.__cluster = cluster

  def send(self, pkt_size, transfer_event):
    yield self.__pkts.put(Packet(pkt_size, transfer_event))

  def _transfer(self):
    env, meter = self.__env, self.__meter
    while True:
      pkt = yield self.__pkts.get()
      pkt_size = min(pkt.num_mbs, Packet.PACKET_SIZE)
      if meter:
        meter.route_check_in(self, pkt.id)
      if self.__bw > 0:
        yield env.timeout(pkt_size/self.__bw)
      # self.logger.debug('[%s] processing packet %s'%(env.now, pkt))
      if meter:
        meter.route_check_out(self, pkt.id, pkt_size)
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