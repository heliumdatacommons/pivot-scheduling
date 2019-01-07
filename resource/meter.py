import simpy

from collections import defaultdict

from util import Loggable


class Meter(Loggable):

  def __init__(self, env):
    assert isinstance(env, simpy.Environment)
    self.__env = env
    self.__hosts = defaultdict(list)
    self.__routes = defaultdict(list)

  def host_check_in(self, h):
    pass

  def host_check_out(self, h):
    pass


