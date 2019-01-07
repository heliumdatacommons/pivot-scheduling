import sys
import logging


class Loggable(object):

  @property
  def logger(self):
    fmt = logging.Formatter('%(name)s.%(funcName)s:%(lineno)s\t%(message)s')
    logger = logging.getLogger(self.__class__.__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
      stream_hdlr = logging.StreamHandler(sys.stdout)
      stream_hdlr.setFormatter(fmt)
      logger.addHandler(stream_hdlr)
    return logger


class Singleton(type):

  _instances = {}

  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
    return cls._instances[cls]