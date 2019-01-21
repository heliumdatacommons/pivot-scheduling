import os
import time
import json
import simpy
import unittest
import datetime
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from collections import defaultdict, Counter

from appliance import Appliance
from appliance.gen import DataParallelApplianceGenerator
from resources.gen import RandomClusterGenerator
from resources.meter import Meter
from scheduler import OpportunisticGlobalScheduler, FirstFitGlobalScheduler, BestFitGlobalScheduler, CostAwareGlobalScheduler

from util import Loggable
from runner import ExperimentRun


def get_avg_data_ingestion_time(app):
  assert isinstance(app, Appliance)
  return np.mean([c.data_ingestion_time for c in app.containers])


def get_avg_data_egestion_time(app):
  assert isinstance(app, Appliance)
  return np.mean([c.data_egestion_time for c in app.containers])


class SchedulerTest(unittest.TestCase, Loggable):

  def setUp(self):
    self.min_nodes, self.max_nodes = 100, 200
    self.cluster_config = dict(cpus_lo=8, cpus_hi=16,
                               mem_lo=1024 * 16, mem_hi=1024 * 32,
                               disk_lo=1024 * 30, disk_hi=1024 * 60,
                               gpus_lo=4, gpus_hi=8)
    self.app_config = dict(min_cpus=1, max_cpus=8,
                           min_mem=1024, max_mem=1024 * 16,
                           min_disk=0, max_disk=1024 * 30,
                           min_gpus=0, max_gpus=4,
                           min_seq_steps=1, max_seq_steps=5,
                           min_parallel_steps=1, max_parallel_steps=5,
                           min_runtime=80, max_runtime=80 * 10,
                           min_output_nbytes=10, max_output_nbytes=100)
    self.n_iter = 30
    self.seeds = [rnd.randint(0, 100000000) for _ in range(self.n_iter)]
    self.n_apps = 10

  def test_breakdown(self):
    min_nodes, max_nodes = self.min_nodes, self.max_nodes
    cluster_config = dict(self.cluster_config)
    app_config = dict(min_cpus=1, max_cpus=4,
                      min_mem=1024, max_mem=1024 * 4,
                      min_disk=0, max_disk=1024 * 15,
                      min_gpus=0, max_gpus=2,
                      min_seq_steps=1, max_seq_steps=5,
                      min_parallel_steps=1, max_parallel_steps=5,
                      min_runtime=80, max_runtime=800,
                      min_output_nbytes=100, max_output_nbytes=1000)
    # 1-100GB per container
    n_iter, n_apps = 1, 10
    seeds = [rnd.randint(0, 100000000) for _ in range(self.n_iter)]
    root_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = '%s/results/data/breakdown/%d'%(root_dir, time.time())
    os.makedirs(exp_dir, exist_ok=True)
    with open('%s/config.json'%exp_dir, 'w') as f:
      json.dump({
        'clsuter': cluster_config,
        'app': app_config,
        'seeds': seeds
      }, f, indent=2)
    for i, seed in enumerate(seeds):
      start_time = datetime.datetime.now()
      rnd.seed(seed)
      n_nodes = rnd.randint(min_nodes, max_nodes)
      env = simpy.Environment()
      meter = Meter(env)
      print('Iteration: %d, seed: %d' % (i, seed))
      cluster = RandomClusterGenerator(env, seed=seed, meter=meter, **cluster_config).generate(n_nodes)
      app_gen = DataParallelApplianceGenerator(env,
                                               min_parallel_level=n_nodes//4,
                                               max_parallel_level=n_nodes//2,
                                               seed=seed, **app_config)
      apps = [app_gen.generate() for _ in range(n_apps)]
      data_dir = '%s/%d'%(exp_dir, i)
      exps = [
        ExperimentRun('Opportunistic', cluster, apps, OpportunisticGlobalScheduler, data_dir, seed=seed),
        ExperimentRun('FF', cluster, apps, FirstFitGlobalScheduler, data_dir, seed=seed),
        ExperimentRun('FFD', cluster, apps, FirstFitGlobalScheduler, data_dir, decreasing=True, seed=seed),
        ExperimentRun('BF', cluster, apps, BestFitGlobalScheduler, data_dir, seed=seed),
        ExperimentRun('BFD', cluster, apps, BestFitGlobalScheduler, data_dir, decreasing=True, seed=seed),
        ExperimentRun('CA FF', cluster, apps, CostAwareGlobalScheduler, data_dir,
                      bin_pack_algo='first-fit', sort_hosts=True),
        ExperimentRun('CA FFD', cluster, apps, CostAwareGlobalScheduler, data_dir,
                      bin_pack_algo='first-fit', sort_containers=True, sort_hosts=True),
        ExperimentRun('CA FF + host decay', cluster, apps, CostAwareGlobalScheduler, data_dir,
                      bin_pack_algo='first-fit', sort_containers=False, sort_hosts=True, host_decay=True),
        ExperimentRun('CA FFD + host decay', cluster, apps, CostAwareGlobalScheduler, data_dir,
                      bin_pack_algo='first-fit', sort_containers=True, sort_hosts=True, host_decay=True),
        ExperimentRun('CA BF', cluster, apps, CostAwareGlobalScheduler, data_dir,
                      bin_pack_algo='best-fit'),
        ExperimentRun('CA BFD', cluster, apps, CostAwareGlobalScheduler, data_dir,
                      bin_pack_algo='best-fit', sort_containers=True),
        ExperimentRun('CA BF + host decay', cluster, apps, CostAwareGlobalScheduler, data_dir,
                      bin_pack_algo='best-fit', host_decay=True),
        ExperimentRun('CA BFD + host decay', cluster, apps, CostAwareGlobalScheduler, data_dir,
                      bin_pack_algo='best-fit', sort_containers=True, host_decay=True),
      ]
      for e in exps:
        e.start()
      for e in exps:
        e.join()
      self.plot_breakdown(exp_dir)
      self.plot_transfers(exp_dir)
      print('Iteration %d takes %.1f seconds'%(i, (datetime.datetime.now() - start_time).total_seconds()))

  def plot_breakdown(self, data_dir):
    metrics, keys, n_iter = {}, set(), len(os.listdir(data_dir)) - 1
    for iter in os.listdir(data_dir):
      if iter == 'config.json':
        continue
      local_insert = False
      for label in sorted(os.listdir('%s/%s'%(data_dir, iter))):
        with open('%s/%s/%s/general.json'%(data_dir, iter, label), 'r') as f:
          data = json.load(f)
          for k, v in data.items():
            if k == 'avg_local_runtime':
              continue
            if not local_insert:
              for kk in data:
                if kk == 'avg_local_runtime':
                  continue
                metrics.setdefault('Local', {}).setdefault(kk, []).append(
                  data['avg_local_runtime'] if kk == 'avg_runtime' else 0)
              local_insert = True
            metrics.setdefault(label, {}).setdefault(k, []).append(v)
            keys.add(k)
    keys = sorted(keys)
    for k in keys:
      for i in range(n_iter):
        max_val = max([vals[k][i] for vals in metrics.values()])
        for algo in dict(metrics):
          metrics[algo][k][i] /= max_val if max_val else 1
    for k, vals in dict(metrics).items():
      metrics[k] = [np.mean(vals[k]) for k in keys]
    metrics['FF/BF'] = [min(metrics['FF'][i], metrics['BF'][i]) for i in range(len(metrics['BF']))]
    metrics.pop('FF')
    metrics.pop('BF')
    metrics['FFD/BFD'] = [min(metrics['FFD'][i], metrics['BFD'][i]) for i in range(len(metrics['BFD']))]
    metrics.pop('FFD')
    metrics.pop('BFD')
    metrics['CA BF/FF'] = [min(metrics['CA BF'][i], metrics['CA FF'][i]) for i in range(len(metrics['CA BF']))]
    metrics.pop('CA BF')
    metrics.pop('CA FF')
    metrics['CA BF/FF + host decay'] = [min(metrics['CA BF + host decay'][i], metrics['CA FF + host decay'][i])
                                        for i in range(len(metrics['CA BF + host decay']))]
    metrics.pop('CA BF + host decay')
    metrics.pop('CA FF + host decay')
    metrics['CA BFD/FFD'] = [min(metrics['CA BFD'][i], metrics['CA FFD'][i]) for i in range(len(metrics['CA BFD']))]
    metrics.pop('CA BFD')
    metrics.pop('CA FFD')
    metrics['CA BFD/FFD + host decay'] = [min(metrics['CA BFD + host decay'][i], metrics['CA FFD + host decay'][i])
                                        for i in range(len(metrics['CA BFD + host decay']))]
    metrics.pop('CA BFD + host decay')
    metrics.pop('CA FFD + host decay')


    w, gap = .25, .05
    x = np.arange(0, (w + gap) * len(metrics) * len(keys), (w + gap) * len(metrics))
    for i, (k, v) in enumerate(metrics.items()):
      plt.bar(x + w * i, v, width=w, label=k)
      plt.xticks(x + w * len(metrics) / 2, keys, fontsize=9, rotation=45)
      plt.ylim(0, 1.5)
      plt.ylabel('Time/cost (seconds/$) norm. to max.')
    plt.legend(ncol=2, frameon=False)
    plt.show()

  def plot_transfers(self, data_dir, normed=True):
    metrics, n_iter = {}, len(os.listdir(data_dir)) - 1
    for iter in os.listdir(data_dir):
      if iter == 'config.json':
        continue
      for label in sorted(os.listdir('%s/%s'%(data_dir, iter))):
        with open('%s/%s/%s/transfers.json'%(data_dir, iter, label), 'r') as f:
          data = json.load(f)
          avg_prop_delay = np.mean([prop_delay for total_delay, prop_delay, _, _ in data])
          avg_queue_delay = np.mean([total_delay - prop_delay for total_delay, prop_delay, _, _ in data])
        # with open('%s/%s/%s/transfers.json' % (data_dir, iter, label), 'r') as f:
        #   data = json.load(f)
        #   turnover_delay = len(data['turnovers']) * 5/data['total_scheduling_ops']
          metrics.setdefault(label, []).append([avg_prop_delay, avg_queue_delay])

    metrics['FF/BF'] = [min(metrics['FF'][i], metrics['BF'][i]) for i in range(len(metrics['BF']))]
    metrics.pop('FF')
    metrics.pop('BF')
    metrics['FFD/BFD'] = [min(metrics['FFD'][i], metrics['BFD'][i]) for i in range(len(metrics['BFD']))]
    metrics.pop('FFD')
    metrics.pop('BFD')
    metrics['CA BF/FF'] = [min(metrics['CA BF'][i], metrics['CA FF'][i]) for i in range(len(metrics['CA BF']))]
    metrics.pop('CA BF')
    metrics.pop('CA FF')
    metrics['CA BF/FF + host decay'] = [min(metrics['CA BF + host decay'][i], metrics['CA FF + host decay'][i])
                                        for i in range(len(metrics['CA BF + host decay']))]
    metrics.pop('CA BF + host decay')
    metrics.pop('CA FF + host decay')
    metrics['CA BFD/FFD'] = [min(metrics['CA BFD'][i], metrics['CA FFD'][i]) for i in range(len(metrics['CA BFD']))]
    metrics.pop('CA BFD')
    metrics.pop('CA FFD')
    metrics['CA BFD/FFD + host decay'] = [min(metrics['CA BFD + host decay'][i], metrics['CA FFD + host decay'][i])
                                          for i in range(len(metrics['CA BFD + host decay']))]
    metrics.pop('CA BFD + host decay')
    metrics.pop('CA FFD + host decay')
    x = defaultdict(list)
    labels, yticks = ['Propagation delay', 'Queueing delay', 'Scheduling turnover delay'], sorted(metrics.keys())
    if normed:
      for label, vals in metrics.items():
        max_val = max([sum([v for v in vals[i]]) for i in range(n_iter)])
        for i in range(n_iter):
          for j in range(len(vals[0])):
            vals[i][j] /= max_val if max_val > 0 else 1
    for k, vals in dict(metrics).items():
      metrics[k] = [np.mean([vals[j][i] for j in range(n_iter)]) for i in range(len(vals[0]))]
    for l in yticks:
      for i in range(len(metrics[l])):
        x[i] += metrics[l][i],
    height, gap = .25, .05
    y = np.arange(0, (height + gap) * len(yticks) - 10 ** -6, height + gap)
    for i in sorted(x):
      plt.barh(y, x[i], height=height, left=np.zeros(len(x[i])) if i == 0 else x[i - 1], label=labels[i])
    plt.yticks(y, yticks)
    if normed:
      plt.xlabel('Data transfer time per task norm. to max')
    else:
      plt.xlabel('Data transfer time per task')
    plt.legend()
    plt.show()

  def plot_placement(self, data_dir):
    metrics, n_iter = defaultdict(list), len(os.listdir(data_dir))
    for iter in os.listdir(data_dir):
      for label in sorted(os.listdir('%s/%s' % (data_dir, iter))):
        with open('%s/%s/%s/transfers.json' % (data_dir, iter, label), 'r') as f:
          data = json.load(f)
          avg_bw = np.mean([8000/avg_bw for _, _, avg_bw, _ in data])
          avg_cost = np.mean([avg_cost for _, _, _, avg_cost in data])
          metrics.setdefault(label, []).append((avg_bw, avg_cost))

    def cmp(p):
      return np.sqrt(p[0] ** 2 + p[1] ** 2)

    # metrics['FF/BF'] = [min(metrics['FF'][i], metrics['BF'][i], key=cmp) for i in range(len(metrics['BF']))]
    metrics['FF/BF'] = metrics['FF'] + metrics['BF']
    metrics.pop('FF')
    metrics.pop('BF')
    # metrics['FFD/BFD'] = [min(metrics['FFD'][i], metrics['BFD'][i], key=cmp) for i in range(len(metrics['BFD']))]
    metrics['FFD/BFD'] = metrics['FFD'] + metrics['BFD']
    metrics.pop('FFD')
    metrics.pop('BFD')
    # metrics['CA BF/FF'] = [min(metrics['CA BF'][i], metrics['CA FF'][i], key=cmp) for i in range(len(metrics['CA BF']))]
    metrics['CA BF/FF'] = metrics['CA BF'] + metrics['CA FF']
    metrics.pop('CA BF')
    metrics.pop('CA FF')
    # metrics['CA BF/FF + host decay'] = [min(metrics['CA BF + host decay'][i], metrics['CA FF + host decay'][i], key=cmp)
    #                                     for i in range(len(metrics['CA BF + host decay']))]
    metrics['CA BF/FF + host decay'] = metrics['CA BF + host decay'] + metrics['CA FF + host decay']
    metrics.pop('CA BF + host decay')
    metrics.pop('CA FF + host decay')
    # metrics['CA BFD/FFD'] = [min(metrics['CA BFD'][i], metrics['CA FFD'][i], key=cmp)
    #                          for i in range(len(metrics['CA BFD']))]
    metrics['CA BFD/FFD'] = metrics['CA BFD']+ metrics['CA FFD']
    metrics.pop('CA BFD')
    metrics.pop('CA FFD')
    # metrics['CA BFD/FFD + host decay'] = [min(metrics['CA BFD + host decay'][i], metrics['CA FFD + host decay'][i], key=cmp)
    #                                       for i in range(len(metrics['CA BFD + host decay']))]
    metrics['CA BFD/FFD + host decay'] = metrics['CA BFD + host decay'] + metrics['CA FFD + host decay']
    metrics.pop('CA BFD + host decay')
    metrics.pop('CA FFD + host decay')
    cluster = defaultdict(dict)
    for l, points in metrics.items():
      for p in points:
        if not cluster[l]:
          cluster[l][p] = 1
          continue
        anchors = list(cluster[l].keys())
        dists = [np.sqrt((p[0] - pp[0]) ** 2 + (p[1] - pp[1]) ** 2) for pp in anchors]
        if min(dists) > 1:
          cluster[l][p] = 1
        else:
          anchor = anchors[np.argmin(dists)]
          cluster[l][anchor] += 1
    for l, v in dict(cluster).items():
      metrics[l] = [sorted(v.keys()), [v[k] for k in sorted(v.keys())]]
    print(metrics)
    for l, (points, cnt) in metrics.items():
      plt.scatter([p[0] for p in points], [p[1] for p in points], np.array(cnt) * 8, label=l)
    plt.xlabel('Bandwidth (seconds/GB)')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

  def test_plot_breakdown(self):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    self.plot_breakdown('%s/results/data/breakdown/1548053930'%root_dir)

  def test_plot_data_transfer(self):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = '%s/results/data/breakdown/1548053930'%root_dir
    self.plot_transfers(exp_dir)
    self.plot_transfers(exp_dir, normed=False)

  def test_plot_placement(self):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = '%s/results/data/breakdown/1548053930' % root_dir
    self.plot_placement(exp_dir)


  def test_temp(self):
    import json

    containers = []
    wf = dict(id='alignment', containers=containers)
    containers += dict(id='pre-align', type='job', image='heliumdatacommons/cwl-wrapper',
                       resources=dict(cpus=2, mem=2048, disk=8096), cmd='pre-align'),
    containers += dict(id='cwl-scatter-0', type='job', image='heliumdatacommons/cwl-wrapper',
                       resources=dict(cpus=2, mem=2048, disk=8096), cmd='scatter',
                       dependencies=['pre-align']),
    containers += dict(id='align', type='job', image='heliumdatacommons/cwl-wrapper',
                       resources=dict(cpus=2, mem=2048, disk=8096), cmd='align',
                       instances=10, dependencies=['cwl-scatter']),
    containers += dict(id='cwl-gather-0', type='job', image='heliumdatacommons/cwl-wrapper',
                       resources=dict(cpus=2, mem=2048, disk=8096), cmd='gather',
                       dependencies=['align']),
    containers += dict(id='cwl-scatter-1', type='job', image='heliumdatacommons/cwl-wrapper',
                       resources=dict(cpus=2, mem=2048, disk=8096), cmd='scatter',
                       dependencies=['cwl-gather-0']),
    containers += dict(id='samtool-sort', type='job', image='heliumdatacommons/cwl-wrapper',
                       resources=dict(cpus=2, mem=2048, disk=8096), cmd='samtool-sort',
                       instances=10, dependencies=['cwl-scatter-1']),
    containers += dict(id='cwl-gather-1', type='job', image='heliumdatacommons/cwl-wrapper',
                       resources=dict(cpus=2, mem=2048, disk=8096), cmd='gather',
                       dependencies=['samtool-sort']),
    containers += dict(id='post-align', type='job', image='heliumdatacommons/cwl-wrapper',
                       resources=dict(cpus=2, mem=2048, disk=8096), cmd='post-align',
                       dependencies=['cwl-gather-1']),
    print(json.dumps(wf, indent=2))

