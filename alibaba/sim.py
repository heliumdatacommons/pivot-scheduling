import os
import json
import time
import simpy
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from collections import defaultdict
from argparse import ArgumentParser

from resources.gen import RandomClusterGenerator
from resources.meter import Meter
from alibaba.runner import ExperimentRun
from scheduler.opportunistic import OpportunisticGlobalScheduler
from scheduler.vbp import FirstFitGlobalScheduler
from scheduler.cost_aware import CostAwareGlobalScheduler


def parse_args():
  parser = ArgumentParser(description='Run simulation on Alibaba cluster trace')
  parsers = parser.add_subparsers(help='Experiment type', dest='command')
  parser.add_argument('--num-hosts', type=int, dest='n_hosts', default=600,
                      help='Number of hosts')
  parser.add_argument('--cpus', type=int, dest='cpus', default=16,
                      help='Number of CPUs per host')
  parser.add_argument('--mem', type=int, dest='mem', default=128 * 1024,
                      help='RAM in MBs per host')
  parser.add_argument('--disk', type=int, dest='disk', default=100,
                      help='Disk space in GBs per host')
  parser.add_argument('--gpus', type=int, dest='gpus', default=1,
                      help='Number of GPU units per host')
  parser.add_argument('--job-dir', type=str, dest='job_dir', default=os.environ.get('JOB_DIR', './jobs'),
                      help='Batch job directory')
  parser.add_argument('--output-dir', type=str, dest='output_dir', default=os.environ.get('OUTPUT_DIR', './output'),
                      help='Output directory for results')
  parser.add_argument('--task-output-scale-factor', type=float, dest='output_scale_factor', default=1000,
                      help='Scale factor of the output data size of tasks (proportional to the memory demand)')
  overall_parser = parsers.add_parser('overall', help='Run the overall experiment')
  overall_parser.add_argument('--num-apps', type=int, dest='num_apps',
                              help='Number of applications')
  n_app_parser = parsers.add_parser('num-apps',
                                    help='Run the experiment with varying number of applications')
  n_app_parser.add_argument('--host-hourly-rate', type=float, dest='host_hourly_rate', default=.932,
                            help='Host hourly rate')
  n_app_parser.add_argument('--num-apps-list', nargs='+', type=int, dest='num_apps_list', required=True,
                            help='Number of applications')
  args = parser.parse_args()
  if args.command is None:
    parser.print_help()
    parser.exit(1)
  return args


def plot_overall(exp_dir):
  data_dir, plot_dir = '%s/data'%exp_dir, '%s/plot'%exp_dir
  os.makedirs(plot_dir, exist_ok=True)
  metrics, keys, n_iter = {}, set(), len(os.listdir(data_dir))
  for iter in sorted(os.listdir(data_dir)):
    for label in sorted(os.listdir('%s/%s'%(data_dir, iter))):
      with open('%s/%s/%s/general.json'%(data_dir, iter, label), 'r') as f:
        data = json.load(f)
        for k, v in data.items():
          metrics.setdefault(label, {}).setdefault(k, []).append(v)
          keys.add(k)
  keys = sorted(keys, key=lambda x: ['egress_cost',
                                     'cum_instance_hours',
                                     'avg_runtime'].index(x))
  for k in keys:
    for i in range(n_iter):
      max_val = max([vals[k][i] for vals in metrics.values()])
      for algo in dict(metrics):
        metrics[algo][k][i] /= max_val if max_val else 1
  for algo, vals in dict(metrics).items():
    metrics[algo] = [np.mean(vals[k]) for k in keys]

  w, gap = .25, .1
  hatches = ['/', '+', '-']
  xlabels = ['egress cost', 'host cost', 'app. runtime']
  x = np.arange(0, (w + gap) * len(metrics) * len(keys), (w + gap) * len(metrics))
  plt.figure(figsize=(7, 4))

  for i, (k, v) in enumerate(sorted(metrics.items(),
                                    key=lambda x: ['Opportunistic', 'Cost-Aware', 'VBP'].index(x[0]))):
    plt.bar(x + w * i, v, width=w, label=k, hatch=hatches[i])
    plt.xticks(x + w * len(metrics) / 2 - gap, xlabels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 1.15)
    plt.ylabel('Cost/runtime norm. to max.', fontsize=14)
  plt.legend(ncol=3, frameon=False, fontsize=14)
  plt.tight_layout()
  plt.savefig('%s/overall.pdf'%plot_dir, format='pdf')


def plot_transfers(exp_dir):
  data_dir, plot_dir = '%s/data'%exp_dir, '%s/plot'%exp_dir
  os.makedirs(plot_dir, exist_ok=True)
  metrics, n_iter = {}, len(os.listdir(data_dir))
  for iter in os.listdir(data_dir):
    for label in sorted(os.listdir('%s/%s'%(data_dir, iter))):
      with open('%s/%s/%s/transfers.json'%(data_dir, iter, label), 'r') as f:
        data = json.load(f)
        avg_prop_delay = np.mean([t['propagation_delay']for t in data])
        avg_queue_delay = np.mean([t['total_delay'] - t['propagation_delay'] for t in data])
      metrics.setdefault(label, []).append([avg_prop_delay, avg_queue_delay])
  x = defaultdict(list)
  labels = ['Transmission', 'Congestion', 'Scheduling turnover delay']
  yticks = ['Opportunistic', 'Cost-Aware', 'VBP']
  for k, vals in metrics.items():
    vals = metrics[k]
    metrics[k] = [np.mean([vals[j][i] for j in range(n_iter)]) for i in range(len(vals[0]))]
  for l in yticks:
    for i in range(len(metrics[l])):
      x[i] += metrics[l][i],
  height, gap = .20, .05
  y = np.arange(0, (height + gap) * len(yticks) - 10 ** -6, height + gap)
  x = np.array(list(x.values()))
  hatches = ['/', '-']
  plt.figure(figsize=(7, 3))
  cumsum = np.zeros(len(x[0]))
  for i, row in enumerate(x):
    plt.barh(y, row, height=height, left=cumsum, hatch=hatches[i], label=labels[i])
    cumsum += row
  plt.xticks(fontsize=14)
  plt.yticks(y, yticks, rotation=45, fontsize=14)
  plt.xlabel('Data transfer time per task (seconds)', fontsize=14)
  plt.legend(ncol=2, frameon=False, fontsize=14)
  plt.tight_layout()
  plt.savefig('%s/transfer.pdf'%plot_dir, format='pdf')


def plot_financial_cost(exp_dir, host_hourly_rate):
  data_dir, plot_dir = '%s/data'%exp_dir, '%s/plot'%exp_dir
  os.makedirs(plot_dir, exist_ok=True)
  metrics = {}
  for n_apps in sorted(filter(lambda d: os.path.isdir('%s/%s'%(data_dir, d)), os.listdir(data_dir))):
    for iter in os.listdir('%s/%s'%(data_dir, n_apps)):
      for label in os.listdir('%s/%s/%s'%(data_dir, n_apps, iter)):
        with open('%s/%s/%s/%s/general.json'%(data_dir, n_apps, iter, label)) as f:
          data = json.load(f)
          egress_cost = data['egress_cost']
          instance_cost = data['cum_instance_hours'] * host_hourly_rate
          metrics.setdefault(label, {}).setdefault(int(n_apps), []).append((egress_cost, instance_cost))
  for label, num_apps in dict(metrics).items():
    for n_apps, vals in dict(num_apps).items():
      num_apps[n_apps] = np.mean([vs[0] for vs in vals]), np.mean([vs[1] for vs in vals])
    metrics[label] = sorted(num_apps.keys()), [num_apps[ds][0] for ds in sorted(num_apps)], [num_apps[ds][1] for ds in sorted(num_apps)]
  markers = ['x', '+', '1']
  colors = []
  plt.figure(figsize=(8, 5))
  for i, (label, (xticks, egress_cost, instance_cost)) in enumerate(metrics.items()):
    l, = plt.plot(xticks, np.array(egress_cost)/1000, ls='--',
                  marker=markers[i], markersize=15, label='%s (egress)'%label)
    colors += l.get_color(),
  for i, (label, (xticks, egress_cost, instance_cost)) in enumerate(metrics.items()):
    plt.plot(xticks, np.array(instance_cost)/1000, color=colors[i],
             marker=markers[i], markersize=15, label='%s (host)'%label)
  plt.xticks(xticks, [x if idx%2 == 0 else '' for idx, x in enumerate(xticks)], fontsize=14)
  plt.yticks(fontsize=14)
  plt.ylim(-0.1, plt.ylim()[1] * 1.05)
  plt.xlabel('# of running applications', fontsize=14)
  plt.ylabel('Total host/egress cost ($1K)', fontsize=14)
  plt.legend(ncol=2, frameon=False, fontsize=14)
  plt.tight_layout()
  plt.savefig('%s/cost.pdf'%plot_dir, format='pdf')


def run_experiment_overall(n_hosts, cpus, mem, disk, gpus, output_scale_factor, n_apps, job_dir, exp_dir=None):
  root_dir = os.path.dirname(os.path.abspath(__file__)) if exp_dir is None else exp_dir
  exp_dir = '%s/overall/%d'%(root_dir, time.time())
  os.makedirs(exp_dir, exist_ok=True)
  env = simpy.Environment()
  meter = Meter(env)
  cluster = RandomClusterGenerator(env, cpus, cpus, mem, mem, disk, disk, gpus, gpus, meter=meter).generate(n_hosts)
  loads = ['%s/%s'%(job_dir, job_f) for job_f in os.listdir(job_dir)]
  all_exps = []
  for i, load_f in enumerate(loads):
    data_dir = '%s/data/%d'%(exp_dir, i)
    exps = [
      ExperimentRun('Opportunistic', cluster, OpportunisticGlobalScheduler, load_f, data_dir,
                    output_scale_factor, n_apps),
      ExperimentRun('VBP', cluster, FirstFitGlobalScheduler, load_f, data_dir, output_scale_factor, n_apps,
                    decreasing=True),
      ExperimentRun('Cost-Aware', cluster, CostAwareGlobalScheduler, load_f, data_dir, output_scale_factor, n_apps,
                    bin_pack_algo='first-fit', sort_tasks=True, sort_hosts=True),
    ]
    for e in exps:
      e.start()
      all_exps += e,
      if len(all_exps) == mp.cpu_count():
        for ee in all_exps:
          ee.join()
        all_exps.clear()
  for e in all_exps:
    e.join()
  return exp_dir


def run_experiment_n_apps(n_hosts, cpus, mem, disk, gpus, output_scale_factor, n_apps, job_dir, exp_dir=None):
  root_dir = os.path.dirname(os.path.abspath(__file__)) if exp_dir is None else exp_dir
  exp_dir = '%s/n_app/%d'%(root_dir, time.time())
  os.makedirs(exp_dir, exist_ok=True)
  env = simpy.Environment()
  meter = Meter(env)
  cluster = RandomClusterGenerator(env, cpus, cpus, mem, mem, disk, disk, gpus, gpus, meter=meter).generate(n_hosts)
  loads = ['%s/%s'%(job_dir, job_f) for job_f in os.listdir(job_dir)]
  all_exps = []
  for n_app in n_apps:
    print('Run %d jobs'%n_app)
    n_app_dir = '%s/data/%d'%(exp_dir, n_app)
    for i, load_f in enumerate(loads):
      data_dir = '%s/%d'%(n_app_dir, i)
      exps = [
          ExperimentRun('Opportunistic', cluster, OpportunisticGlobalScheduler, load_f, data_dir,
                        output_scale_factor, n_app),
          ExperimentRun('VBP', cluster, FirstFitGlobalScheduler, load_f, data_dir, output_scale_factor, n_app,
                        decreasing=True),
          ExperimentRun('Cost-Aware', cluster, CostAwareGlobalScheduler, load_f, data_dir, output_scale_factor, n_app,
                        bin_pack_algo='first-fit', sort_tasks=True, sort_hosts=True),
      ]
      for e in exps:
        e.start()
        all_exps += e,
        if len(all_exps) == mp.cpu_count():
          for ee in all_exps:
            ee.join()
          all_exps.clear()
  for e in all_exps:
    e.join()
  return exp_dir


if __name__ == '__main__':
  args = parse_args()
  n_hosts, cpus, mem, disk, gpus = args.n_hosts, args.cpus, args.mem, args.disk, args.gpus
  job_dir, exp_dir = args.job_dir, args.output_dir
  output_scale_factor = args.output_scale_factor
  if args.command == 'overall':
    exp_dir = run_experiment_overall(n_hosts, cpus, mem, disk, gpus, output_scale_factor,
                                     args.num_apps, job_dir, exp_dir)
    plot_overall(exp_dir)
    plot_transfers(exp_dir)
  else:
    exp_dir = run_experiment_n_apps(n_hosts, cpus, mem, disk, gpus, output_scale_factor,
                                    args.num_apps_list, job_dir, exp_dir)
    plot_financial_cost(exp_dir, args.host_hourly_rate)



