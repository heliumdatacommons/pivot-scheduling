import os
import sys
import yaml
import datetime

import numpy as np

from collections import OrderedDict
from argparse import ArgumentParser


def parse_args():
  parser = ArgumentParser(description='Script for sampling batch jobs from Alibaba cluster trace dataset')
  parser.add_argument('--num-jobs', '-n', type=int, required=True, dest='n_jobs',
                      help='Number of sampled jobs')
  parser.add_argument('--min-runtime', '-l', type=int, default=60, dest='min_runtime',
                      help='Minimum runtime')
  parser.add_argument('--max-runtime', '-u', type=int, default=1000, dest='max_runtime',
                      help='Maximum runtime')
  parser.add_argument('--start', '-s', type=int, required=True, dest='start',
                      help='Start timestamp of the sampling')
  parser.add_argument('--interval', '-i', type=int, required=True, dest='interval',
                      help='Interval of the sampling')
  parser.add_argument('--min-deps', '-d', type=int, default=1, dest='min_deps',
                      help='Minimum number of tasks with dependencies in a job')
  parser.add_argument('--max-parallel', '-p', type=int, default=100, dest='max_parallel',
                      help='Maximum level of parallelism of tasks')
  parser.add_argument('--output-dir', '-o', type=str, required=True, dest='output_dir',
                      help='Output directory of the sample data')
  args = parser.parse_args()
  return args


def load_machine_data(machine_meta_f):
  machines, cpus, mem = set(), None, None
  with open(machine_meta_f) as f:
    for l in f.readlines():
      machine_id, _, _, _, cpus, mem, _ = l[:-1].split(',')
      machines.add(machine_id)
      if not cpus:
        cpus = int(cpus)
      if not mem:
        mem = int(mem)
  return machines, cpus, mem


def load_batch_task_data(batch_task_f):
  jobs = {}
  with open(batch_task_f) as f:
    for l in f:
      t_name, n_inst, j_name, t_type, status, start_time, end_time, cpus, mem = l[:-1].split(',')
      if not t_name or not j_name or not cpus or not mem or not start_time or not end_time:
        continue
      if status == 'Failed':
        jobs.pop(j_name, None)
        continue
      start_time, end_time, n_inst, cpus, mem = int(start_time), int(end_time), int(n_inst), float(cpus)/100, float(mem)
      job = jobs.setdefault(j_name, dict(id=j_name, tasks={}))
      job['submit_time'] = min(job.setdefault('submit_time', start_time), start_time)
      job['finish_time'] = max(job.setdefault('finish_time', end_time), end_time)
      if t_name.startswith('task') or t_name == 'MergeTask':
        task_id, deps = t_name, []
      else:
        splits = [t for t in t_name[1:].strip('\n').split('_') if t and not t.startswith('Stg')]
        task_id, deps = int(splits[0]), [int(d) for d in splits[1:]]
      try:
        job['tasks'][task_id] = dict(id=task_id, cpus=cpus, mem=mem, start_time=start_time, end_time=end_time,
                                     n_instances=n_inst, dependencies=deps)
      except KeyError:
        jobs.pop(j_name)
  return jobs


def load_batch_instance_data(batch_instance_f, jobs, n_jobs, sampling_start, sampling_interval,
                             min_runtime=60, max_runtime=1000, min_deps=1, max_parallel=100):
  selected, excluded = {}, set()
  with open(batch_instance_f) as f:
    cur_job = None
    for l in f:
      _, t_name, j_name, _, status, start_time, end_time, machine_id, _, _, _, _, _, _ = l[:-1].split(',')
      if not t_name or not j_name or j_name in excluded or j_name not in jobs or not status or not start_time or not end_time or not machine_id:
        continue
      if status == 'Failed':
        continue
      start_time, end_time = int(start_time), int(end_time)
      if start_time <= 0 or end_time <= 0 or start_time >= end_time or end_time - start_time > max_runtime:
        excluded.add(j_name)
        for b in selected.values():
          b.pop(j_name, None)
        continue
      job = jobs[j_name]
      max_instances = max([t['n_instances'] for t in job['tasks'].values()])
      n_deps = len([t for t in job['tasks'].values() if len(t['dependencies']) > 0])
      if max_instances > max_parallel or n_deps < min_deps:
        excluded.add(j_name)
        continue
      if not cur_job:
        cur_job = job
      elif cur_job != job:
        min_start_time = min([t['start_time'] for t in cur_job['tasks'].values()])
        max_end_time = max([t['end_time'] for t in cur_job['tasks'].values()])
        job['submit_time'] = min(job['submit_time'], min_start_time)
        job['finish_time'] = max(job['finish_time'], max_end_time)
        if sampling_start < min_start_time < max_end_time and max_end_time - min_start_time >= min_runtime:
          key = min_start_time // sampling_interval * sampling_interval
          if any(['runtime' not in t or t['start_time'] >= t['end_time'] for t in dict(cur_job['tasks']).values()]) \
              or any([d not in cur_job['tasks'] for t in cur_job['tasks'].values() for d in t['dependencies']]):
            excluded.add(cur_job['id'])
            if key in selected:
              selected[key].pop(cur_job['id'], None)
          elif key not in selected or len(selected[key]) < n_jobs:
            cur_job['tasks'] = [{k: v for k, v in t.items() if k not in ('start_time', 'end_time')}
                                for t in dict(cur_job['tasks']).values()]
            selected.setdefault(key, OrderedDict())[cur_job['id']] = cur_job
            write('\rSampled jobs: %s'%{k: len(v) for k, v in sorted(selected.items())})
        cur_job = job
      try:
        task_id = t_name if t_name.startswith('task') or t_name == 'MergeTask' else int(t_name[1:].split('_')[0])
        task = job['tasks'][task_id]
        task['start_time'], task['end_time'], task['runtime'] = start_time, end_time, end_time - start_time
      except KeyError:
        excluded.add(j_name)
        cur_job = None
      if len(selected) > 0 and all([len(b) == n_jobs for b in selected.values()]):
        break
  write('\n')
  return selected


def load_service_data(container_meta_f):
  apps = {}
  with open(container_meta_f) as f:
    cur_contr = None
    for l in f:
      contr_id, machine_id, _, app_id, status, cpus, _, mem = l[:-1].split(',')
      if not contr_id or not machine_id or not status or not cpus or not mem:
        continue
      cpus, mem = float(cpus)/100, float(mem)
      if cur_contr is None:
        cur_contr = dict(id=contr_id, application=app_id, cpus=cpus, mem=mem, placement=machine_id, statuses=[status])
      elif cur_contr['id'] != contr_id:
        app = apps.setdefault(cur_contr['application'], dict(id=cur_contr['application'], tasks=[]))
        app['tasks'] += cur_contr,
        cur_contr = dict(id=contr_id, application=app_id, cpus=cpus, mem=mem, placement=machine_id, statuses=[status])
      if status not in cur_contr['statuses']:
        cur_contr['statuses'] += status,
    app = apps.setdefault(cur_contr['application'], dict(id=cur_contr['application'], tasks=[]))
    app['tasks'] += cur_contr,
  return apps


def collect_task_runtime(instances):
  res = {}
  for runs in instances.values():
    for task_id, job_id, start_time, end_time, _ in runs:
      metrics = res.setdefault(job_id, {}).setdefault(task_id, dict(start_time=[], end_time=[], runtime=[]))
      metrics['start_time'] += start_time,
      metrics['end_time'] += end_time,
      metrics['runtime'] += end_time - start_time,
  for tasks in res.values():
    for task_id in dict(tasks):
      tasks[task_id]['start_time'] = int(np.min(tasks[task_id]['start_time']))
      tasks[task_id]['end_time'] = int(np.max(tasks[task_id]['end_time']))
      tasks[task_id]['runtime'] = round(float(np.mean(tasks[task_id]['runtime'])), 2)
  return res


def collect_hosts(instances):
  return set([instance[-1] for tasks in instances.values() for instance in tasks])


def write(text):
  sys.stdout.write(text)
  sys.stdout.flush()


if __name__ == '__main__':
  # load_machine_data('alibaba/machine_meta.csv')
  args = parse_args()
  output_dir, n_jobs = args.output_dir, args.n_jobs
  sampling_start, sampling_interval = args.start, args.interval
  max_parallel = args.max_parallel
  start_time = datetime.datetime.now()
  os.makedirs(output_dir, exist_ok=True)
  write('Load job data... ')
  jobs = load_batch_task_data('csv/batch_task.csv')
  write('(%.2f seconds)\n'%(datetime.datetime.now() - start_time).total_seconds())
  start_time = datetime.datetime.now()
  write('Sample %d jobs submitted after %d ...\n'%(n_jobs, sampling_start))
  jobs = load_batch_instance_data('csv/batch_instance.csv', jobs, n_jobs,
                                  sampling_start, sampling_interval,
                                  min_runtime=args.min_runtime, max_runtime=args.max_runtime,
                                  min_deps=args.min_deps, max_parallel=args.max_parallel)
  write('(%.2f seconds)\n'%(datetime.datetime.now() - start_time).total_seconds())
  start_time = datetime.datetime.now()
  write('Save job data (%d periods)... '%len(jobs))
  for s, j in jobs.items():
    with open('%s/jobs-%d-%d-%d-%d.yaml'%(output_dir, n_jobs, max_parallel, s, s + sampling_interval), 'w') as f:
      yaml.dump(list(j.values()), f, default_flow_style=False)
  write('(%.2f seconds)\n' % (datetime.datetime.now() - start_time).total_seconds())
  # apps = load_service_data('csv/container_meta.csv')
  # write('Save service data (%d services)\n'%len(apps))
  # with open('apps.yaml', 'w') as f:
  #   yaml.dump(apps, f, default_flow_style=False)
  # write('Save instance data\n')
  # with open('instances.csv', 'w') as f:
  #   for runs in instances.values():
  #     for run in runs:
  #       f.write(','.join([str(r) for r in run]) + '\n')
  # hosts = collect_hosts(instances)
  # write('Save host data (%d hosts)\n'%len(hosts))
  # with open('hosts.csv', 'w') as f:
  #   f.write(','.join(hosts))


