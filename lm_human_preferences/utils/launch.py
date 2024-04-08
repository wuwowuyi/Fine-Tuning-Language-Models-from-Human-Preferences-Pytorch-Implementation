import os
import time
from pathlib import Path

import fire
import torch
import wandb
from torch.distributed import init_process_group

from lm_human_preferences.utils import hyperparams


# def launch(name, f, *, namespace='safety', mode='local') -> None:
#     if mode == 'local':
#         with open('/tmp/pickle_fn', 'wb') as file:
#             cloudpickle.dump(f, file)
#
#         subprocess.check_call(['python', '-c', 'import sys; import pickle; pickle.loads(open("/tmp/pickle_fn", "rb").read())()'])
#         return
#     raise Exception('Other modes unimplemented!')

# def parallel(jobs, mode):
#     if mode == 'local':
#         assert len(jobs) == 1, "Cannot run jobs in parallel locally"
#         for job in jobs:
#             job()
#     else:
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             futures = [executor.submit(job) for job in jobs]
#             for f in futures:
#                 f.result()

def params(trial, hparam_class, extra_hparams=None):
    descriptors = []
    kwargs = {}
    for k, v, s in trial:
        if k is not None:
            if k in kwargs:
                print(f'WARNING: overriding key {k} from {kwargs[k]} to {v}')
            kwargs[k] = v
        if s.get('descriptor'):
            descriptors.append(str(s['descriptor']))
    hparams = hparam_class()
    hparams.override_from_dict(kwargs)
    if extra_hparams:
        hparams.override_from_str_dict(extra_hparams)
    hparams.validate()
    return hparams, descriptors


def launch_trials(name, fn, trials, hparam_class, extra_hparams=None, dry_run=False):
    for trial in trials:  # each trial is a group of hparams
        hparams, descriptors = params(trial, hparam_class, extra_hparams)
        if dry_run:
            hyperparams.dump(hparams)  # output hparams to out (default to stdout)
        else:
            job_name = (name + '/' + '-'.join(descriptors)).rstrip('/')

            # setup ddp
            if hparams.run.ddp:
                init_process_group(backend=hparams.run.ddp_backend)  # default to nccl for GPU training
                torch.cuda.set_device(hparams.run.device)  # set default device for current process
            else:
                print('\nDDP is not enabled.\n')

            hparams.run.save_dir = Path(hparams.run.save_dir) / job_name
            hparams.run.labels_dir = Path(hparams.run.labels_dir)
            if hparams.run.master_process:
                Path.mkdir(hparams.run.save_dir, exist_ok=True)
                Path.mkdir(hparams.run.labels_dir, exist_ok=True)
                if hparams.run.wandb_log:
                    wandb_run_name = f'{job_name}-' + str(time.time())
                    wandb.init(project=hparams.run.wandb_project, name=wandb_run_name, config=hparams.to_nested_dict())

            fn(hparams)


def main(commands_dict):
    """Similar to fire.Fire, but with support for multiple commands without having a class."""
    class _Commands:
        def __init__(self):
            for name, cmd in commands_dict.items():
                setattr(self, name, cmd)
    fire.Fire(_Commands)
