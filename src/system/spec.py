from collections import defaultdict
from jittor import optim
from typing import Dict, List, Optional
from tqdm import tqdm

import jittor as jt
import os

from ..data.asset import Asset
from ..data.dataset import PCDatasetModule
from ..model.spec import ModelSpec

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def get_rank() -> int:
    """MPI rank (0 when running single-process)."""
    try:
        return jt.rank
    except Exception:
        return 0


def get_world_size() -> int:
    try:
        return jt.world_size
    except Exception:
        return 1


def is_main_process() -> bool:
    return get_rank() == 0


def _wandb_log(data: dict, step: Optional[int] = None):
    """Log to Weights & Biases if a run is active (rank-0 only); else a no-op."""
    if _HAS_WANDB and wandb.run is not None and is_main_process():
        wandb.log(data, step=step)


def gpu_stats() -> dict:
    """Query per-GPU utilisation / memory via nvidia-smi for wandb logging.

    Returns a flat dict like {'gpu/0/util': 97, 'gpu/0/mem_used_mb': 1234, ...}
    plus averages. Best-effort: returns {} if nvidia-smi is unavailable.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            timeout=5,
        ).decode()
    except Exception:
        return {}
    stats, utils, mems = {}, [], []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
        idx, util, mem = parts[0], float(parts[1]), float(parts[2])
        stats[f'gpu/{idx}/util'] = util
        stats[f'gpu/{idx}/mem_used_mb'] = mem
        utils.append(util)
        mems.append(mem)
    if utils:
        stats['gpu/util_mean'] = sum(utils) / len(utils)
        stats['gpu/mem_used_mb_mean'] = sum(mems) / len(mems)
    return stats

def _get_item(x):
    if isinstance(x, jt.Var):
        return x.item()
    return x

def get_optimizer(optimizer_config, model):
    __target__ = optimizer_config.pop('__target__')
    MAPPING = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
    }
    if __target__ not in MAPPING:
        raise ValueError(f"unsupported optimizer: {__target__}")
    OptimizerClass = MAPPING[__target__]
    optimizer = OptimizerClass(model.parameters(), **optimizer_config)
    return optimizer

class DummyWriter():
    
    def __init__(self):
        pass
    
    def write(self, batch, prediction: List[Dict], dataset_module: Optional[PCDatasetModule]=None):
        pass

class DummySystem():
    
    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: ModelSpec,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
    ):
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = loss_config
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.writer = writer
        if trainer_config is None:
            trainer_config = {}
        self.epochs = trainer_config.get('epochs', 1)
        self.log_interval = trainer_config.get('log_interval', 10)
        self.warmup_epochs = trainer_config.get('warmup_epochs', 0)
        self.grad_clip = trainer_config.get('grad_clip', 0.0)   # 0 disables clipping
        self.current_epoch = 0

        if optimizer_config is not None and model is not None:
            self.optimizer = get_optimizer(optimizer_config, model)
            self.base_lr = optimizer_config.get('lr', self.optimizer.lr)
        else:
            self.optimizer = None
            self.base_lr = 0.001

        self._validation_loss = defaultdict(list)
        self._last_loss_dict = {}

    def forward(self, batch, validate: bool=False): # return loss sum
        loss_dict = self.model.training_step(batch)
        assert isinstance(loss_dict, dict), "loss_dict must be a dict containing loss/metrics"
        assert self.loss_config is not None, "do not have loss_confing"
        loss_sum = 0.
        if validate:
            assets: List[Asset] = [a for a in batch['asset']]
            cls = assets[0].cls # guaranteed to be the same cls in dataloader
            for name in loss_dict:
                assert name in self.loss_config, f'unspecified loss {name}'
                self._validation_loss[f"val/{cls}_{name}"].append(_get_item(loss_dict[name]))
                loss_sum += self.loss_config[name] * loss_dict[name]
            self._validation_loss[f"val/{cls}_loss_sum"].append(_get_item(loss_sum))
            # TODO: log
            # self.log('val/loss_sum', loss_sum, prog_bar=True, logger=True, sync_dist=True, batch_size=len(assets))
        else:
            for name in loss_dict:
                assert name in self.loss_config, f"unspecified loss name: `{name}`"
                if self.loss_config[name] > 0:
                    loss_sum += self.loss_config[name] * loss_dict[name]
            loss_dict['loss_sum'] = loss_sum
            # stash scalar loss components so the train loop can log them
            self._last_loss_dict = {k: _get_item(v) for k, v in loss_dict.items()}
            # TODO: log
            # # add train prefix to loss_dict
            # prefixed_loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            # d = dict(sorted(prefixed_loss_dict.items()))
        if not isinstance(loss_sum, jt.Var):
            return jt.array(loss_sum)
        return loss_sum
    
    def on_train_epoch_start(self):
        pass
    
    def on_train_batch_start(self):
        pass
    
    def training_step(self, batch):
        return self.forward(batch, validate=False)
    
    def on_train_batch_end(self):
        pass
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        self._validation_loss = defaultdict(list)
    
    def on_validation_batch_start(self):
        pass
    
    def validation_step(self, batch):
        assert self.loss_config is not None, "do not have loss_confing"
        with jt.no_grad():
            return self.forward(batch, validate=True)
    
    def on_validation_batch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        pass
    
    def on_before_optimizer_step(self, optimizer):
        pass
    
    def on_predict_epoch_start(self):
        pass
    
    def on_predict_batch_start(self):
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.model.predict_step(batch)
    
    def on_predict_batch_end(self):
        pass
    
    def on_predict_epoch_end(self):
        pass
    
    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self.model.set_predict(False)
        global_step = 0
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.model.train()
            self.on_train_epoch_start()
            _wandb_log({'train/lr': self.optimizer.lr, 'epoch': epoch}, step=global_step)
            train_dataloader = self.dataset_module.train_dataloader()
            assert train_dataloader is not None, "train_dataloader is None"
            pbar = tqdm(train_dataloader, total=len(train_dataloader)//train_dataloader.batch_size) # type: ignore
            epoch_loss_sum, epoch_loss_cnt = 0.0, 0
            for batch_idx, batch in enumerate(pbar):
                self.on_train_batch_start()
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                loss_val = _get_item(loss)
                epoch_loss_sum += loss_val
                epoch_loss_cnt += 1
                if batch_idx % self.log_interval == 0:
                    pbar.set_description(f"Epoch {epoch}, Loss: {loss_val}")
                    log = {f"train/{k}": v for k, v in self._last_loss_dict.items()}
                    log['train/loss'] = loss_val
                    log['train/lr'] = self.optimizer.lr
                    log['epoch'] = epoch
                    log.update(gpu_stats())     # GPU util / memory of all cards
                    _wandb_log(log, step=global_step)
                self.on_before_optimizer_step(self.optimizer)
                if self.grad_clip and self.grad_clip > 0:
                    # guard against rare exploding score-matching gradients
                    self.optimizer.clip_grad_norm(self.grad_clip)
                self.optimizer.step()
                self.on_train_batch_end()
                global_step += 1
            self.on_train_epoch_end()
            if epoch_loss_cnt > 0:
                _wandb_log({'train/epoch_loss': epoch_loss_sum / epoch_loss_cnt, 'epoch': epoch},
                           step=global_step)

            self.model.eval()
            validate_dataloader = self.dataset_module.validate_dataloader()
            if validate_dataloader is not None:
                self.on_validation_epoch_start()
                if isinstance(validate_dataloader, dict):
                    for name, dataloader in validate_dataloader.items():
                        pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size)
                        for batch in pbar:
                            self.on_validation_batch_start()
                            loss = self.validation_step(batch)
                            pbar.set_description(f"Epoch {epoch}, Validate {name}, Loss: {_get_item(loss)}")
                            self.on_validation_batch_end()
                else:
                    pbar = tqdm(validate_dataloader, total=len(validate_dataloader)//validate_dataloader.batch_size)
                    for batch in pbar:
                        self.on_validation_batch_start()
                        loss = self.validation_step(batch)
                        pbar.set_description(f"Epoch {epoch}, Validate, Loss: {_get_item(loss)}")
                        self.on_validation_batch_end()
                # log mean of each accumulated validation metric for this epoch
                val_means = {k: (sum(v) / len(v)) for k, v in self._validation_loss.items() if v}
                if val_means:
                    val_means['epoch'] = epoch
                    _wandb_log(val_means, step=global_step)
                self.on_validation_epoch_end()

            # Save checkpoints on the main process only (avoids MPI write races).
            if is_main_process():
                checkpoint_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_{epoch}.pkl')
                os.makedirs(self.ckpt_save_dir, exist_ok=True)
                self.model.save(checkpoint_path)
            if get_world_size() > 1:
                # keep ranks in lock-step across the epoch boundary
                try:
                    jt.sync_all()
                    if getattr(jt, 'mpi', None) is not None:
                        jt.mpi.barrier()
                except Exception:
                    pass

        if _HAS_WANDB and wandb.run is not None and is_main_process():
            wandb.finish()

    def predict(self):
        # only iterate once
        self.model.set_predict(True)
        self.model.eval()
        self.on_predict_epoch_start()
        predict_dataloader = self.dataset_module.predict_dataloader()
        assert predict_dataloader is not None, "predict_dataloader is None"
        if not isinstance(predict_dataloader, dict):
            predict_dataloader = {"predict": predict_dataloader}
        for dataloader_name, dataloader in predict_dataloader.items():
            pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size) # type: ignore
            for batch_idx, batch in enumerate(pbar):
                self.on_predict_batch_start()
                output = self.predict_step(batch, batch_idx)
                if self.writer is not None:
                    self.writer.write(batch, output, dataset_module=self.dataset_module)
                pbar.set_description(f"Predicting {dataloader_name}, Batch {batch_idx}")