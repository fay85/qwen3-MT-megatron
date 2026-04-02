# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain / SFT GPT with Torch Profiler + Misight enabled."""

import os
import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import Union
if os.getenv("ACCELERATOR_BACKEND", "musa") == "musa":
    import musa_patch
else:
    pass
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

# ========== PROFILER INTEGRATION START ==========
import time
from torch.profiler import profile, schedule, ProfilerActivity

# Global profiler instance
_prof = None
_profiler_initialized = False
_profiler_step = 0
_profiler_finished = False

def _trace_handler(p):
    """Save profiler traces."""
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        rank = torch.distributed.get_rank()
        chrome_trace_path = f"./output/profiler_rank{rank}_step{p.step_num}.json"
        p.export_chrome_trace(chrome_trace_path)
        print_rank_0(f"[Profiler] Saved Chrome trace to {chrome_trace_path}")
        try:
            misight_trace_path = f"./output/misight_rank{rank}_step{p.step_num}.json"
            p.export_misight(misight_trace_path)
            print_rank_0(f"[Profiler] Saved Misight trace to {misight_trace_path}")
        except AttributeError:
            pass

def _init_profiler():
    """Initialize the profiler on rank 0."""
    global _prof, _profiler_initialized
    if _profiler_initialized:
        return
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        print_rank_0("[Profiler] Initializing Torch Profiler with MUSA support...")
        os.makedirs("./output", exist_ok=True)
        profiler_schedule = schedule(wait=5, warmup=2, active=5, repeat=1)
        _prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.MUSA],
            schedule=profiler_schedule,
            on_trace_ready=_trace_handler,
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            with_flops=True
        )
        _prof.start()
        print_rank_0("[Profiler] Profiler started.")
    _profiler_initialized = True

def _step_profiler():
    """Advance profiler one step."""
    global _prof, _profiler_step, _profiler_finished
    if _profiler_finished:
        return
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        if _prof is not None:
            _prof.step()
            _profiler_step += 1
            if _profiler_step >= 15:
                print_rank_0(f"[Profiler] Profiling completed at step {_profiler_step}, stopping profiler.")
                _prof.stop()
                _profiler_finished = True


# ========== PERFORMANCE MONITOR ==========
import atexit
from datetime import datetime

class PerfMonitor:
    """Lightweight per-iteration performance monitor with Excel export.

    Reports per iteration (stdout + accumulated for export):
      1. samples/sec          — global batch samples per second
      2. tokens/sec/gpu       — token throughput per GPU
      3. time_per_iter (ms)   — wall-clock time of the training iteration
      4. TFLOPS/gpu           — estimated achieved TFLOPS per GPU

    At training end (or process exit), writes all recorded data to an Excel
    file under ./perf_results/ with:
      - "per_iteration" sheet:  raw data for every logged iteration
      - "summary" sheet:        mean / median / min / max / std / p95 / p99

    FLOP estimation follows the standard Megatron formula:
      FLOPs_per_token ≈ 6 * P * (1 + S / (6 * H))
    For MoE models the MLP FLOPs are scaled by top-k.
    """

    def __init__(self, log_interval: int = 1, warmup_iters: int = 5):
        self.log_interval = log_interval
        self.warmup_iters = warmup_iters
        self._iter = 0
        self._t_start = 0.0
        self._args = None
        self._flops_per_token = None
        self._records = []
        self._saved = False

    def _lazy_init(self):
        if self._args is not None:
            return
        self._args = get_args()
        self._flops_per_token = self._estimate_flops_per_token()
        if torch.distributed.get_rank() == 0:
            print(
                f"[PerfMonitor] initialized | "
                f"world_size={self._args.world_size}  "
                f"seq_length={self._args.seq_length}  "
                f"micro_batch_size={self._args.micro_batch_size}  "
                f"global_batch_size={self._args.global_batch_size}  "
                f"estimated FLOPs/token={self._flops_per_token:.3e}"
            )
            atexit.register(self.save_excel)

    def _estimate_flops_per_token(self) -> float:
        """Estimate FLOPs per token for the model (forward + backward = 3x forward).

        Dense:
          forward_flops_per_token ≈ 2 * P * (1 + S / (6*H))
          total (fwd+bwd)          ≈ 6 * P * (1 + S / (6*H))

        MoE: MLP portion is scaled by (top_k * expert_ffn / dense_ffn).
        """
        a = self._args
        h = a.hidden_size
        s = a.seq_length
        L = a.num_layers

        attn_params = 4 * h * h
        if a.group_query_attention and a.num_query_groups is not None:
            kv_ratio = a.num_query_groups / a.num_attention_heads
            attn_params = h * h * (1 + kv_ratio + kv_ratio + 1)

        ffn_hidden = getattr(a, 'ffn_hidden_size', 4 * h)
        if a.swiglu:
            mlp_params = 3 * h * ffn_hidden
        else:
            mlp_params = 2 * h * ffn_hidden

        if a.num_experts is not None:
            moe_ffn = getattr(a, 'moe_ffn_hidden_size', None) or ffn_hidden
            if a.swiglu:
                expert_mlp_params = 3 * h * moe_ffn
            else:
                expert_mlp_params = 2 * h * moe_ffn
            topk = getattr(a, 'moe_router_topk', 1)
            mlp_params = expert_mlp_params * topk

        shared_ffn = getattr(a, 'moe_shared_expert_intermediate_size', None)
        shared_mlp_params = 0
        if shared_ffn is not None:
            if a.swiglu:
                shared_mlp_params = 3 * h * shared_ffn
            else:
                shared_mlp_params = 2 * h * shared_ffn

        params_per_layer = attn_params + mlp_params + shared_mlp_params
        total_params = L * params_per_layer

        vocab = getattr(a, 'padded_vocab_size', getattr(a, 'vocab_size', 0))
        embed_params = 2 * vocab * h

        P = total_params + embed_params
        flops = 6 * P * (1 + s / (6 * h))
        return flops

    def on_step_start(self):
        self._lazy_init()
        torch.cuda.synchronize()
        self._t_start = time.perf_counter()

    def on_step_end(self):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._t_start
        self._iter += 1

        if self._iter <= self.warmup_iters:
            return
        if torch.distributed.get_rank() != 0:
            return

        a = self._args
        from megatron.core.num_microbatches_calculator import get_num_microbatches
        num_micro = get_num_microbatches()
        global_batch = a.micro_batch_size * a.data_parallel_size * num_micro
        tokens_per_iter = global_batch * a.seq_length
        world_size = a.world_size

        samples_per_sec = global_batch / elapsed
        tokens_per_sec_per_gpu = tokens_per_iter / elapsed / world_size
        tflops_per_gpu = (self._flops_per_token * tokens_per_iter) / elapsed / world_size / 1e12
        time_ms = elapsed * 1000.0

        self._records.append({
            'iteration': self._iter,
            'samples_per_sec': round(samples_per_sec, 4),
            'tokens_per_sec_per_gpu': round(tokens_per_sec_per_gpu, 2),
            'time_ms': round(time_ms, 2),
            'tflops_per_gpu': round(tflops_per_gpu, 4),
        })

        if self._iter % self.log_interval == 0:
            print(
                f"[PerfMonitor] iter {self._iter:6d} | "
                f"samples/sec {samples_per_sec:8.2f} | "
                f"tokens/sec/gpu {tokens_per_sec_per_gpu:10.1f} | "
                f"time {time_ms:8.1f} ms | "
                f"TFLOPS/gpu {tflops_per_gpu:7.2f}"
            )

    def save_excel(self):
        """Write all collected records to an Excel file with two sheets."""
        if self._saved or not self._records:
            return
        self._saved = True

        try:
            import pandas as pd
        except ImportError:
            print("[PerfMonitor] WARNING: pandas not installed, falling back to CSV export.")
            self._save_csv_fallback()
            return

        df = pd.DataFrame(self._records)

        a = self._args
        num_experts = a.num_experts if a.num_experts is not None else 0
        topk = getattr(a, 'moe_router_topk', 0) if num_experts else 0

        summary_data = {}
        metric_cols = ['samples_per_sec', 'tokens_per_sec_per_gpu', 'time_ms', 'tflops_per_gpu']
        for col in metric_cols:
            s = df[col]
            summary_data[col] = {
                'mean': round(s.mean(), 4),
                'median': round(s.median(), 4),
                'std': round(s.std(), 4),
                'min': round(s.min(), 4),
                'max': round(s.max(), 4),
                'p95': round(s.quantile(0.95), 4),
                'p99': round(s.quantile(0.99), 4),
            }
        df_summary = pd.DataFrame(summary_data).T
        df_summary.index.name = 'metric'

        config_data = {
            'model': f"Qwen3-{a.hidden_size}",
            'num_layers': a.num_layers,
            'hidden_size': a.hidden_size,
            'num_attention_heads': a.num_attention_heads,
            'seq_length': a.seq_length,
            'micro_batch_size': a.micro_batch_size,
            'global_batch_size': a.global_batch_size,
            'world_size': a.world_size,
            'tp_size': a.tensor_model_parallel_size,
            'pp_size': a.pipeline_model_parallel_size,
            'num_experts': num_experts,
            'moe_router_topk': topk,
            'bf16': getattr(a, 'bf16', False),
            'fp8': getattr(a, 'fp8', None) is not None,
            'total_iters_logged': len(df),
            'warmup_iters_skipped': self.warmup_iters,
            'estimated_flops_per_token': self._flops_per_token,
        }
        df_config = pd.DataFrame(list(config_data.items()), columns=['parameter', 'value'])

        out_dir = os.path.join(os.environ.get('WORK_HOME', '.'), 'perf_results')
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        xlsx_path = os.path.join(out_dir, f'perf_{timestamp}_rank0.xlsx')

        try:
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='per_iteration', index=False)
                df_summary.to_excel(writer, sheet_name='summary')
                df_config.to_excel(writer, sheet_name='config', index=False)
            print(f"[PerfMonitor] Saved performance report to {xlsx_path}")
        except ImportError:
            print("[PerfMonitor] openpyxl not installed, falling back to CSV.")
            self._save_csv_fallback()
        except Exception as e:
            print(f"[PerfMonitor] Excel export failed ({e}), falling back to CSV.")
            self._save_csv_fallback()

    def _save_csv_fallback(self):
        """Fallback: write records as CSV if pandas/openpyxl unavailable."""
        import csv

        out_dir = os.path.join(os.environ.get('WORK_HOME', '.'), 'perf_results')
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(out_dir, f'perf_{timestamp}_rank0.csv')

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._records[0].keys())
            writer.writeheader()
            writer.writerows(self._records)
        print(f"[PerfMonitor] Saved performance data to {csv_path}")


_perf = PerfMonitor(log_interval=1, warmup_iters=5)
# ========== PROFILER INTEGRATION END ==========

stimer = StragglerDetector()

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model."""
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts,
                    args.moe_grouped_gemm,
                    args.qk_layernorm,
                    args.multi_latent_attention,
                    args.moe_use_legacy_grouped_gemm
                )
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                    args.num_experts,
                    args.moe_grouped_gemm,
                    args.qk_layernorm
                )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base
            )

    return model


def get_batch(data_iterator):
    """Generate a batch."""
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function."""
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    reporting_loss = loss.clone().detach()
    if not int(os.getenv("NO_LOSS_REDUCE", 0)):
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    local_num_tokens = loss[1].clone().detach().to(torch.int)

    averaged_loss = reporting_loss[0] / torch.clamp(reporting_loss[1], min=1)

    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {'lm loss': averaged_loss},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step WITH PROFILER + PERF MONITOR."""
    args = get_args()
    timers = get_timers()

    if not _profiler_initialized:
        _init_profiler()
    _step_profiler()

    _perf.on_step_start()

    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

    _perf.on_step_end()

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets."""
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
