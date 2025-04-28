import torch
import transformers
from typing import Dict, List, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LlamaFLOPsCounter:
    """Counts FLOPs for Llama model operations during generation."""
    
    def __init__(self, model: transformers.LlamaForCausalLM):
        self.model = model
        self.reset_counters()
        self.hooks = []
        
        # Model dimensions for FLOP calculations
        self.hidden_size = model.config.hidden_size
        self.intermediate_size = model.config.intermediate_size
        self.num_attention_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.num_key_value_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        self.vocab_size = model.config.vocab_size
        self.model_type = model.config.model_type
        
        # Cache for sequence lengths to avoid repeated computation
        self.seq_lens_cache = {}
        
    def reset_counters(self):
        """Reset all FLOP counters."""
        self.total_flops = 0
        self.layer_flops = {}
        self.operation_flops = {
            "attention": 0,
            "mlp": 0,
            "layernorm": 0,
            "embedding": 0,
            "lm_head": 0,
            "residual": 0
        }
        self.layer_counts = {}
        
    def register_hooks(self):
        """Register hooks on model components to count FLOPs."""
        self.reset_counters()
        
        # Remove any existing hooks
        self.remove_hooks()
        
        # Register hooks for all transformer layers
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Initialize layer count
            self.layer_counts[layer_idx] = 0
            
            # Create closure functions to properly capture layer_idx
            def create_hook_fn(layer_idx, component_name, op_type):
                def hook_fn(m, i, o):
                    return self._count_flops_linear(m, i[0], o, f"layer_{layer_idx}_{component_name}", op_type)
                return hook_fn
            
            # Attention hooks
            self.hooks.append(layer.self_attn.q_proj.register_forward_hook(
                create_hook_fn(layer_idx, "q_proj", "attention")
            ))
            self.hooks.append(layer.self_attn.k_proj.register_forward_hook(
                create_hook_fn(layer_idx, "k_proj", "attention")
            ))
            self.hooks.append(layer.self_attn.v_proj.register_forward_hook(
                create_hook_fn(layer_idx, "v_proj", "attention")
            ))
            self.hooks.append(layer.self_attn.o_proj.register_forward_hook(
                create_hook_fn(layer_idx, "o_proj", "attention")
            ))
            
            # MLP hooks
            self.hooks.append(layer.mlp.gate_proj.register_forward_hook(
                create_hook_fn(layer_idx, "gate_proj", "mlp")
            ))
            self.hooks.append(layer.mlp.up_proj.register_forward_hook(
                create_hook_fn(layer_idx, "up_proj", "mlp")
            ))
            self.hooks.append(layer.mlp.down_proj.register_forward_hook(
                create_hook_fn(layer_idx, "down_proj", "mlp")
            ))
            
        # Hooks for embedding and LM head
        self.hooks.append(self.model.model.embed_tokens.register_forward_hook(
            lambda m, i, o: self._count_flops_embedding(m, i[0], o)
        ))
        self.hooks.append(self.model.lm_head.register_forward_hook(
            lambda m, i, o: self._count_flops_linear(m, i[0], o, "lm_head", "lm_head")
        ))
            
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def _count_flops_linear(self, module, x, output, name, op_type):
        """Count FLOPs for linear layer (matmul + bias)."""
        if x.dim() == 2:
            # Forward pass: [batch_size, in_features] x [in_features, out_features]
            batch_size, in_features = x.shape
            out_features = output.shape[1]
            # 2 FLOPs per multiplication-addition
            flops = batch_size * in_features * out_features * 2
        else:
            # Sequence input: [batch_size, seq_len, in_features]
            batch_size, seq_len, in_features = x.shape
            out_features = output.shape[2]
            # 2 FLOPs per multiplication-addition
            flops = batch_size * seq_len * in_features * out_features * 2
        
        # Add to counters
        self.total_flops += flops
        self.operation_flops[op_type] += flops
        
        # Extract layer number from name (if applicable)
        if "layer_" in name:
            layer_idx = int(name.split("_")[1])
            if layer_idx not in self.layer_flops:
                self.layer_flops[layer_idx] = 0
            self.layer_flops[layer_idx] += flops
            self.layer_counts[layer_idx] += 1
    
    def _count_flops_embedding(self, module, x, output):
        """Count FLOPs for embedding lookup."""
        # Embedding lookup is essentially free in terms of FLOPs
        # but we'll count it for completeness
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        embed_dim = output.shape[2]
        
        # Only counting the actual embedding lookup
        flops = batch_size * seq_len * embed_dim
        
        self.total_flops += flops
        self.operation_flops["embedding"] += flops
    
    def track_layer_usage(self, layer_idx):
        """Record that a particular layer was used during generation."""
        if layer_idx in self.layer_counts:
            self.layer_counts[layer_idx] += 1
    
    def get_summary(self):
        """Return a summary of FLOP usage."""
        return {
            "total_flops": self.total_flops,
            "operation_flops": self.operation_flops,
            "layer_flops": self.layer_flops,
            "layer_counts": self.layer_counts
        }
    
    def estimate_early_exit_savings(self, exit_layer):
        """Estimate FLOPs saved by early exiting at a given layer."""
        if not self.layer_flops:
            return 0
        
        total_layer_flops = sum(self.layer_flops.values())
        saved_flops = sum(self.layer_flops.get(i, 0) for i in range(exit_layer, len(self.model.model.layers)))
        
        return {
            "total_layer_flops": total_layer_flops,
            "saved_flops": saved_flops,
            "saving_percentage": (saved_flops / total_layer_flops) * 100 if total_layer_flops > 0 else 0
        }

    def humanize_flops(self, flops):
        """Convert raw FLOP count to human-readable format."""
        if flops < 1e9:
            return f"{flops / 1e6:.2f} MFLOPs"
        elif flops < 1e12:
            return f"{flops / 1e9:.2f} GFLOPs"
        else:
            return f"{flops / 1e12:.2f} TFLOPs" 