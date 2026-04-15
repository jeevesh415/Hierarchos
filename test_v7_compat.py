"""Quick V7-only compatibility check - no DeepEmbed, no ROSA."""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hierarchos import HierarchosCore, AttrDict

config = AttrDict({
    'vocab_size': 100, 'context_dim': 16, 'persistent_dim': 4,
    'ltm_slots': 10, 'ltm_key_dim': 8, 'ltm_val_dim': 8,
    'ltm_lr': 0.01, 'ltm_topk': 2, 'h_hidden': 16, 'l_hidden': 16,
    'max_h_steps': 2, 'max_l_steps': 2, 'l_conv_atol': 1e-4,
    'h_stride': 2, 'compile': False,
    'use_deepembed': False, 'use_rosa': False,  # V7 mode
})

model = HierarchosCore(config)
model.train()

# Verify no V8 modules exist
assert not hasattr(model, 'h_deepemb'), "h_deepemb should not exist in V7 mode"
assert not hasattr(model, 'l_deepemb'), "l_deepemb should not exist in V7 mode"
assert not hasattr(model, 'rosa_emb'), "rosa_emb should not exist in V7 mode"
print("[OK] V7 mode: No V8 modules created")

# Forward + backward
B, T = 2, 8
input_ids = torch.randint(0, 100, (B, T))
labels = torch.randint(0, 100, (B, T))
out = model(input_ids=input_ids, labels=labels)

assert not torch.isnan(out['loss']), "Loss is NaN"
print(f"[OK] V7 forward: loss={out['loss'].item():.4f}")

out['loss'].backward()
grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
print(f"[OK] V7 backward: {grad_count} params with nonzero grads")

# State continuity
model.eval()
with torch.no_grad():
    out1 = model(input_ids=input_ids[:, :4])
    out2 = model(input_ids=input_ids[:, 4:], h_state=out1['h_state'], l_state=out1['l_state'])
assert not torch.allclose(out1['h_state'], out2['h_state']), "States should differ"
print("[OK] V7 state continuity works")

print("\n=== V7 backward compatibility: ALL CHECKS PASSED ===")
