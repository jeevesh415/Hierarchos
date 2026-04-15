#!/usr/bin/env python3
"""
Test: Basic forward pass with loss computation.
Self-contained — creates a fresh model in-memory, no checkpoint needed.
"""
import sys
sys.path.insert(0, '.')
import torch
from hierarchos import HierarchosCore, AttrDict

def test_forward():
    print("=== Test: Forward Pass (Self-Contained) ===")

    cfg = AttrDict(
        vocab_size=500,
        context_dim=32,
        h_hidden=32,
        l_hidden=32,
        ltm_slots=64,
        ltm_key_dim=16,
        ltm_val_dim=16,
        ltm_topk=2,
        persistent_dim=16,
        max_h_steps=3,
        max_l_steps=3,
        h_stride=4,
        l_conv_atol=1e-4,
        use_deepembed=True,
        use_rosa=True,
        compile=False,
    )

    model = HierarchosCore(cfg)
    model.eval()

    torch.manual_seed(42)
    x = torch.randint(0, cfg.vocab_size, (1, 10))
    labels = x.clone()
    labels[:, 0] = -100  # Mask first token

    # Forward pass
    with torch.no_grad():
        out = model(x, labels=labels)

    loss = out["loss"]
    logits = out["logits"]

    print(f"Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits sample (first 5): {logits[0, 0, :5]}")

    assert not torch.isnan(loss), "FAIL: Loss is NaN"
    assert not torch.isinf(loss), "FAIL: Loss is Inf"
    assert logits.shape == (1, 10, cfg.vocab_size), f"FAIL: Unexpected logits shape {logits.shape}"

    # Verify states are returned
    assert out.get("h_state") is not None, "FAIL: h_state not returned"
    assert out.get("l_state") is not None, "FAIL: l_state not returned"
    assert out.get("ltm_memory_state") is not None, "FAIL: ltm_memory_state not returned"

    print("[PASS] Forward pass test")

if __name__ == "__main__":
    test_forward()
