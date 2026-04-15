#!/usr/bin/env python3
"""
Deep Parity Check: Verify that two independently-constructed HierarchosCore models
with identical weights produce numerically identical outputs.

Self-contained — creates shared random weights in-memory, no checkpoint needed.
This tests the core architectural invariant: same weights → same outputs.
"""
import sys
import os
import torch

sys.path.insert(0, '.')
from hierarchos.models.core import HierarchosCore
from hierarchos.training.trainer import AttrDict

def test_weight_sync_parity():
    """Create two models, sync weights, and verify outputs match exactly."""
    print("=== Deep Parity Check: Weight-Synced Models ===")

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
        max_l_steps=5,
        h_stride=4,
        l_conv_atol=1e-4,
        use_deepembed=True,
        use_rosa=True,
        compile=False,
        detach_every_n_steps=32,
    )

    # Create Model A
    torch.manual_seed(42)
    model_a = HierarchosCore(cfg)
    model_a.eval()

    # Create Model B and load Model A's weights
    model_b = HierarchosCore(cfg)
    model_b.load_state_dict(model_a.state_dict())
    model_b.eval()

    # Test Inputs
    torch.manual_seed(123)
    x = torch.randint(0, cfg.vocab_size, (1, 16))  # Use 16 tokens to cross stride boundaries
    labels = x.clone()
    labels[:, :5] = -100  # Mask some labels

    print("Running Forward Passes...")
    with torch.no_grad():
        out_a = model_a(x, labels=labels)
        out_b = model_b(x, labels=labels)

    # --- Numerical Comparison ---
    print("\n--- Numerical Comparison ---")
    all_pass = True

    # 1. Logits
    logits_close = torch.allclose(out_a['logits'], out_b['logits'], atol=1e-6)
    logits_max_diff = (out_a['logits'] - out_b['logits']).abs().max().item()
    print(f"Logits Match: {logits_close} (Max Diff: {logits_max_diff:.2e})")
    if not logits_close: all_pass = False

    # 2. Loss
    loss_close = torch.allclose(out_a['loss'], out_b['loss'], atol=1e-7)
    print(f"Loss Match:   {loss_close} (A: {out_a['loss'].item():.6f}, B: {out_b['loss'].item():.6f})")
    if not loss_close: all_pass = False

    # 3. Ponder Cost
    p_a = out_a.get('ponder_cost')
    p_b = out_b.get('ponder_cost')
    if p_a is not None and p_b is not None:
        p_close = torch.allclose(p_a, p_b, atol=1e-7)
        print(f"Ponder Match: {p_close} (A: {p_a.item():.6f}, B: {p_b.item():.6f})")
        if not p_close: all_pass = False
    else:
        print(f"Ponder Cost: Skipped (no ponder steps in this config)")

    # 4. Commitment Cost
    c_a = out_a.get('commitment_cost')
    c_b = out_b.get('commitment_cost')
    if c_a is not None and c_b is not None:
        c_close = torch.allclose(c_a, c_b, atol=1e-7)
        print(f"Commitment Match: {c_close} (A: {c_a.item():.6f}, B: {c_b.item():.6f})")
        if not c_close: all_pass = False
    else:
        print(f"Commitment Cost: Skipped (zero commitment)")

    # 5. Hidden States
    h_close = torch.allclose(out_a['h_state'], out_b['h_state'], atol=1e-6)
    l_close = torch.allclose(out_a['l_state'], out_b['l_state'], atol=1e-6)
    print(f"H-State Match: {h_close}")
    print(f"L-State Match: {l_close}")
    if not h_close: all_pass = False
    if not l_close: all_pass = False

    # 6. LTM Memory State
    ltm_a = out_a.get('ltm_memory_state')
    ltm_b = out_b.get('ltm_memory_state')
    if ltm_a is not None and ltm_b is not None:
        fast_close = torch.allclose(ltm_a[0], ltm_b[0], atol=1e-6)
        mom_close = torch.allclose(ltm_a[1], ltm_b[1], atol=1e-6)
        print(f"LTM Fast Match: {fast_close}")
        print(f"LTM Mom Match:  {mom_close}")
        if not fast_close or not mom_close: all_pass = False

    if all_pass:
        print("\n[PASS] Weight-synced models are numerically IDENTICAL.")
    else:
        print("\n[FAIL] Numerical discrepancy detected.")
        sys.exit(1)

if __name__ == "__main__":
    test_weight_sync_parity()
