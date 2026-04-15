#!/usr/bin/env python3
"""
Test: Autoregressive inference loop with state carryover.
Self-contained — creates a fresh model in-memory, no checkpoint needed.
Verifies: token generation, state persistence, repetition detection, state diagnostics.
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
from hierarchos import HierarchosCore, AttrDict

def test_inference():
    print("=== Test: Inference Generation Loop (Self-Contained) ===")

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

    # Test prompt (random tokens since we have no real tokenizer for this mini model)
    torch.manual_seed(42)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    temperature = 0.8
    max_tokens = 20
    generated = []

    h_state, l_state, prev_ctx, target_ctx, drift_state = None, None, None, None, None
    current_ids = input_ids

    print(f"Prompt tokens: {input_ids.tolist()}")
    print("Generating...")

    with torch.no_grad():
        for step in range(max_tokens):
            outputs = model(
                current_ids,
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_ctx,
                target_context=target_ctx,
                drift_state=drift_state,
                global_pos_offset=len(generated) + input_ids.shape[1]
            )

            # Update states
            h_state = outputs.get('h_state')
            l_state = outputs.get('l_state')
            prev_ctx = outputs.get('prev_context')
            target_ctx = outputs.get('target_context')
            drift_state = outputs.get('drift_state')

            # Get next token
            logits = outputs['logits'][:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token.item())
            current_ids = next_token

            # Check for repetition
            if len(generated) > 10:
                last_5 = generated[-5:]
                prev_5 = generated[-10:-5]
                if last_5 == prev_5:
                    print(f"  [REPETITION DETECTED at step {step}]")
                    break

    print(f"Generated {len(generated)} tokens: {generated}")

    # State diagnostics
    print(f"\nState diagnostics:")
    assert h_state is not None, "FAIL: h_state is None after generation"
    assert l_state is not None, "FAIL: l_state is None after generation"

    print(f"  h_state: min={h_state.min().item():.3f}, max={h_state.max().item():.3f}, mean={h_state.mean().item():.3f}")
    print(f"  l_state: min={l_state.min().item():.3f}, max={l_state.max().item():.3f}, mean={l_state.mean().item():.3f}")

    assert not torch.isnan(h_state).any(), "FAIL: h_state contains NaN"
    assert not torch.isnan(l_state).any(), "FAIL: l_state contains NaN"
    assert not torch.isinf(h_state).any(), "FAIL: h_state contains Inf"
    assert not torch.isinf(l_state).any(), "FAIL: l_state contains Inf"

    if drift_state is not None:
        print(f"  drift_state: min={drift_state.min().item():.3f}, max={drift_state.max().item():.3f}")
        assert not torch.isnan(drift_state).any(), "FAIL: drift_state contains NaN"

    print("[PASS] Inference generation loop test")

if __name__ == "__main__":
    test_inference()
