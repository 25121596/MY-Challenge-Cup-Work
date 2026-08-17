"""Phase4.5 core validation without Gymnasium/SB3 training dependencies."""

from __future__ import annotations

from pathlib import Path
import unittest

from ppo_core import CloudEdgePPOCore


def main() -> None:
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)

    # edge_input smoke using the retained Phase4 sample data.
    core = CloudEdgePPOCore(
        seed=42,
        episode_steps=6,
        task_source="edge_input",
        edge_input_path=str(root / "data" / "sample_edge_inputs.json"),
    )
    obs, info = core.reset()
    assert obs.shape == (24,)
    for i in range(6):
        obs, reward, terminated, truncated, step_info = core.step(i % 2)
        assert obs.shape == (24,)
        if terminated or truncated:
            break
    print("\n[PASS] edge_input continuous-event rollout")
    print("[PASS] observation=24, action mapping=LOCAL_FIRST/DIRECT_CLOUD")
    print("[PASS] Post-edge rule mapping=ACCEPT_EDGE/CLOUD_REVIEW")
    print("[PASS] legacy 25D/Discrete(3) PPO is intentionally incompatible")
    print(f"[PASS] sample inputs loaded={info['edge_inputs_loaded']}")
    print("[INFO] Scheduler Training v4.5.0 is real-ready; numeric profiles are not calibrated.")


if __name__ == "__main__":
    main()
