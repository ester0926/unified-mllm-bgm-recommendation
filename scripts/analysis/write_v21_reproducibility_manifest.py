from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "v21_reproducibility_manifest.json"


def sha256(path: Path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path):
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        ).strip()
    except Exception:
        return "unavailable"


def main():
    profiles = Path(os.environ.get(
        "PROFILES_JSONL_PATH",
        r"data/user_profiling/long_term_preference\stage4_recLLM\profiles.jsonl",
    ))
    music_meta = Path(os.environ.get(
        "MUSIC_METADATA_PATH",
        r"data/user_profiling/music_metadata_simple\music_metadata_enriched.json",
    ))
    youtube_meta = Path(os.environ.get(
        "YOUTUBE_METADATA_PATH",
        r"data/user_profiling/music_metadata_simple\youtube_metadata.jsonl",
    ))
    files = [
        ROOT / "cache" / "ltp_hybrid.npy",
        ROOT / "cache" / "ltp_hybrid_ids.json",
        ROOT / "cache" / "ltp_explicit_only.npy",
        ROOT / "cache" / "ltp_explicit_only_ids.json",
        ROOT / "cache" / "ltp_implicit_only.npy",
        ROOT / "cache" / "ltp_implicit_only_ids.json",
        ROOT / "results" / "analysis" / "b5_personas_v21" / "persona_specs.json",
        ROOT / "results" / "analysis" / "b5_personas_v21" / "persona_ltp.npz",
        profiles,
        music_meta,
        youtube_meta,
        ROOT / "checkpoints" / "exp_01" / "best" / "adapter_model.safetensors",
        ROOT / "checkpoints" / "exp_01" / "best" / "projectors.pt",
        ROOT / "checkpoints" / "exp_01" / "best" / "ranking_head.pt",
        ROOT / "checkpoints" / "exp_04" / "best" / "adapter_model.safetensors",
        ROOT / "checkpoints" / "exp_04" / "best" / "projectors.pt",
        ROOT / "checkpoints" / "exp_04" / "best" / "ranking_head.pt",
    ]
    scripts = [
        ROOT / "scripts" / "analysis" / "b5_build_persona_specs.py",
        ROOT / "scripts" / "analysis" / "b5_build_persona_ltp.py",
        ROOT / "scripts" / "eval_main" / "run_eval_500pool_persona.py",
        ROOT / "scripts" / "eval_main" / "run_eval_500pool_persona_v2.py",
        ROOT / "scripts" / "analysis" / "b5_persona_metrics_v21.py",
        ROOT / "scripts" / "analysis" / "reuse_noltp_v2_for_persona_v21.py",
        ROOT / "scripts" / "analysis" / "b6_preference_video_conflict_v21.py",
        ROOT / "scripts" / "analysis" / "path_level_generation_analysis_v21.py",
        ROOT / "scripts" / "analysis" / "prepare_v21_preference_claim_blind_audit.py",
        ROOT / "scripts" / "eval_main" / "run_eval_fixed_hybrid_components_v21.py",
        ROOT / "scripts" / "analysis" / "reuse_full_exp01_for_fixed_component_v21.py",
        ROOT / "scripts" / "analysis" / "fixed_hybrid_component_analysis_v21.py",
    ]
    artifacts = [
        ROOT / "results" / "analysis" / "path_level_generation_v21" / "path_level_summary.json",
        ROOT / "results" / "analysis" / "path_level_generation_v21" / "preference_claims_full_audit.csv",
        ROOT / "results" / "analysis" / "path_level_generation_v21" / "preference_claim_blind_audit_packet.csv",
        ROOT / "results" / "analysis" / "b5_personas_v21" / "persona_metrics_v21_summary.json",
        ROOT / "results" / "main_eval" / "exp_01" / "persona_eval_v21" / "persona_v2_no_ltp_reuse_provenance.json",
        ROOT / "results" / "analysis" / "b6_conflict_v21" / "conflict_summary.json",
        ROOT / "results" / "main_eval" / "exp_01" / "fixed_component_intervention_v21" / "fixed_component_full_reuse_provenance.json",
        ROOT / "results" / "analysis" / "fixed_hybrid_component_v21" / "fixed_component_summary.json",
    ]
    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_info = {"error": str(exc)}

    manifest = {
        "schema": "v21-experiment-provenance-1",
        "project_root": str(ROOT),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch_info,
        },
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "status_porcelain": git_value("status", "--porcelain"),
        },
        "inputs_and_checkpoints": [describe(p) for p in files],
        "analysis_scripts": [describe(p) for p in scripts],
        "analysis_artifacts": [describe(p) for p in artifacts],
        "path_overrides": {
            "PROFILES_JSONL_PATH": os.environ.get("PROFILES_JSONL_PATH"),
            "MUSIC_METADATA_PATH": os.environ.get("MUSIC_METADATA_PATH"),
            "YOUTUBE_METADATA_PATH": os.environ.get("YOUTUBE_METADATA_PATH"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
