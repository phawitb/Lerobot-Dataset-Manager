"""
LeRobot Dataset Manager — Standalone dataset visualizer & manager.
Usage:  python main.py [--port 8080]
"""

import argparse
import json
import random
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="LeRobot Dataset Manager")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--host", type=str, default="0.0.0.0")
args, _ = parser.parse_known_args()

ROOT = Path(__file__).parent
DATA_DIR = (ROOT / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
AUGMENT_CONFIG_DIR = (ROOT / "augment_configs").resolve()
AUGMENT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LeRobot Dataset Manager")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


# ---------------------------------------------------------------------------
# Serve data files (videos, etc.) — mounted dynamically per request
# ---------------------------------------------------------------------------
@app.get("/data/{path:path}")
async def serve_data_file(path: str):
    """Serve any file under DATA_DIR (videos, parquet, etc.)."""
    file_path = DATA_DIR / path
    if not file_path.exists() or not file_path.is_file():
        return {"ok": False, "error": "File not found"}
    return FileResponse(str(file_path))


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(str(ROOT / "static" / "index.html"))


@app.get("/api/data-dir")
async def get_data_dir():
    """Return the configured data directory path."""
    return {"ok": True, "data_dir": str(DATA_DIR)}


# ---------------------------------------------------------------------------
# Dataset listing
# ---------------------------------------------------------------------------
@app.get("/api/datasets")
async def list_datasets():
    """List available LeRobot datasets in data directory."""
    datasets = []
    if DATA_DIR.exists():
        for d in sorted(DATA_DIR.iterdir()):
            info_path = d / "meta" / "info.json"
            if not info_path.exists():
                continue
            try:
                info = json.loads(info_path.read_text())
                task_list = []
                tasks_path = d / "meta" / "tasks.parquet"
                if tasks_path.exists():
                    try:
                        tbl = pq.read_table(str(tasks_path))
                        task_list = tbl.to_pydict().get("task", [])
                    except Exception:
                        pass
                datasets.append({
                    "name": d.name,
                    "total_episodes": info.get("total_episodes", 0),
                    "total_frames": info.get("total_frames", 0),
                    "fps": info.get("fps", 0),
                    "robot_type": info.get("robot_type", ""),
                    "tasks": task_list,
                })
            except Exception:
                pass
    return {"ok": True, "datasets": datasets}


# ---------------------------------------------------------------------------
# Dataset info
# ---------------------------------------------------------------------------
@app.get("/api/datasets/{name}/info")
async def dataset_info(name: str):
    """Get full info.json + tasks for a dataset."""
    info_path = DATA_DIR / name / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": "Dataset not found"}
    info = json.loads(info_path.read_text())
    task_list = []
    tasks_path = DATA_DIR / name / "meta" / "tasks.parquet"
    if tasks_path.exists():
        try:
            tbl = pq.read_table(str(tasks_path))
            task_list = tbl.to_pydict().get("task", [])
        except Exception:
            pass
    return {"ok": True, "info": info, "tasks": task_list}


@app.post("/api/datasets/{name}/update-tasks")
async def update_tasks(name: str, request: Request):
    """Update tasks for specific episodes, or all episodes if no episodes specified."""
    body = await request.json()
    task = body.get("task", "").strip()
    episodes = body.get("episodes")  # list of episode indices, or None for all
    if not task:
        return {"ok": False, "error": "Provide a non-empty task string"}
    ds_dir = DATA_DIR / name
    if not (ds_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Dataset not found"}

    # Update tasks.parquet — add new task if not present
    tasks_path = ds_dir / "meta" / "tasks.parquet"
    existing_tasks = []
    task_index_map = {}
    if tasks_path.exists():
        tbl = pq.read_table(str(tasks_path))
        d = tbl.to_pydict()
        existing_tasks = d.get("task", [])
        for i, t in enumerate(d.get("task_index", list(range(len(existing_tasks))))):
            task_index_map[existing_tasks[i] if i < len(existing_tasks) else ""] = t

    if task not in existing_tasks:
        new_idx = max(task_index_map.values(), default=-1) + 1
        existing_tasks.append(task)
        task_index_map[task] = new_idx
        pq.write_table(
            pa.table({"task_index": list(range(len(existing_tasks))), "task": existing_tasks}),
            str(tasks_path),
        )
        info_path = ds_dir / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["total_tasks"] = len(existing_tasks)
        info_path.write_text(json.dumps(info, indent=4))

    new_task_idx = task_index_map.get(task, 0)

    # Update episode parquet files — set tasks for specified episodes
    ep_dir = ds_dir / "meta" / "episodes"
    updated = 0
    if ep_dir.exists():
        ep_set = set(episodes) if episodes is not None else None
        for pf in sorted(ep_dir.rglob("*.parquet")):
            try:
                tbl = pq.read_table(str(pf))
            except Exception:
                continue
            d = tbl.to_pydict()
            ep_indices = d.get("episode_index", [])
            tasks_col = d.get("tasks", [[] for _ in ep_indices])
            changed = False
            for i, ei in enumerate(ep_indices):
                if ep_set is None or ei in ep_set:
                    tasks_col[i] = [task]
                    changed = True
                    updated += 1
            if changed:
                d["tasks"] = tasks_col
                pq.write_table(pa.table(d), str(pf))

    # Update task_index in data parquet files for affected episodes
    data_dir = ds_dir / "data"
    if data_dir.exists():
        ep_set = set(episodes) if episodes is not None else None
        for pf in sorted(data_dir.rglob("*.parquet")):
            try:
                tbl = pq.read_table(str(pf))
            except Exception:
                continue
            d = tbl.to_pydict()
            ep_indices = d.get("episode_index", [])
            if "task_index" not in d:
                continue
            task_indices = d["task_index"]
            changed = False
            for i, ei in enumerate(ep_indices):
                if ep_set is None or ei in ep_set:
                    task_indices[i] = new_task_idx
                    changed = True
            if changed:
                d["task_index"] = task_indices
                pq.write_table(pa.table(d), str(pf))

    return {"ok": True, "updated_episodes": updated, "task": task}


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------
@app.get("/api/datasets/{name}/episodes")
async def list_episodes(name: str):
    """List episodes with metadata."""
    ds_dir = DATA_DIR / name
    info_path = ds_dir / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": "Dataset not found"}
    info = json.loads(info_path.read_text())
    fps = info.get("fps", 30)

    episodes = []
    ep_dir = ds_dir / "meta" / "episodes"
    if ep_dir.exists():
        for pf in sorted(ep_dir.rglob("*.parquet")):
            try:
                tbl = pq.read_table(str(pf))
            except Exception:
                continue
            d = tbl.to_pydict()
            for i in range(len(d.get("episode_index", []))):
                episodes.append({k: v[i] for k, v in d.items()})

    return {"ok": True, "fps": fps, "total_episodes": info.get("total_episodes", 0), "episodes": episodes}


# ---------------------------------------------------------------------------
# Frame data
# ---------------------------------------------------------------------------
@app.get("/api/datasets/{name}/frames/{episode_index}")
async def dataset_frames(name: str, episode_index: int):
    """Get frame data (action, state) for a specific episode."""
    data_dir = DATA_DIR / name / "data"
    if not data_dir.exists():
        return {"ok": False, "error": "Data not found"}
    try:
        rows = []
        for pf in sorted(data_dir.rglob("*.parquet")):
            try:
                tbl = pq.read_table(str(pf))
            except Exception:
                continue
            d = tbl.to_pydict()
            for i in range(len(d.get("episode_index", []))):
                if d["episode_index"][i] == episode_index:
                    row = {}
                    for k, v in d.items():
                        val = v[i]
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        row[k] = val
                    rows.append(row)
        rows.sort(key=lambda r: r.get("frame_index", 0))
        return {"ok": True, "frames": rows}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------
@app.get("/api/datasets/{name}/segments/{ep}")
async def get_segments(name: str, ep: int):
    """Get saved segments for an episode."""
    seg_path = DATA_DIR / name / "meta" / "segments.json"
    if seg_path.exists():
        try:
            data = json.loads(seg_path.read_text())
            segs = data.get(str(ep))
            if segs:
                return {"ok": True, "segments": segs}
        except Exception:
            pass
    return {"ok": True, "segments": None}


@app.post("/api/datasets/{name}/segments/{ep}")
async def save_segments(name: str, ep: int, request: Request):
    """Save segments for an episode."""
    body = await request.json()
    segments = body.get("segments", [])
    seg_path = DATA_DIR / name / "meta" / "segments.json"
    data = {}
    if seg_path.exists():
        try:
            data = json.loads(seg_path.read_text())
        except Exception:
            pass
    data[str(ep)] = segments
    seg_path.write_text(json.dumps(data, indent=2))
    return {"ok": True}


# ---------------------------------------------------------------------------
# Rename dataset
# ---------------------------------------------------------------------------
@app.post("/api/datasets/{name}/rename")
async def rename_dataset(name: str, request: Request):
    """Rename a dataset directory."""
    body = await request.json()
    new_name = body.get("new_name", "").strip()
    if not new_name:
        return {"ok": False, "error": "New name is required"}
    if new_name == name:
        return {"ok": False, "error": "Same name"}
    if not re.match(r'^[a-zA-Z0-9_\-]+$', new_name):
        return {"ok": False, "error": "Invalid name. Use letters, numbers, dash, underscore only."}
    src = DATA_DIR / name
    dst = DATA_DIR / new_name
    if not src.exists():
        return {"ok": False, "error": "Dataset not found"}
    if dst.exists():
        return {"ok": False, "error": f"'{new_name}' already exists"}
    try:
        src.rename(dst)
        # Also rename augment config if exists
        old_cfg = AUGMENT_CONFIG_DIR / f"{name}.json"
        if old_cfg.exists():
            old_cfg.rename(AUGMENT_CONFIG_DIR / f"{new_name}.json")
        return {"ok": True, "new_name": new_name}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Delete dataset
# ---------------------------------------------------------------------------
@app.delete("/api/datasets/{name}")
async def delete_dataset(name: str):
    """Delete an entire dataset."""
    ds_dir = DATA_DIR / name
    if not ds_dir.exists() or not (ds_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Dataset not found"}
    try:
        shutil.rmtree(ds_dir)
        # Also remove augment config if exists
        cfg_path = AUGMENT_CONFIG_DIR / f"{name}.json"
        if cfg_path.exists():
            cfg_path.unlink()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Delete episodes
# ---------------------------------------------------------------------------
@app.post("/api/datasets/{name}/delete-episodes")
async def delete_episodes(name: str, request: Request):
    """Delete selected episodes and re-index."""
    body = await request.json()
    episodes_to_delete = set(body.get("episodes", []))
    if not episodes_to_delete:
        return {"ok": False, "error": "No episodes specified"}

    dataset_dir = DATA_DIR / name
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": "Dataset not found"}

    try:
        info = json.loads(info_path.read_text())
        original_total = info.get("total_episodes", 0)

        # 1. Filter episode metadata
        ep_dir = dataset_dir / "meta" / "episodes"
        all_ep_rows = []
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    if d["episode_index"][i] not in episodes_to_delete:
                        all_ep_rows.append({k: v[i] for k, v in d.items()})

        remaining_old = sorted(set(r["episode_index"] for r in all_ep_rows))
        ep_remap = {old: new for new, old in enumerate(remaining_old)}
        for row in all_ep_rows:
            row["episode_index"] = ep_remap[row["episode_index"]]

        # 2. Filter data
        data_dir = dataset_dir / "data"
        all_data_rows = []
        if data_dir.exists():
            for pf in sorted(data_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    if d["episode_index"][i] not in episodes_to_delete:
                        row = {}
                        for k, v in d.items():
                            val = v[i]
                            if hasattr(val, "tolist"):
                                val = val.tolist()
                            row[k] = val
                        all_data_rows.append(row)

        for idx, row in enumerate(all_data_rows):
            row["episode_index"] = ep_remap[row["episode_index"]]
            row["index"] = idx

        # 3. Write filtered episode metadata
        if ep_dir.exists():
            shutil.rmtree(ep_dir)
        ep_out = ep_dir / "chunk-000"
        ep_out.mkdir(parents=True, exist_ok=True)
        if all_ep_rows:
            cols = {k: [r[k] for r in all_ep_rows] for k in all_ep_rows[0]}
            pq.write_table(pa.table(cols), str(ep_out / "file-000.parquet"))

        # 4. Write filtered data
        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_out = data_dir / "chunk-000"
        data_out.mkdir(parents=True, exist_ok=True)
        if all_data_rows:
            cols = {k: [r[k] for r in all_data_rows] for k in all_data_rows[0]}
            pq.write_table(pa.table(cols), str(data_out / "file-000.parquet"))

        # 5. Update info.json
        new_total_episodes = len(remaining_old)
        info["total_episodes"] = new_total_episodes
        info["total_frames"] = len(all_data_rows)
        info["splits"] = {"train": f"0:{new_total_episodes}"}
        info_path.write_text(json.dumps(info, indent=4))

        # 6. Remove stale stats
        stats_path = dataset_dir / "meta" / "stats.json"
        if stats_path.exists():
            stats_path.unlink()

        deleted_count = original_total - new_total_episodes
        return {
            "ok": True,
            "deleted": deleted_count,
            "remaining_episodes": new_total_episodes,
            "remaining_frames": len(all_data_rows),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Combine datasets
# ---------------------------------------------------------------------------
@app.post("/api/datasets/combine")
async def combine_datasets(request: Request):
    """Combine multiple datasets into a new one."""
    body = await request.json()
    source_names = body.get("datasets", [])
    new_name = body.get("name", "").strip().replace(" ", "-").replace("/", "-")
    if not source_names or len(source_names) < 2:
        return {"ok": False, "error": "Select at least 2 datasets"}
    if not new_name:
        return {"ok": False, "error": "Enter a name"}

    new_dir = DATA_DIR / new_name
    if new_dir.exists():
        return {"ok": False, "error": f"'{new_name}' already exists"}

    try:
        sources = []
        for sname in source_names:
            sdir = DATA_DIR / sname
            ip = sdir / "meta" / "info.json"
            if not ip.exists():
                return {"ok": False, "error": f"'{sname}' not found"}
            sources.append({"name": sname, "dir": sdir, "info": json.loads(ip.read_text())})

        base_info = dict(sources[0]["info"])

        # Collect tasks
        all_tasks, task_set = [], set()
        for src in sources:
            tp = src["dir"] / "meta" / "tasks.parquet"
            if tp.exists():
                try:
                    for t in pq.read_table(str(tp)).to_pydict().get("task", []):
                        if t not in task_set:
                            task_set.add(t)
                            all_tasks.append(t)
                except Exception:
                    pass
        task_to_idx = {t: i for i, t in enumerate(all_tasks)}

        combined_data, combined_ep = [], []
        new_ep_idx, global_frame_idx = 0, 0

        for src in sources:
            sdir = src["dir"]
            # Read episode metadata
            ep_meta_list = []
            ep_dir = sdir / "meta" / "episodes"
            if ep_dir.exists():
                for pf in sorted(ep_dir.rglob("*.parquet")):
                    try:
                        tbl = pq.read_table(str(pf))
                    except Exception:
                        continue
                    d = tbl.to_pydict()
                    for i in range(len(d.get("episode_index", []))):
                        ep_meta_list.append({k: v[i] for k, v in d.items()})
            ep_meta_list.sort(key=lambda r: r.get("episode_index", 0))

            # Read data frames
            data_dir = sdir / "data"
            src_frames = {}
            if data_dir.exists():
                for pf in sorted(data_dir.rglob("*.parquet")):
                    try:
                        tbl = pq.read_table(str(pf))
                    except Exception:
                        continue
                    d = tbl.to_pydict()
                    for i in range(len(d.get("episode_index", []))):
                        ep = d["episode_index"][i]
                        row = {}
                        for k, v in d.items():
                            val = v[i]
                            if hasattr(val, "tolist"):
                                val = val.tolist()
                            row[k] = val
                        src_frames.setdefault(ep, []).append(row)
            for ep in src_frames:
                src_frames[ep].sort(key=lambda r: r.get("frame_index", 0))

            src_tasks = []
            tp = sdir / "meta" / "tasks.parquet"
            if tp.exists():
                try:
                    src_tasks = pq.read_table(str(tp)).to_pydict().get("task", [])
                except Exception:
                    pass

            for old_ep in sorted(src_frames.keys()):
                frames = src_frames[old_ep]
                ep_meta = next((m for m in ep_meta_list if m["episode_index"] == old_ep), None)
                ep_start = global_frame_idx

                for fi, row in enumerate(frames):
                    new_row = dict(row)
                    new_row["episode_index"] = new_ep_idx
                    new_row["frame_index"] = fi
                    new_row["index"] = global_frame_idx
                    old_ti = row.get("task_index", 0)
                    if old_ti < len(src_tasks) and src_tasks[old_ti] in task_to_idx:
                        new_row["task_index"] = task_to_idx[src_tasks[old_ti]]
                    else:
                        new_row["task_index"] = 0
                    combined_data.append(new_row)
                    global_frame_idx += 1

                new_meta = {"episode_index": new_ep_idx, "length": len(frames)}
                if ep_meta and "tasks" in ep_meta:
                    new_meta["tasks"] = ep_meta["tasks"]
                elif src_tasks:
                    new_meta["tasks"] = [src_tasks[0]]
                else:
                    new_meta["tasks"] = all_tasks[:1] if all_tasks else [""]

                # Discover camera keys from episode metadata
                if ep_meta:
                    cam_keys = set()
                    for k in ep_meta:
                        if k.startswith("videos/"):
                            cam_name = "/".join(k.split("/")[1:-1])
                            if cam_name:
                                cam_keys.add(cam_name)
                    for cam in cam_keys:
                        cam_key = f"videos/{cam}"
                        orig_chunk = ep_meta.get(f"{cam_key}/chunk_index", 0)
                        orig_file = ep_meta.get(f"{cam_key}/file_index", 0)
                        src_vid = sdir / "videos" / cam / f"chunk-{orig_chunk:03d}" / f"file-{orig_file:03d}.mp4"
                        new_meta[f"{cam_key}/from_timestamp"] = ep_meta.get(f"{cam_key}/from_timestamp", 0.0)
                        new_meta[f"{cam_key}/to_timestamp"] = ep_meta.get(f"{cam_key}/to_timestamp", 0.0)
                        new_meta[f"{cam_key}/chunk_index"] = 0
                        new_meta[f"{cam_key}/file_index"] = 0
                        new_meta[f"_src_vid_{cam}"] = str(src_vid) if src_vid.exists() else None

                new_meta["data/chunk_index"] = 0
                new_meta["data/file_index"] = 0
                new_meta["dataset_from_index"] = ep_start
                new_meta["dataset_to_index"] = global_frame_idx
                combined_ep.append(new_meta)
                new_ep_idx += 1

        if not combined_data:
            return {"ok": False, "error": "No data found"}

        # Create dirs
        new_dir.mkdir(parents=True)
        (new_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (new_dir / "data" / "chunk-000").mkdir(parents=True)

        # Copy videos — discover camera names from metadata
        all_cam_keys = set()
        for em in combined_ep:
            for k in list(em.keys()):
                if k.startswith("_src_vid_"):
                    all_cam_keys.add(k[len("_src_vid_"):])

        for cam in all_cam_keys:
            vid_dir = new_dir / "videos" / cam / "chunk-000"
            vid_dir.mkdir(parents=True)
            src_vids, src_set = [], set()
            for em in combined_ep:
                sv = em.get(f"_src_vid_{cam}")
                if sv and sv not in src_set:
                    src_set.add(sv)
                    src_vids.append(sv)
            vid_map = {}
            for fi, sv in enumerate(src_vids):
                dst = vid_dir / f"file-{fi:03d}.mp4"
                shutil.copy2(sv, str(dst))
                vid_map[sv] = fi
            cam_key = f"videos/{cam}"
            for em in combined_ep:
                sv = em.pop(f"_src_vid_{cam}", None)
                if sv and sv in vid_map:
                    em[f"{cam_key}/file_index"] = vid_map[sv]

        # Clean remaining _src_vid_ keys
        for em in combined_ep:
            for k in [k for k in em if k.startswith("_src_vid_")]:
                del em[k]

        # Write data parquet
        cols = {k: [r[k] for r in combined_data] for k in combined_data[0]}
        pq.write_table(pa.table(cols), str(new_dir / "data" / "chunk-000" / "file-000.parquet"))

        # Write episode metadata
        cols = {k: [r.get(k) for r in combined_ep] for k in combined_ep[0]}
        pq.write_table(pa.table(cols), str(new_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))

        # Write tasks parquet
        if all_tasks:
            pq.write_table(pa.table({"task": all_tasks}), str(new_dir / "meta" / "tasks.parquet"))

        # Write info.json
        new_info = dict(base_info)
        new_info["total_episodes"] = new_ep_idx
        new_info["total_frames"] = len(combined_data)
        new_info["total_tasks"] = len(all_tasks)
        new_info["splits"] = {"train": f"0:{new_ep_idx}"}
        (new_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

        return {"ok": True, "name": new_name, "episodes": new_ep_idx, "frames": len(combined_data), "tasks": all_tasks}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Import dataset (copy from external path into ./data)
# ---------------------------------------------------------------------------
import_status = {"running": False, "progress": 0, "status": "", "error": None, "name": None}


@app.post("/api/datasets/import")
async def import_dataset(request: Request):
    """Import a dataset by copying from an external folder into ./data."""
    body = await request.json()
    source_path = body.get("path", "").strip()
    custom_name = body.get("name", "").strip()

    if not source_path:
        return {"ok": False, "error": "Path is required"}

    src = Path(source_path).resolve()
    if not src.exists():
        return {"ok": False, "error": f"Path does not exist: {src}"}
    if not src.is_dir():
        return {"ok": False, "error": "Path must be a directory"}

    # Check if it's a single dataset (has meta/info.json) or a folder of datasets
    datasets_to_import = []
    if (src / "meta" / "info.json").exists():
        # Single dataset
        name = custom_name or src.name
        datasets_to_import.append({"src": src, "name": name})
    else:
        # Folder containing multiple datasets
        for child in sorted(src.iterdir()):
            if child.is_dir() and (child / "meta" / "info.json").exists():
                datasets_to_import.append({"src": child, "name": child.name})

    if not datasets_to_import:
        return {"ok": False, "error": "No valid LeRobot datasets found (requires meta/info.json)"}

    if import_status["running"]:
        return {"ok": False, "error": "An import is already in progress"}

    # Run copy in background thread
    import_status.update({"running": True, "progress": 0, "status": "Starting...", "error": None, "name": None})

    def do_import():
        try:
            total = len(datasets_to_import)
            imported = []
            skipped = []
            for i, ds in enumerate(datasets_to_import):
                name = ds["name"]
                # Sanitize name
                name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
                dst = DATA_DIR / name
                import_status["status"] = f"Copying {name} ({i+1}/{total})..."
                import_status["progress"] = int((i / total) * 100)

                if dst.exists():
                    skipped.append(name)
                    continue

                shutil.copytree(str(ds["src"]), str(dst))
                imported.append(name)

            import_status["progress"] = 100
            parts = []
            if imported:
                parts.append(f"Imported {len(imported)} dataset(s): {', '.join(imported)}")
            if skipped:
                parts.append(f"Skipped {len(skipped)} (already exist): {', '.join(skipped)}")
            import_status["status"] = ". ".join(parts) if parts else "Nothing to import"
            import_status["name"] = imported[0] if len(imported) == 1 else None
        except Exception as e:
            import_status["error"] = str(e)
            import_status["status"] = f"Error: {e}"
        finally:
            import_status["running"] = False

    threading.Thread(target=do_import, daemon=True).start()
    return {"ok": True, "count": len(datasets_to_import)}


@app.get("/api/import/status")
async def get_import_status():
    """Get current import progress."""
    return import_status


# ---------------------------------------------------------------------------
# Browse filesystem (for import dialog)
# ---------------------------------------------------------------------------
@app.get("/api/browse")
async def browse_filesystem(path: str = "~"):
    """List directories for the import file browser."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {"ok": False, "error": "Path does not exist"}
    if not target.is_dir():
        target = target.parent

    entries = []
    try:
        for child in sorted(target.iterdir()):
            if child.name.startswith('.'):
                continue
            if child.is_dir():
                is_dataset = (child / "meta" / "info.json").exists()
                has_datasets = False
                if not is_dataset:
                    try:
                        has_datasets = any(
                            (gc / "meta" / "info.json").exists()
                            for gc in child.iterdir() if gc.is_dir()
                        )
                    except PermissionError:
                        pass
                entries.append({
                    "name": child.name,
                    "path": str(child),
                    "is_dataset": is_dataset,
                    "has_datasets": has_datasets,
                })
    except PermissionError:
        return {"ok": False, "error": "Permission denied"}

    return {
        "ok": True,
        "current": str(target),
        "parent": str(target.parent) if target.parent != target else None,
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Upload dataset (drag & drop from browser)
# ---------------------------------------------------------------------------
@app.post("/api/datasets/upload")
async def upload_dataset(
    folder_name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Receive files from browser drag & drop and save as a dataset."""
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', folder_name)
    dst = DATA_DIR / name
    if dst.exists():
        return {"ok": False, "error": f"'{name}' already exists"}

    try:
        for f in files:
            # f.filename contains the relative path (e.g. "meta/info.json")
            rel = f.filename or ""
            file_dst = dst / rel
            file_dst.parent.mkdir(parents=True, exist_ok=True)
            content = await f.read()
            file_dst.write_bytes(content)

        # Verify it's a valid dataset
        if (dst / "meta" / "info.json").exists():
            return {"ok": True, "name": name}

        # Check if sub-datasets were uploaded
        sub = [d.name for d in dst.iterdir() if d.is_dir() and (d / "meta" / "info.json").exists()]
        if sub:
            return {"ok": True, "name": name, "sub_datasets": sub}

        # Not valid — clean up
        shutil.rmtree(dst)
        return {"ok": False, "error": "No valid LeRobot dataset found (requires meta/info.json)"}
    except Exception as e:
        if dst.exists():
            shutil.rmtree(dst)
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = ROOT / "config.json"
DEFAULT_CONFIG = {
    "hf_username": "",
    "lerobot_path": "~/lerobot",
}


def load_config():
    if CONFIG_PATH.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text())}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


@app.get("/api/config")
async def get_config():
    return {"ok": True, "config": load_config()}


@app.post("/api/config")
async def save_config(request: Request):
    body = await request.json()
    cfg = load_config()
    cfg.update(body)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return {"ok": True, "config": cfg}


# ---------------------------------------------------------------------------
# Remove idle frames
# ---------------------------------------------------------------------------
def _auto_detect_segments(frames):
    """Auto-detect idle/movement segments using velocity-based method (mirrors frontend)."""
    n = len(frames)
    if n < 5:
        return [{"start": 0, "end": n - 1, "phase": "movement"}]

    # Compute velocity (sum of absolute action diffs)
    action_key = "action"
    action_len = len(frames[0].get(action_key, [])) if frames[0].get(action_key) else 6
    vel = [0.0]
    for i in range(1, n):
        v = 0.0
        a_cur = frames[i].get(action_key, [])
        a_prev = frames[i - 1].get(action_key, [])
        for j in range(min(action_len, len(a_cur), len(a_prev))):
            v += abs((a_cur[j] if j < len(a_cur) else 0) - (a_prev[j] if j < len(a_prev) else 0))
        vel.append(v)

    # Smooth
    hw = 5
    sv = []
    for i in range(n):
        lo, hi = max(0, i - hw), min(n - 1, i + hw)
        sv.append(sum(vel[lo:hi + 1]) / (hi - lo + 1))

    sorted_sv = sorted(sv)
    vel_thresh = max(sorted_sv[int(n * 0.85)] * 0.12, 0.3)
    labels = ["movement" if v > vel_thresh else "idle" for v in sv]

    # Build segments
    segs = []
    seg_start = 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[seg_start]:
            segs.append({"start": seg_start, "end": i - 1, "phase": labels[seg_start]})
            seg_start = i

    # Merge tiny segments
    for _ in range(3):
        i = len(segs) - 1
        while i >= 0:
            if segs[i]["end"] - segs[i]["start"] < 4 and len(segs) > 1:
                if i > 0:
                    segs[i - 1]["end"] = segs[i]["end"]
                else:
                    segs[1]["start"] = segs[0]["start"]
                segs.pop(i)
            i -= 1

    # Idle only at start/end — convert middle idle segments to movement
    for i in range(len(segs)):
        if segs[i]["phase"] == "idle" and i > 0 and i < len(segs) - 1:
            segs[i]["phase"] = "movement"
            segs[i]["label"] = "movement"
    # Merge consecutive movement segments
    merged = [segs[0]]
    for s in segs[1:]:
        if s["phase"] == merged[-1]["phase"]:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(s)

    return merged


@app.post("/api/datasets/{name}/remove-idle")
async def remove_idle_dataset(name: str, request: Request):
    """Create new dataset with idle frames removed from start/end of each episode."""
    body = await request.json()
    custom_name = body.get("new_name", "").strip().replace(" ", "-").replace("/", "-")

    src_dir = DATA_DIR / name
    new_name = custom_name if custom_name else name + "_removeidle"
    new_dir = DATA_DIR / new_name

    if not (src_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Dataset not found"}
    if new_dir.exists():
        return {"ok": False, "error": f"'{new_name}' already exists"}

    # Load saved segments
    seg_path = src_dir / "meta" / "segments.json"
    saved_segments = {}
    if seg_path.exists():
        try:
            saved_segments = json.loads(seg_path.read_text())
        except Exception:
            pass

    try:
        src_info = json.loads((src_dir / "meta" / "info.json").read_text())
        all_tasks = []
        tp = src_dir / "meta" / "tasks.parquet"
        if tp.exists():
            try:
                all_tasks = pq.read_table(str(tp)).to_pydict().get("task", [])
            except Exception:
                pass

        # Read episode metadata
        ep_meta_list = []
        ep_dir = src_dir / "meta" / "episodes"
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep_meta_list.append({k: v[i] for k, v in d.items()})
        ep_meta_list.sort(key=lambda r: r.get("episode_index", 0))

        # Read data grouped by episode
        src_frames = {}
        data_dir = src_dir / "data"
        if data_dir.exists():
            for pf in sorted(data_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep = d["episode_index"][i]
                    row = {}
                    for k, v in d.items():
                        val = v[i]
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        row[k] = val
                    src_frames.setdefault(ep, []).append(row)
        for ep in src_frames:
            src_frames[ep].sort(key=lambda r: r.get("frame_index", 0))

        combined_data, combined_ep = [], []
        new_ep_idx, global_frame_idx, total_removed = 0, 0, 0

        for old_ep in range(src_info.get("total_episodes", len(src_frames))):
            frames = src_frames.get(old_ep, [])
            if not frames:
                continue

            # Use saved segments or auto-detect to find movement frames
            ep_segs = saved_segments.get(str(old_ep))
            if not ep_segs:
                # Auto-detect segments (same algorithm as frontend)
                ep_segs = _auto_detect_segments(frames)

            # Keep only movement segments
            keep_ranges = []
            for seg in ep_segs:
                if seg.get("phase") == "movement":
                    keep_ranges.append((seg["start"], seg["end"]))
            if keep_ranges:
                trimmed = []
                for s, e in keep_ranges:
                    trimmed.extend(frames[s:e + 1])
                total_removed += len(frames) - len(trimmed)
            else:
                total_removed += len(frames)
                continue
            if not trimmed:
                continue

            ep_meta = next((m for m in ep_meta_list if m["episode_index"] == old_ep), None)
            ep_start = global_frame_idx
            t0 = trimmed[0].get("timestamp", 0.0)

            for fi, row in enumerate(trimmed):
                new_row = dict(row)
                new_row["episode_index"] = new_ep_idx
                new_row["frame_index"] = fi
                new_row["index"] = global_frame_idx
                new_row["timestamp"] = row.get("timestamp", 0.0) - t0
                combined_data.append(new_row)
                global_frame_idx += 1

            new_meta = {"episode_index": new_ep_idx, "length": len(trimmed),
                        "data/chunk_index": 0, "data/file_index": 0,
                        "dataset_from_index": ep_start, "dataset_to_index": global_frame_idx}
            if ep_meta and "tasks" in ep_meta:
                new_meta["tasks"] = ep_meta["tasks"]
            elif all_tasks:
                new_meta["tasks"] = [all_tasks[0]]

            # Copy video references with adjusted timestamps
            if ep_meta:
                cam_keys = set()
                for k in ep_meta:
                    if k.startswith("videos/"):
                        parts = k.split("/")
                        cam_name = "/".join(parts[1:-1])
                        if cam_name:
                            cam_keys.add(cam_name)
                for cam in cam_keys:
                    ck = f"videos/{cam}"
                    orig_chunk = ep_meta.get(f"{ck}/chunk_index", 0)
                    orig_file = ep_meta.get(f"{ck}/file_index", 0)
                    src_vid = src_dir / "videos" / cam / f"chunk-{orig_chunk:03d}" / f"file-{orig_file:03d}.mp4"
                    new_meta[f"{ck}/from_timestamp"] = ep_meta.get(f"{ck}/from_timestamp", 0.0) + t0
                    new_meta[f"{ck}/to_timestamp"] = ep_meta.get(f"{ck}/to_timestamp", 0.0)
                    new_meta[f"{ck}/chunk_index"] = 0
                    new_meta[f"{ck}/file_index"] = 0
                    new_meta[f"_src_vid_{cam}"] = str(src_vid) if src_vid.exists() else None

            combined_ep.append(new_meta)
            new_ep_idx += 1

        if not combined_data:
            return {"ok": False, "error": "No data after removing idle"}

        # Write output
        new_dir.mkdir(parents=True)
        (new_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (new_dir / "data" / "chunk-000").mkdir(parents=True)

        # Copy videos
        all_cams = set()
        for em in combined_ep:
            for k in list(em.keys()):
                if k.startswith("_src_vid_"):
                    all_cams.add(k[len("_src_vid_"):])
        for cam in all_cams:
            vid_dir = new_dir / "videos" / cam / "chunk-000"
            vid_dir.mkdir(parents=True)
            src_vids, src_set = [], set()
            for em in combined_ep:
                sv = em.get(f"_src_vid_{cam}")
                if sv and sv not in src_set:
                    src_set.add(sv)
                    src_vids.append(sv)
            vid_map = {}
            for fi, sv in enumerate(src_vids):
                shutil.copy2(sv, str(vid_dir / f"file-{fi:03d}.mp4"))
                vid_map[sv] = fi
            for em in combined_ep:
                sv = em.pop(f"_src_vid_{cam}", None)
                if sv and sv in vid_map:
                    em[f"videos/{cam}/file_index"] = vid_map[sv]

        for em in combined_ep:
            for k in [k for k in em if k.startswith("_")]:
                del em[k]

        cols = {k: [r[k] for r in combined_data] for k in combined_data[0]}
        pq.write_table(pa.table(cols), str(new_dir / "data" / "chunk-000" / "file-000.parquet"))
        cols = {k: [r.get(k) for r in combined_ep] for k in combined_ep[0]}
        pq.write_table(pa.table(cols), str(new_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))
        if all_tasks:
            pq.write_table(pa.table({"task": all_tasks}), str(new_dir / "meta" / "tasks.parquet"))

        new_info = dict(src_info)
        new_info["total_episodes"] = new_ep_idx
        new_info["total_frames"] = len(combined_data)
        new_info["splits"] = {"train": f"0:{new_ep_idx}"}
        (new_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

        return {"ok": True, "name": new_name, "episodes": new_ep_idx,
                "frames": len(combined_data), "removed_frames": total_removed}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Random subset
# ---------------------------------------------------------------------------
@app.post("/api/datasets/{name}/random-subset")
async def random_subset(name: str, request: Request):
    """Create new dataset by randomly sampling episodes."""
    import random as _random

    body = await request.json()
    num_episodes = int(body.get("num_episodes", 10))
    new_name = body.get("new_name", "").strip().replace(" ", "-")
    if not new_name:
        return {"ok": False, "error": "Enter a name"}

    src_dir = DATA_DIR / name
    new_dir = DATA_DIR / new_name
    if not (src_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Dataset not found"}
    if new_dir.exists():
        return {"ok": False, "error": f"'{new_name}' already exists"}

    try:
        src_info = json.loads((src_dir / "meta" / "info.json").read_text())
        total_eps = src_info.get("total_episodes", 0)
        if num_episodes > total_eps:
            return {"ok": False, "error": f"Requested {num_episodes} but only {total_eps} available"}

        selected = sorted(_random.sample(range(total_eps), num_episodes))

        all_tasks = []
        tp = src_dir / "meta" / "tasks.parquet"
        if tp.exists():
            try:
                all_tasks = pq.read_table(str(tp)).to_pydict().get("task", [])
            except Exception:
                pass

        ep_meta_list = []
        ep_dir = src_dir / "meta" / "episodes"
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep_meta_list.append({k: v[i] for k, v in d.items()})
        ep_meta_list.sort(key=lambda r: r.get("episode_index", 0))

        src_frames = {}
        data_dir = src_dir / "data"
        if data_dir.exists():
            for pf in sorted(data_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep = d["episode_index"][i]
                    if ep not in selected:
                        continue
                    row = {}
                    for k, v in d.items():
                        val = v[i]
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        row[k] = val
                    src_frames.setdefault(ep, []).append(row)
        for ep in src_frames:
            src_frames[ep].sort(key=lambda r: r.get("frame_index", 0))

        combined_data, combined_ep = [], []
        new_ep_idx, global_frame_idx = 0, 0

        for old_ep in selected:
            frames = src_frames.get(old_ep, [])
            if not frames:
                continue
            ep_meta = next((m for m in ep_meta_list if m["episode_index"] == old_ep), None)
            ep_start = global_frame_idx

            for fi, row in enumerate(frames):
                new_row = dict(row)
                new_row["episode_index"] = new_ep_idx
                new_row["frame_index"] = fi
                new_row["index"] = global_frame_idx
                combined_data.append(new_row)
                global_frame_idx += 1

            new_meta = {"episode_index": new_ep_idx, "length": len(frames),
                        "data/chunk_index": 0, "data/file_index": 0,
                        "dataset_from_index": ep_start, "dataset_to_index": global_frame_idx}
            if ep_meta and "tasks" in ep_meta:
                new_meta["tasks"] = ep_meta["tasks"]
            elif all_tasks:
                new_meta["tasks"] = [all_tasks[0]]

            if ep_meta:
                cam_keys = set()
                for k in ep_meta:
                    if k.startswith("videos/"):
                        cam_name = "/".join(k.split("/")[1:-1])
                        if cam_name:
                            cam_keys.add(cam_name)
                for cam in cam_keys:
                    ck = f"videos/{cam}"
                    orig_chunk = ep_meta.get(f"{ck}/chunk_index", 0)
                    orig_file = ep_meta.get(f"{ck}/file_index", 0)
                    src_vid = src_dir / "videos" / cam / f"chunk-{orig_chunk:03d}" / f"file-{orig_file:03d}.mp4"
                    new_meta[f"{ck}/from_timestamp"] = ep_meta.get(f"{ck}/from_timestamp", 0.0)
                    new_meta[f"{ck}/to_timestamp"] = ep_meta.get(f"{ck}/to_timestamp", 0.0)
                    new_meta[f"{ck}/chunk_index"] = 0
                    new_meta[f"{ck}/file_index"] = 0
                    new_meta[f"_src_vid_{cam}"] = str(src_vid) if src_vid.exists() else None

            combined_ep.append(new_meta)
            new_ep_idx += 1

        if not combined_data:
            return {"ok": False, "error": "No data found"}

        new_dir.mkdir(parents=True)
        (new_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (new_dir / "data" / "chunk-000").mkdir(parents=True)

        all_cams = set()
        for em in combined_ep:
            for k in list(em.keys()):
                if k.startswith("_src_vid_"):
                    all_cams.add(k[len("_src_vid_"):])
        for cam in all_cams:
            vid_dir = new_dir / "videos" / cam / "chunk-000"
            vid_dir.mkdir(parents=True)
            src_vids, src_set = [], set()
            for em in combined_ep:
                sv = em.get(f"_src_vid_{cam}")
                if sv and sv not in src_set:
                    src_set.add(sv)
                    src_vids.append(sv)
            vid_map = {}
            for fi, sv in enumerate(src_vids):
                shutil.copy2(sv, str(vid_dir / f"file-{fi:03d}.mp4"))
                vid_map[sv] = fi
            for em in combined_ep:
                sv = em.pop(f"_src_vid_{cam}", None)
                if sv and sv in vid_map:
                    em[f"videos/{cam}/file_index"] = vid_map[sv]

        for em in combined_ep:
            for k in [k for k in em if k.startswith("_")]:
                del em[k]

        cols = {k: [r[k] for r in combined_data] for k in combined_data[0]}
        pq.write_table(pa.table(cols), str(new_dir / "data" / "chunk-000" / "file-000.parquet"))
        cols = {k: [r.get(k) for r in combined_ep] for k in combined_ep[0]}
        pq.write_table(pa.table(cols), str(new_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))
        if all_tasks:
            pq.write_table(pa.table({"task": all_tasks}), str(new_dir / "meta" / "tasks.parquet"))

        new_info = dict(src_info)
        new_info["total_episodes"] = new_ep_idx
        new_info["total_frames"] = len(combined_data)
        new_info["splits"] = {"train": f"0:{new_ep_idx}"}
        (new_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

        return {"ok": True, "name": new_name, "episodes": new_ep_idx, "frames": len(combined_data)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------

augment_state = {"running": False, "progress": 0, "status": "", "error": None}

_INTENSITY_SCALE = {1: 0.25, 2: 0.5, 3: 1.0, 4: 1.5, 5: 2.0}


def _augp_camera(opts, intensity=3, fixed=None):
    s = _INTENSITY_SCALE.get(intensity, 1.0)
    p = {}
    if fixed:
        if opts.get("perspective") and fixed.get("persp_mag") is not None:
            m = float(fixed["persp_mag"])
            p["persp"] = np.array([[m, m], [-m, m], [m, -m], [-m, -m]])
        if opts.get("affine"):
            p["tx"] = float(fixed.get("tx", 0))
            p["ty"] = float(fixed.get("ty", 0))
            p["shear"] = float(fixed.get("shear", 0))
            p["scale"] = float(fixed.get("scale", 1.0))
        if opts.get("rotation") and fixed.get("angle") is not None:
            p["angle"] = float(fixed["angle"])
        return p or None
    if opts.get("perspective"):
        p["persp"] = np.random.uniform(-0.05 * s, 0.05 * s, (4, 2))
    if opts.get("affine"):
        p["tx"] = random.uniform(-0.04 * s, 0.04 * s)
        p["ty"] = random.uniform(-0.04 * s, 0.04 * s)
        p["shear"] = random.uniform(-0.03 * s, 0.03 * s)
        p["scale"] = random.uniform(1 - 0.05 * s, 1 + 0.05 * s)
    if opts.get("rotation"):
        p["angle"] = random.uniform(-4 * s, 4 * s)
    return p or None


def _augp_light(opts, intensity=3, fixed=None):
    s = _INTENSITY_SCALE.get(intensity, 1.0)
    p = {}
    if fixed:
        if opts.get("brightness") and fixed.get("brightness") is not None:
            p["brightness"] = float(fixed["brightness"])
        if opts.get("contrast") and fixed.get("contrast") is not None:
            p["contrast"] = float(fixed["contrast"])
        if opts.get("saturation") and fixed.get("saturation") is not None:
            p["saturation"] = float(fixed["saturation"])
        if opts.get("color_jitter") and fixed.get("hue") is not None:
            p["hue"] = int(fixed["hue"])
            p["sat_off"] = int(fixed.get("sat_off", 0))
            p["val_off"] = int(fixed.get("val_off", 0))
        if opts.get("shadow") and fixed.get("shadow_a") is not None:
            p["shadow_dir"] = fixed.get("shadow_dir", "left")
            p["shadow_a"] = float(fixed["shadow_a"])
        if opts.get("noise") and fixed.get("noise_s") is not None:
            p["noise_s"] = float(fixed["noise_s"])
        if opts.get("blur") and fixed.get("blur_s") is not None:
            p["blur_k"] = int(fixed.get("blur_k", 3))
            p["blur_s"] = float(fixed["blur_s"])
        return p or None
    if opts.get("brightness"):
        p["brightness"] = random.uniform(1 - 0.25 * s, 1 + 0.25 * s)
    if opts.get("contrast"):
        p["contrast"] = random.uniform(1 - 0.25 * s, 1 + 0.25 * s)
    if opts.get("saturation"):
        p["saturation"] = random.uniform(1 - 0.35 * s, 1 + 0.35 * s)
    if opts.get("color_jitter"):
        h = int(8 * s)
        sv = int(15 * s)
        p["hue"] = random.randint(-h, h)
        p["sat_off"] = random.randint(-sv, sv)
        p["val_off"] = random.randint(-sv, sv)
    if opts.get("shadow"):
        p["shadow_dir"] = random.choice(["left", "right", "top", "bottom"])
        p["shadow_a"] = random.uniform(0.15 * s, 0.4 * s)
    if opts.get("noise"):
        p["noise_s"] = random.uniform(4 * s, 12 * s)
    if opts.get("blur"):
        p["blur_k"] = random.choice([3, 5] if s <= 1.0 else [3, 5, 7])
        p["blur_s"] = random.uniform(0.4 * s, 1.2 * s)
    return p or None


def _augp_robot(opts, n_frames, fps, intensity=3, fixed=None):
    s = _INTENSITY_SCALE.get(intensity, 1.0)
    p = {}
    if fixed:
        if opts.get("random_start") and n_frames > 30 and fixed.get("trim") is not None:
            p["trim"] = max(0, int(fixed["trim"]))
        if opts.get("initial_noise") and fixed.get("offset_sigma") is not None:
            sigma = float(fixed["offset_sigma"])
            p["joint_offset"] = [sigma * (1 if i % 2 == 0 else -1) for i in range(5)] + [0.0]
        if opts.get("trajectory_jitter") and fixed.get("jitter_s") is not None:
            p["jitter_s"] = float(fixed["jitter_s"])
        p["fps"] = fps
        return p or None
    if opts.get("random_start") and n_frames > 30:
        p["trim"] = random.randint(1, max(1, int(n_frames * 0.12 * s)))
    if opts.get("initial_noise"):
        p["joint_offset"] = [random.gauss(0, 1.5 * s) for _ in range(5)] + [0.0]
    if opts.get("trajectory_jitter"):
        p["jitter_s"] = random.uniform(0.3 * s, 1.0 * s)
    p["fps"] = fps
    return p or None


_PICK_SYNS = ["pick up", "grab", "lift", "take", "grasp"]
_PLACE_SYNS = ["place", "put", "set down", "drop", "lay"]
_PREP_SYNS = {"to": ["to", "into", "onto", "in", "on"],
              "from": ["from", "out of", "off"]}


def _aug_task_text(task, opts):
    if not opts or not opts.get("enabled"):
        return task
    result = task
    do_syn = opts.get("synonym") or opts.get("paraphrase")
    if do_syn:
        for verb, syns in [("pick up", _PICK_SYNS), ("place", _PLACE_SYNS)]:
            if verb in result.lower():
                rep = random.choice(syns)
                result = re.sub(re.escape(verb), rep, result, count=1, flags=re.I)
        for prep, syns in _PREP_SYNS.items():
            pat = rf"\b{prep}\b"
            if re.search(pat, result, re.I):
                result = re.sub(pat, random.choice(syns), result, count=1, flags=re.I)
    if opts.get("typo") and random.random() < 0.35:
        words = result.split()
        if len(words) > 2:
            idx = random.randint(1, len(words) - 1)
            w = list(words[idx])
            if len(w) > 2:
                pos = random.randint(0, len(w) - 2)
                op = random.choice(["swap", "del", "dup"])
                if op == "swap":
                    w[pos], w[pos + 1] = w[pos + 1], w[pos]
                elif op == "del":
                    w.pop(pos)
                else:
                    w.insert(pos, w[pos])
                words[idx] = "".join(w)
            result = " ".join(words)
    return result


def _aug_frame(frame, cam_p, light_p):
    import cv2
    h, w = frame.shape[:2]
    f = frame
    if cam_p:
        if "persp" in cam_p:
            pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            pts2 = (pts1 + cam_p["persp"] * [w, h]).astype(np.float32)
            M = cv2.getPerspectiveTransform(pts1, pts2)
            f = cv2.warpPerspective(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if "tx" in cam_p:
            M = np.float32([[cam_p.get("scale", 1), cam_p.get("shear", 0), cam_p["tx"] * w],
                            [0, cam_p.get("scale", 1), cam_p.get("ty", 0) * h]])
            f = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if "angle" in cam_p:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), cam_p["angle"], 1.0)
            f = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if light_p:
        ff = f.astype(np.float32)
        if "brightness" in light_p:
            ff *= light_p["brightness"]
        if "contrast" in light_p:
            mean = ff.mean(axis=(0, 1), keepdims=True)
            ff = (ff - mean) * light_p["contrast"] + mean
        ff = np.clip(ff, 0, 255).astype(np.uint8)
        if "saturation" in light_p:
            hsv = cv2.cvtColor(ff, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= light_p["saturation"]
            ff = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        if "hue" in light_p:
            hsv = cv2.cvtColor(ff, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + light_p["hue"]) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] + light_p["sat_off"], 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + light_p["val_off"], 0, 255)
            ff = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if "shadow_a" in light_p:
            hh, ww = ff.shape[:2]
            a = light_p["shadow_a"]
            d = light_p["shadow_dir"]
            if d in ("left", "right"):
                g = np.linspace(1 - a, 1, ww) if d == "left" else np.linspace(1, 1 - a, ww)
                shadow = np.tile(g, (hh, 1))
            else:
                g = np.linspace(1 - a, 1, hh) if d == "top" else np.linspace(1, 1 - a, hh)
                shadow = np.tile(g.reshape(-1, 1), (1, ww))
            ff = (ff.astype(np.float32) * shadow[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)
        if "noise_s" in light_p:
            noise = np.random.normal(0, light_p["noise_s"], ff.shape)
            ff = np.clip(ff.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if "blur_k" in light_p:
            ff = cv2.GaussianBlur(ff, (light_p["blur_k"], light_p["blur_k"]), light_p["blur_s"])
        f = ff
    return f


def _aug_episode_data(rows, robot_p):
    if not robot_p:
        return rows
    rows = [dict(r) for r in rows]
    trim = robot_p.get("trim", 0)
    if trim > 0 and trim < len(rows):
        rows = rows[trim:]
    fps = robot_p.get("fps", 30)
    offset = robot_p.get("joint_offset", None)
    jitter = robot_p.get("jitter_s", 0)
    for i, r in enumerate(rows):
        r["frame_index"] = i
        r["timestamp"] = i / fps
        action = list(r.get("action", []))
        state = list(r.get("observation.state", []))
        for j in range(min(len(action), 5)):
            if offset:
                action[j] += offset[j]
                state[j] += offset[j]
            if jitter > 0:
                jit = random.gauss(0, jitter)
                action[j] += jit
                state[j] += jit
        r["action"] = action
        r["observation.state"] = state
    return rows


def _aug_process_video(src_path, dst_path, from_ts, to_ts, fps, cam_p, light_p, vid_w, vid_h):
    duration = to_ts - from_ts
    if duration <= 0 or not Path(src_path).exists():
        return 0.0, 0.0, 0
    read_cmd = [
        "ffmpeg", "-v", "quiet",
        "-ss", str(from_ts), "-i", str(src_path),
        "-t", str(duration),
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{vid_w}x{vid_h}", "pipe:1",
    ]
    write_cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{vid_w}x{vid_h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23", str(dst_path),
    ]
    try:
        reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE)
        writer = subprocess.Popen(write_cmd, stdin=subprocess.PIPE)
        frame_size = vid_w * vid_h * 3
        count = 0
        while True:
            raw = reader.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(vid_h, vid_w, 3)
            if cam_p or light_p:
                frame = _aug_frame(frame, cam_p, light_p)
            writer.stdin.write(frame.tobytes())
            count += 1
        reader.stdout.close()
        writer.stdin.close()
        reader.wait()
        writer.wait()
        return 0.0, count / fps if fps > 0 else 0.0, count
    except Exception as e:
        print(f"[Augment] Video processing error: {e}")
        return 0.0, 0.0, 0


def _get_camera_keys(info):
    """Extract camera feature keys from dataset info."""
    cams = []
    for k, v in info.get("features", {}).items():
        if k.startswith("observation.images.") and isinstance(v, dict) and v.get("dtype") == "video":
            cams.append(k)
    return cams


def _run_augmentation(src_name, new_name, target_episodes, techniques, intensity=3, mode="random", fixed_params=None):
    state = augment_state
    state["running"] = True
    state["progress"] = 0
    state["status"] = "Loading source dataset..."
    state["error"] = None

    try:
        src_dir = ROOT / "data" / src_name
        new_dir = ROOT / "data" / new_name
        info = json.loads((src_dir / "meta" / "info.json").read_text())
        fps = info.get("fps", 30)
        cam_keys = _get_camera_keys(info)

        # Get video dimensions from first camera feature
        vid_h, vid_w = 240, 320
        if cam_keys:
            shape = info.get("features", {}).get(cam_keys[0], {}).get("shape", [240, 320, 3])
            vid_h, vid_w = shape[0], shape[1]

        # Read tasks
        src_tasks = []
        tp = src_dir / "meta" / "tasks.parquet"
        if tp.exists():
            try:
                src_tasks = pq.read_table(str(tp)).to_pydict().get("task", [])
            except Exception:
                pass

        # Read episode metadata
        ep_meta_list = []
        ep_dir = src_dir / "meta" / "episodes"
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep_meta_list.append({k: v[i] for k, v in d.items()})
        ep_meta_list.sort(key=lambda r: r.get("episode_index", 0))

        # Read all data frames by episode
        src_frames = {}
        data_dir = src_dir / "data"
        if data_dir.exists():
            for pf in sorted(data_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep = d["episode_index"][i]
                    row = {}
                    for k, v in d.items():
                        val = v[i]
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        row[k] = val
                    src_frames.setdefault(ep, []).append(row)
        for ep in src_frames:
            src_frames[ep].sort(key=lambda r: r.get("frame_index", 0))

        n_src_eps = len(src_frames)
        if n_src_eps == 0:
            raise ValueError("Source dataset has no episodes")

        sorted_eps = sorted(src_frames.keys())
        if mode == "random":
            # Random: pick random source episode for each augmented copy
            ep_schedule = [random.choice(sorted_eps) for _ in range(target_episodes)]
        else:
            # Fix: evenly distribute copies across episodes
            ep_schedule = []
            base = target_episodes // n_src_eps
            remainder = target_episodes % n_src_eps
            for i, ep in enumerate(sorted_eps):
                n = base + (1 if i < remainder else 0)
                ep_schedule.extend([ep] * n)

        total_steps = target_episodes
        current_step = 0

        new_dir.mkdir(parents=True, exist_ok=True)
        (new_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (new_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in cam_keys:
            (new_dir / "videos" / cam / "chunk-000").mkdir(parents=True, exist_ok=True)

        combined_data = []
        combined_ep = []
        aug_ep_info = []  # per-episode augmentation info
        all_new_tasks = list(src_tasks)
        task_set = set(src_tasks)
        new_ep_idx = 0
        global_frame_idx = 0

        cam_opts = techniques.get("camera_shift", {})
        light_opts = techniques.get("light_quality", {})
        robot_opts = techniques.get("robot_noise", {})
        lang_opts = techniques.get("language", {})
        cam_fixed = fixed_params.get("camera") if fixed_params else None
        light_fixed = fixed_params.get("light") if fixed_params else None
        robot_fixed = fixed_params.get("robot") if fixed_params else None

        seen_eps = set()
        for sched_idx, old_ep in enumerate(ep_schedule):
                is_original = old_ep not in seen_eps
                seen_eps.add(old_ep)
                frames = src_frames[old_ep]
                ep_meta = next((m for m in ep_meta_list if m["episode_index"] == old_ep), None)

                state["status"] = f"Episode {new_ep_idx + 1}/{target_episodes} (src {old_ep})"
                current_step += 1
                state["progress"] = int(current_step / total_steps * 100)

                cam_p = None if is_original else (_augp_camera(cam_opts, intensity, cam_fixed) if cam_opts.get("enabled") else None)
                light_p = None if is_original else (_augp_light(light_opts, intensity, light_fixed) if light_opts.get("enabled") else None)
                robot_p = None if is_original else (
                    _augp_robot(robot_opts, len(frames), fps, intensity, robot_fixed) if robot_opts.get("enabled") else None
                )

                # Record per-episode augmentation info
                def _serialize_params(p):
                    if p is None: return None
                    out = {}
                    for k, v in p.items():
                        if hasattr(v, 'tolist'): out[k] = v.tolist()
                        else: out[k] = v
                    return out
                aug_ep_info.append({
                    "new_ep": new_ep_idx,
                    "source_ep": old_ep,
                    "is_original": is_original,
                    "camera": _serialize_params(cam_p),
                    "light": _serialize_params(light_p),
                    "robot": _serialize_params(robot_p),
                })

                aug_rows = _aug_episode_data(frames, robot_p) if not is_original else [dict(r) for r in frames]

                if is_original:
                    ep_task_list = ep_meta["tasks"] if ep_meta and "tasks" in ep_meta else src_tasks[:1]
                else:
                    orig_tasks = ep_meta["tasks"] if ep_meta and "tasks" in ep_meta else src_tasks[:1]
                    ep_task_list = [_aug_task_text(t, lang_opts) for t in orig_tasks]
                for t in ep_task_list:
                    if t not in task_set:
                        task_set.add(t)
                        all_new_tasks.append(t)
                task_to_idx = {t: i for i, t in enumerate(all_new_tasks)}

                ep_global_start = global_frame_idx
                for fi, row in enumerate(aug_rows):
                    nr = dict(row)
                    nr["episode_index"] = new_ep_idx
                    nr["frame_index"] = fi
                    nr["index"] = global_frame_idx
                    nr["timestamp"] = fi / fps
                    old_ti = row.get("task_index", 0)
                    if old_ti < len(ep_task_list):
                        nr["task_index"] = task_to_idx.get(ep_task_list[old_ti], 0)
                    elif ep_task_list:
                        nr["task_index"] = task_to_idx.get(ep_task_list[0], 0)
                    combined_data.append(nr)
                    global_frame_idx += 1

                new_meta = {
                    "episode_index": new_ep_idx,
                    "tasks": ep_task_list,
                    "length": len(aug_rows),
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": ep_global_start,
                    "dataset_to_index": global_frame_idx,
                }

                for cam in cam_keys:
                    cam_key = f"videos/{cam}"
                    orig_chunk = ep_meta.get(f"{cam_key}/chunk_index", 0) if ep_meta else 0
                    orig_file = ep_meta.get(f"{cam_key}/file_index", 0) if ep_meta else 0
                    orig_from = ep_meta.get(f"{cam_key}/from_timestamp", 0.0) if ep_meta else 0.0
                    orig_to = ep_meta.get(f"{cam_key}/to_timestamp", 0.0) if ep_meta else 0.0
                    src_vid = src_dir / "videos" / cam / f"chunk-{orig_chunk:03d}" / f"file-{orig_file:03d}.mp4"
                    dst_vid = new_dir / "videos" / cam / "chunk-000" / f"file-{new_ep_idx:03d}.mp4"
                    new_meta[f"{cam_key}/chunk_index"] = 0
                    new_meta[f"{cam_key}/file_index"] = new_ep_idx

                    if src_vid.exists():
                        new_from, new_to, _ = _aug_process_video(
                            src_vid, dst_vid, orig_from, orig_to, fps,
                            cam_p if not is_original else None,
                            light_p if not is_original else None,
                            vid_w, vid_h,
                        )
                    else:
                        new_from, new_to = 0.0, 0.0

                    trim = robot_p.get("trim", 0) if robot_p else 0
                    if trim > 0:
                        new_from += trim / fps
                    new_meta[f"{cam_key}/from_timestamp"] = new_from
                    new_meta[f"{cam_key}/to_timestamp"] = new_to

                if ep_meta:
                    for k, v in ep_meta.items():
                        if k.startswith("stats/"):
                            new_meta[k] = v
                new_meta["meta/episodes/chunk_index"] = 0
                new_meta["meta/episodes/file_index"] = 0

                combined_ep.append(new_meta)
                new_ep_idx += 1

        state["status"] = "Writing dataset..."
        state["progress"] = 95

        cols = {k: [r[k] for r in combined_data] for k in combined_data[0]}
        pq.write_table(pa.table(cols), str(new_dir / "data" / "chunk-000" / "file-000.parquet"))
        cols = {k: [r.get(k) for r in combined_ep] for k in combined_ep[0]}
        pq.write_table(pa.table(cols), str(new_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))
        if all_new_tasks:
            pq.write_table(pa.table({"task": all_new_tasks}), str(new_dir / "meta" / "tasks.parquet"))

        new_info = dict(info)
        new_info["total_episodes"] = new_ep_idx
        new_info["total_frames"] = len(combined_data)
        new_info["total_tasks"] = len(all_new_tasks)
        new_info["splits"] = {"train": f"0:{new_ep_idx}"}
        for feat_key in cam_keys:
            feat = new_info.get("features", {}).get(feat_key, {})
            if "info" in feat:
                feat["info"]["video.codec"] = "h264"
        (new_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

        # Compute stats
        state["status"] = "Computing stats..."
        stat_keys = ["action", "observation.state", "timestamp", "frame_index",
                     "episode_index", "index", "task_index"]
        stats = {}
        for key in stat_keys:
            vals = []
            for row in combined_data:
                v = row.get(key)
                if v is None:
                    continue
                vals.append(v if isinstance(v, (list, tuple)) else [v])
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float64)
            stats[key] = {
                "min": np.min(arr, axis=0).tolist(),
                "max": np.max(arr, axis=0).tolist(),
                "mean": np.mean(arr, axis=0).tolist(),
                "std": np.std(arr, axis=0).tolist(),
                "count": [len(arr)],
            }
        (new_dir / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

        # Save augmentation config to separate folder (keep dataset as pure LeRobot format)
        aug_meta = {
            "source_dataset": src_name,
            "target_episodes": target_episodes,
            "intensity": intensity,
            "mode": mode,
            "techniques": techniques,
            "fixed_params": fixed_params,
            "episodes": aug_ep_info,
        }
        (AUGMENT_CONFIG_DIR / f"{new_name}.json").write_text(json.dumps(aug_meta, indent=2))

        state["progress"] = 100
        state["status"] = f"Done! {new_ep_idx} episodes, {len(combined_data)} frames"
        print(f"[Augment] Created {new_name}: {new_ep_idx} eps, {len(combined_data)} frames")

    except Exception as e:
        import traceback
        traceback.print_exc()
        state["error"] = str(e)
        state["status"] = f"Error: {e}"
    finally:
        state["running"] = False


@app.get("/api/datasets/{name}/augment-config")
async def get_augment_config(name: str):
    """Get augmentation config if this dataset was created by augmentation."""
    cfg_path = AUGMENT_CONFIG_DIR / f"{name}.json"
    if not cfg_path.exists():
        return {"ok": True, "is_augmented": False}
    try:
        cfg = json.loads(cfg_path.read_text())
        return {"ok": True, "is_augmented": True, "config": cfg}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/datasets/{name}/augment-preview")
async def augment_preview(name: str, request: Request):
    """Generate preview of augmentation: original vs augmented first frames."""
    import cv2
    import base64

    body = await request.json()
    intensity = body.get("intensity", 3)
    techniques = body.get("techniques", {})
    num_previews = body.get("count", 5)

    src_dir = DATA_DIR / name
    if not (src_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Dataset not found"}

    info = json.loads((src_dir / "meta" / "info.json").read_text())
    cam_keys = _get_camera_keys(info)
    if not cam_keys:
        return {"ok": False, "error": "No camera keys found"}

    total_eps = info.get("total_episodes", 0)
    if total_eps == 0:
        return {"ok": False, "error": "No episodes"}

    # Read episode metadata to find video files
    ep_meta_list = []
    ep_dir = src_dir / "meta" / "episodes"
    if ep_dir.exists():
        for pf in sorted(ep_dir.rglob("*.parquet")):
            try:
                tbl = pq.read_table(str(pf))
            except Exception:
                continue
            d = tbl.to_pydict()
            for i in range(len(d.get("episode_index", []))):
                ep_meta_list.append({k: v[i] for k, v in d.items()})
    ep_meta_list.sort(key=lambda r: r.get("episode_index", 0))

    # Pick random episodes
    indices = random.sample(range(len(ep_meta_list)), min(num_previews, len(ep_meta_list)))

    cam_opts = techniques.get("camera", {})
    light_opts = techniques.get("light", {})
    mode = body.get("mode", "random")
    fixed_params = body.get("fixed_params") if mode == "fix" else None
    cam_fixed = fixed_params.get("camera") if fixed_params else None
    light_fixed = fixed_params.get("light") if fixed_params else None
    first_cam = cam_keys[0]

    previews = []
    for idx in indices:
        em = ep_meta_list[idx]
        ck = f"videos/{first_cam}"
        chunk = em.get(f"{ck}/chunk_index", 0)
        fi = em.get(f"{ck}/file_index", 0)
        from_ts = em.get(f"{ck}/from_timestamp", 0.0)
        vid_path = src_dir / "videos" / first_cam / f"chunk-{chunk:03d}" / f"file-{fi:03d}.mp4"
        if not vid_path.exists():
            continue

        cap = cv2.VideoCapture(str(vid_path))
        if from_ts > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, from_ts * 1000)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            continue

        # Resize for preview (max 160px wide)
        h, w = frame.shape[:2]
        scale = min(160 / w, 120 / h)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Encode original
        _, buf_orig = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_orig = base64.b64encode(buf_orig).decode()

        # Apply augmentation
        cam_p = _augp_camera(cam_opts, intensity, cam_fixed) if cam_opts.get("enabled") else None
        light_p = _augp_light(light_opts, intensity, light_fixed) if light_opts.get("enabled") else None
        aug_frame = _aug_frame(small, cam_p, light_p)

        _, buf_aug = cv2.imencode(".jpg", aug_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_aug = base64.b64encode(buf_aug).decode()

        previews.append({
            "ep": em.get("episode_index", idx),
            "original": b64_orig,
            "augmented": b64_aug,
        })

    return {"ok": True, "previews": previews}


@app.post("/api/datasets/{name}/augment")
async def start_augment(name: str, request: Request):
    if augment_state["running"]:
        return {"ok": False, "error": "Augmentation already running"}
    body = await request.json()
    new_name = body.get("name", "").strip().replace(" ", "-").replace("/", "-")
    target_episodes = body.get("target_episodes")
    intensity = body.get("intensity", 3)
    techniques = body.get("techniques", {})
    if not new_name:
        return {"ok": False, "error": "Enter a dataset name"}
    src_dir = ROOT / "data" / name
    if not (src_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Source dataset not found"}
    if (ROOT / "data" / new_name).exists():
        return {"ok": False, "error": f"Dataset '{new_name}' already exists"}
    if not target_episodes or target_episodes < 1:
        return {"ok": False, "error": "Invalid target episodes"}
    if intensity not in (1, 2, 3, 4, 5):
        intensity = 3
    mode = body.get("mode", "random")
    fixed_params = body.get("fixed_params") if mode == "fix" else None
    threading.Thread(
        target=_run_augmentation,
        args=(name, new_name, target_episodes, techniques, intensity, mode, fixed_params),
        daemon=True,
    ).start()
    return {"ok": True, "message": "Augmentation started"}


@app.get("/api/augment/status")
async def augment_status():
    return {
        "running": augment_state["running"],
        "progress": augment_state["progress"],
        "status": augment_state["status"],
        "error": augment_state["error"],
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print(f"LeRobot Dataset Manager")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  URL:            http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
