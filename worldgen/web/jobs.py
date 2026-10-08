"""Generation runs, one at a time, with a log of their stages that a browser can follow.

A run is minutes of CPU on a large map, so it happens on a worker thread and the request
that started it returns at once with an id. Every stage boundary the pipeline reports is
appended to the job's event list; a listener reads the list from wherever it last got to
and waits on the condition for more, so a browser that connects late still sees the
stages it missed.

Runs are serialised on one worker: two at once would each go at half speed and neither
would finish sooner, and the numpy work does not share well between threads anyway.
"""

import itertools
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.config import WorldConfig
from ..core.pipeline import GeneratorPipeline
from ..core.world_state import WorldState
from ..export.heightmap_import import HeightmapError
from ..naming.packs import CulturePackError
from ..stages import stages_for
from . import views


@dataclass
class Job:
    id: str
    seed: int
    config: WorldConfig
    events: list[dict[str, Any]] = field(default_factory=list)
    state: WorldState | None = None
    error: str | None = None
    finished: bool = False
    svgs: dict[str, str] = field(default_factory=dict)
    changed: threading.Condition = field(default_factory=threading.Condition)

    def emit(self, event: dict[str, Any]) -> None:
        with self.changed:
            self.events.append(event)
            self.changed.notify_all()

    def wait(self, seen: int, timeout: float) -> list[dict[str, Any]]:
        """Events after the first *seen*, waiting up to *timeout* seconds for one."""
        with self.changed:
            if len(self.events) <= seen and not self.finished:
                self.changed.wait(timeout)
            return self.events[seen:]

    def summary(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "seed": self.seed,
            "finished": self.finished,
            "error": self.error,
            "config": asdict(self.config),
        }
        if self.state is not None:
            out["views"] = views.views_for(self.config.model)
            out["settlements"] = len(self.state.settlements)
            out["rivers"] = len(self.state.rivers)
        return out


class JobStore:
    """Every job still held, newest last, dropping the oldest finished world past *keep*.

    A 200x200 world is tens of megabytes of Python objects, so the store cannot grow for
    as long as the server runs.
    """

    def __init__(self, keep: int = 4):
        self.keep = keep
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="worldgen")

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, seed: int, config: WorldConfig) -> Job:
        with self._lock:
            job = Job(id=f"{next(self._ids)}-{seed}", seed=seed, config=config)
            self._jobs[job.id] = job
            while len(self._jobs) > self.keep:
                oldest = next(iter(self._jobs.values()))
                if not oldest.finished:
                    break
                self._jobs.popitem(last=False)
        job.emit({"type": "queued"})
        self._worker.submit(run, job)
        return job

    def shutdown(self) -> None:
        self._worker.shutdown(wait=False, cancel_futures=True)


def run(job: Job) -> None:
    """Generate *job*'s world, reporting each stage as it starts and ends."""

    def on_stage(index: int, total: int, name: str, elapsed: float | None) -> None:
        job.emit(
            {
                "type": "stage",
                "index": index,
                "total": total,
                "name": name.removesuffix("Stage"),
                "elapsed": elapsed,
            }
        )

    started = time.perf_counter()
    try:
        pipeline = GeneratorPipeline(job.seed, job.config)
        for stage in stages_for(job.config, job.config.model):
            pipeline.add_stage(stage)
        job.state = pipeline.run(on_stage=on_stage)
    except (HeightmapError, CulturePackError) as exc:
        # The user-input failures, as the CLI draws the line: a bad image or culture pack.
        job.error = str(exc)
    except Exception as exc:
        # Anything else is a bug. It must reach the browser with its traceback, not vanish
        # into a worker thread.
        job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"
    with job.changed:
        job.finished = True
        if job.error is None:
            job.events.append({"type": "done", "elapsed": time.perf_counter() - started})
        else:
            job.events.append({"type": "error", "message": job.error})
        job.changed.notify_all()
