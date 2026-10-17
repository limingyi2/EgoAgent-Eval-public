"""Crash-safe, cross-process request budget and experiment result ledger.

SQLite's Windows file locks and BEGIN IMMEDIATE transactions serialize request
reservations. A successful reservation is permanent budget consumption, even
when the process later crashes or cannot determine whether bytes were sent.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
import uuid


GLOBAL_LIMIT = 12
# Match the maximum demand in the paired plan: three Uniform-8 runs at one
# request each, plus three V4 runs at three requests each.
STRATEGY_LIMITS = {"uniform8": 3, "v4_staged": 9}
RUN_STATUSES = {"success", "failed", "incomplete", "unknown"}
ANSWER_STATUSES = {"correct", "incorrect", "refused", "protocol_error", "unscorable", "unknown"}
EVIDENCE_STATUSES = {"sufficient", "insufficient", "manual_review_required", "unknown"}
MANUAL_REVIEW_STATUSES = {"pending", "completed", "unknown"}
RESULT_COLUMNS = (
    "experiment_id", "snapshot_id", "strategy", "task_id", "run_id", "video_sha256",
    "model_id", "run_status", "request_attempts", "http_statuses", "answer_status",
    "answer_correct", "evidence_status", "timestamp_coverage", "input_tokens",
    "output_tokens", "total_tokens", "cost_usd", "duration_seconds", "failure_reason",
    "manual_review_status", "evidence_reviewer", "evidence_review_independent",
    "evidence_review_blinded",
)


class LedgerError(RuntimeError):
    """Ledger is missing, corrupt, inconsistent, or could not be persisted."""


class BudgetExceeded(LedgerError):
    """A run, strategy, or experiment request budget is exhausted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ-") + uuid.uuid4().hex[:12]


class ExperimentLedger:
    """One immutable experiment identity, a request ledger, and CSV results."""

    def __init__(self, path: str | Path, *, experiment_id: str, snapshot_id: str,
                 _registry_path: str | Path | None = None):
        self.path = Path(path).resolve()
        self.csv_path = self.path.with_name("results.csv")
        self.experiment_id = experiment_id
        self.snapshot_id = snapshot_id
        if not self.path.is_file():
            raise LedgerError(f"Budget ledger does not exist; initialize it explicitly: {self.path}")
        try:
            with self._connection() as db:
                self._validate_database(db)
                metadata = dict(db.execute("SELECT key, value FROM metadata"))
                expected = {"experiment_id": experiment_id, "snapshot_id": snapshot_id,
                            "global_limit": str(GLOBAL_LIMIT),
                            "uniform8_limit": str(STRATEGY_LIMITS["uniform8"]),
                            "v4_staged_limit": str(STRATEGY_LIMITS["v4_staged"])}
                if any(metadata.get(k) != v for k, v in expected.items()):
                    raise LedgerError("Ledger identity or budget limits do not match this invocation")
                stored_registry_path = metadata.get("registry_path")
                if not stored_registry_path:
                    raise LedgerError("Ledger has no global experiment identity registration")
                self.registry_path = Path(stored_registry_path).resolve()
                if _registry_path is not None and Path(_registry_path).resolve() != self.registry_path:
                    raise LedgerError("Experiment registry path mismatch")
            self.validate_results_csv()
            self._validate_registry_entry()
        except LedgerError:
            raise
        except (OSError, sqlite3.Error, csv.Error, ValueError, TypeError) as exc:
            raise LedgerError(f"Cannot safely open experiment ledger: {exc}") from exc

    @classmethod
    def initialize(cls, path: str | Path, *, experiment_id: str, snapshot_id: str,
                   _registry_path: str | Path | None = None) -> "ExperimentLedger":
        target = Path(path).resolve()
        registry_target = Path(_registry_path).resolve() if _registry_path is not None else cls.default_registry_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        csv_target = target.with_name("results.csv")
        if target.exists():
            raise LedgerError(f"Refusing to overwrite existing budget ledger: {target}")
        if csv_target.exists():
            raise LedgerError(f"Refusing to reuse an existing results table: {csv_target}")
        if any(target.parent.iterdir()):
            raise LedgerError(f"Experiment directory must be new or empty: {target.parent}")
        # O_EXCL is the experiment-creation mutex. Never truncate or reset an
        # existing ledger, even if it is empty or initialization was interrupted.
        try:
            fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError as exc:
            raise LedgerError(f"Refusing to overwrite existing budget ledger: {target}") from exc
        os.close(fd)
        try:
            db = sqlite3.connect(target, timeout=10, isolation_level=None)
            try:
                db.execute("PRAGMA journal_mode=DELETE")
                db.execute("PRAGMA synchronous=FULL")
                db.execute("PRAGMA foreign_keys=ON")
                db.executescript("""
                    CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE runs (
                        run_id TEXT PRIMARY KEY,
                        strategy TEXT NOT NULL CHECK(strategy IN ('uniform8','v4_staged')),
                        task_id TEXT NOT NULL,
                        video_sha256 TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        run_status TEXT NOT NULL CHECK(run_status IN ('success','failed','incomplete','unknown')),
                        created_at TEXT NOT NULL,
                        UNIQUE(strategy, task_id)
                    );
                    CREATE TABLE attempts (
                        global_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_id TEXT NOT NULL,
                        strategy TEXT NOT NULL CHECK(strategy IN ('uniform8','v4_staged')),
                        task_id TEXT NOT NULL,
                        run_id TEXT NOT NULL REFERENCES runs(run_id),
                        request_seq INTEGER NOT NULL,
                        reservation_status TEXT NOT NULL CHECK(reservation_status='reserved'),
                        send_status TEXT NOT NULL CHECK(send_status IN ('not_sent','attempting','response_received','unknown')),
                        http_status INTEGER,
                        error_type TEXT,
                        reserved_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(run_id, request_seq)
                    );
                    CREATE TABLE results (
                        run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                        experiment_id TEXT NOT NULL,
                        snapshot_id TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        video_sha256 TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        run_status TEXT NOT NULL CHECK(run_status IN ('success','failed','incomplete','unknown')),
                        request_attempts INTEGER NOT NULL DEFAULT 0,
                        http_statuses TEXT NOT NULL DEFAULT '[]',
                        answer_status TEXT NOT NULL CHECK(answer_status IN ('correct','incorrect','refused','protocol_error','unscorable','unknown')),
                        answer_correct TEXT NOT NULL DEFAULT '',
                        evidence_status TEXT NOT NULL CHECK(evidence_status IN ('sufficient','insufficient','manual_review_required','unknown')),
                        timestamp_coverage TEXT NOT NULL DEFAULT '',
                        input_tokens TEXT NOT NULL DEFAULT '',
                        output_tokens TEXT NOT NULL DEFAULT '',
                        total_tokens TEXT NOT NULL DEFAULT '',
                        cost_usd TEXT NOT NULL DEFAULT '',
                        duration_seconds TEXT NOT NULL DEFAULT '',
                        failure_reason TEXT NOT NULL DEFAULT '',
                        manual_review_status TEXT NOT NULL CHECK(manual_review_status IN ('pending','completed','unknown')),
                        evidence_reviewer TEXT NOT NULL DEFAULT '',
                        evidence_review_independent TEXT NOT NULL DEFAULT '',
                        evidence_review_blinded TEXT NOT NULL DEFAULT ''
                    );
                """)
                metadata = {
                    "experiment_id": experiment_id, "snapshot_id": snapshot_id,
                    "global_limit": str(GLOBAL_LIMIT),
                    "uniform8_limit": str(STRATEGY_LIMITS["uniform8"]),
                    "v4_staged_limit": str(STRATEGY_LIMITS["v4_staged"]),
                    "created_at": utc_now(), "ledger_schema": "1",
                    "registry_path": str(registry_target),
                }
                db.executemany("INSERT INTO metadata(key,value) VALUES(?,?)", metadata.items())
                db.execute("PRAGMA user_version=1")
                db.execute("PRAGMA optimize")
            finally:
                db.close()
        except Exception as exc:
            # Leave the exclusive-create file in place. A partial ledger is
            # intentionally unrecoverable by re-initialization; no silent reset.
            raise LedgerError(f"Ledger initialization failed; file preserved and no request is safe: {exc}") from exc
        try:
            cls._register_global_identity(registry_target, experiment_id, snapshot_id, target)
        except Exception as exc:
            # The local ledger remains as an orphan and is never reset. Its
            # missing global registration makes all later opens fail closed.
            raise LedgerError(f"Could not reserve experiment ID globally: {exc}") from exc
        ledger = cls(target, experiment_id=experiment_id, snapshot_id=snapshot_id,
                     _registry_path=registry_target)
        ledger._export_results()
        return ledger

    @staticmethod
    def default_registry_path() -> Path:
        local_app_data = Path(os.environ.get("LOCALAPPDATA") or
                              (Path.home() / "AppData" / "Local"))
        return (local_app_data / "EgoAgent-Eval" / "experiment_registry.sqlite3").resolve()

    @staticmethod
    def _register_global_identity(path: Path, experiment_id: str, snapshot_id: str,
                                  ledger_path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            db = sqlite3.connect(path, timeout=10, isolation_level=None)
            try:
                db.execute("PRAGMA busy_timeout=10000")
                db.execute("PRAGMA synchronous=FULL")
                db.execute("BEGIN IMMEDIATE")
                db.execute("CREATE TABLE IF NOT EXISTS experiments ("
                           "experiment_id TEXT PRIMARY KEY,snapshot_id TEXT NOT NULL,"
                           "ledger_path TEXT NOT NULL,created_at TEXT NOT NULL)")
                db.execute("INSERT INTO experiments(experiment_id,snapshot_id,ledger_path,created_at) "
                           "VALUES(?,?,?,?)", (experiment_id, snapshot_id, str(ledger_path), utc_now()))
                db.commit()
            finally:
                db.close()
        except sqlite3.IntegrityError as exc:
            raise LedgerError(f"Experiment ID already exists in the machine registry: {experiment_id}") from exc
        except sqlite3.Error as exc:
            raise LedgerError(f"Experiment registry is unavailable: {exc}") from exc

    def _validate_registry_entry(self) -> None:
        db = None
        try:
            db = sqlite3.connect(self.registry_path, timeout=10)
            db.execute("PRAGMA busy_timeout=10000")
            row = db.execute("SELECT snapshot_id,ledger_path FROM experiments WHERE experiment_id=?",
                             (self.experiment_id,)).fetchone()
            if not row or row[0] != self.snapshot_id or Path(row[1]).resolve() != self.path:
                raise LedgerError("Global experiment registry does not match this ledger")
        except LedgerError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise LedgerError(f"Cannot validate global experiment identity: {exc}") from exc
        finally:
            if db is not None:
                db.close()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @contextmanager
    def _connection(self):
        db = self._connect()
        try:
            yield db
        finally:
            db.close()

    def _validate_database(self, db: sqlite3.Connection) -> None:
        check = db.execute("PRAGMA quick_check").fetchone()
        if not check or check[0] != "ok":
            raise LedgerError("SQLite quick_check failed")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version != 1:
            raise LedgerError(f"Unsupported or missing ledger schema version: {version}")

    def validate_results_csv(self) -> None:
        if not self.csv_path.exists():
            return
        try:
            with self.csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != RESULT_COLUMNS:
                    raise LedgerError("results.csv has an invalid or changed column header")
                seen: set[str] = set()
                for row in reader:
                    run_id = row.get("run_id", "")
                    if not run_id or run_id in seen:
                        raise LedgerError("results.csv contains a missing or duplicate run_id")
                    seen.add(run_id)
                    if (row.get("run_status") not in RUN_STATUSES
                            or row.get("answer_status") not in ANSWER_STATUSES
                            or row.get("evidence_status") not in EVIDENCE_STATUSES
                            or row.get("manual_review_status") not in MANUAL_REVIEW_STATUSES):
                        raise LedgerError("results.csv contains an invalid status value")
                    if row.get("evidence_status") == "sufficient":
                        if (row.get("evidence_review_independent") != "true"
                                or row.get("evidence_review_blinded") != "true"
                                or not row.get("evidence_reviewer")):
                            raise LedgerError("Evidence cannot be sufficient without independent blinded review")
        except LedgerError:
            raise
        except (OSError, csv.Error, UnicodeError) as exc:
            raise LedgerError(f"Cannot validate results.csv: {exc}") from exc

    def _export_results(self) -> None:
        """Atomically materialize the canonical SQLite result rows as CSV."""
        with self._connection() as db:
            db.execute("BEGIN")
            rows = db.execute("SELECT " + ",".join(RESULT_COLUMNS) + " FROM results ORDER BY rowid").fetchall()
            db.commit()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=self.csv_path.parent,
                                             prefix=self.csv_path.name + ".", suffix=".tmp", delete=False) as stream:
                temp_name = stream.name
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(RESULT_COLUMNS)
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.csv_path)
            temp_name = None
        except OSError as exc:
            raise LedgerError(f"Could not atomically write results CSV: {exc}") from exc
        finally:
            if temp_name:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass

    def _validate_identity(self, db: sqlite3.Connection) -> None:
        self._validate_database(db)
        metadata = dict(db.execute("SELECT key,value FROM metadata"))
        if metadata.get("experiment_id") != self.experiment_id or metadata.get("snapshot_id") != self.snapshot_id:
            raise LedgerError("Experiment or code snapshot identity mismatch")

    def register_run(self, *, strategy: str, task_id: str, run_id: str,
                     video_sha256: str, model_id: str) -> None:
        if strategy not in STRATEGY_LIMITS:
            raise LedgerError(f"Unsupported strategy group: {strategy}")
        if not all((task_id, run_id, video_sha256, model_id)):
            raise LedgerError("Run identity fields must be nonempty")
        try:
            with self._connection() as db:
                db.execute("BEGIN IMMEDIATE")
                self._validate_identity(db)
                db.execute("INSERT INTO runs(run_id,strategy,task_id,video_sha256,model_id,run_status,created_at) "
                           "VALUES(?,?,?,?,?,'unknown',?)",
                           (run_id, strategy, task_id, video_sha256, model_id, utc_now()))
                db.execute("INSERT INTO results(run_id,experiment_id,snapshot_id,strategy,task_id,video_sha256,model_id,"
                           "run_status,answer_status,evidence_status,manual_review_status) "
                           "VALUES(?,?,?,?,?,?,?,'unknown','unknown','unknown','pending')",
                           (run_id, self.experiment_id, self.snapshot_id, strategy, task_id, video_sha256, model_id))
                db.commit()
            self._export_results()
        except sqlite3.IntegrityError as exc:
            raise LedgerError(f"Duplicate run or strategy/task execution refused: {exc}") from exc
        except (OSError, sqlite3.Error) as exc:
            raise LedgerError(f"Could not persist run registration; refusing requests: {exc}") from exc

    def reserve_request(self, *, run_id: str) -> int:
        """Durably consume one slot before any network request can be sent."""
        try:
            with self._connection() as db:
                db.execute("BEGIN IMMEDIATE")
                self._validate_identity(db)
                run = db.execute("SELECT strategy,task_id,run_status FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if not run:
                    raise LedgerError("Request run_id is not registered in this experiment")
                strategy, task_id, run_status = run
                if run_status != "unknown":
                    raise LedgerError("Completed or failed runs cannot submit more requests")
                total = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
                group_count = db.execute("SELECT COUNT(*) FROM attempts WHERE strategy=?", (strategy,)).fetchone()[0]
                if total >= GLOBAL_LIMIT:
                    raise BudgetExceeded("Global real-request attempt limit reached")
                if group_count >= STRATEGY_LIMITS[strategy]:
                    raise BudgetExceeded(f"{strategy} real-request attempt limit reached")
                seq = db.execute("SELECT COALESCE(MAX(request_seq),0)+1 FROM attempts WHERE run_id=?", (run_id,)).fetchone()[0]
                now = utc_now()
                cur = db.execute("INSERT INTO attempts(experiment_id,strategy,task_id,run_id,request_seq,"
                                 "reservation_status,send_status,reserved_at,updated_at) "
                                 "VALUES(?,?,?,?,?,'reserved','not_sent',?,?)",
                                 (self.experiment_id, strategy, task_id, run_id, seq, now, now))
                attempt_id = cur.lastrowid
                db.commit()
                return int(attempt_id)
        except LedgerError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise LedgerError(f"Request reservation failed closed; no request sent: {exc}") from exc

    def mark_attempting(self, attempt_id: int) -> None:
        self._update_attempt(attempt_id, send_status="attempting")

    def finish_attempt(self, attempt_id: int, *, http_status: int | None, error_type: str | None) -> None:
        self._update_attempt(attempt_id, send_status="response_received", http_status=http_status,
                             error_type=error_type)

    def mark_unknown(self, attempt_id: int, *, error_type: str = "result_unknown") -> None:
        self._update_attempt(attempt_id, send_status="unknown", error_type=error_type)

    def _update_attempt(self, attempt_id: int, *, send_status: str,
                        http_status: int | None = None, error_type: str | None = None) -> None:
        if send_status not in {"attempting", "response_received", "unknown"}:
            raise LedgerError("Invalid request send state")
        try:
            with self._connection() as db:
                db.execute("BEGIN IMMEDIATE")
                self._validate_identity(db)
                cur = db.execute("UPDATE attempts SET send_status=?,http_status=?,error_type=?,updated_at=? "
                                 "WHERE global_seq=? AND send_status IN ('not_sent','attempting')",
                                 (send_status, http_status, error_type, utc_now(), attempt_id))
                if cur.rowcount != 1:
                    raise LedgerError("Request ledger attempt missing or already finalized")
                db.commit()
        except LedgerError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise LedgerError(f"Request status could not be persisted: {exc}") from exc

    def complete_run(self, run_id: str, result: Any) -> None:
        """Persist parsed run outcomes; correctness stays unscored pending review."""
        attempt_rows = self.attempts_for_run(run_id)
        attempts = len(attempt_rows)
        statuses = [row[0] for row in attempt_rows if row[0] is not None]
        status = getattr(result, "status", "failed")
        reason = getattr(result, "termination_reason", None)
        errors = [getattr(error, "code", "unknown") for error in getattr(result, "errors", [])]
        failure_reason = ";".join(dict.fromkeys(([reason] if reason else []) + errors))
        if status in {"ok", "refused"}:
            run_status = "success"
        elif reason == "budget_exhausted":
            run_status = "incomplete"
        else:
            run_status = "failed"
        if status == "refused":
            answer_status = "refused"
        elif reason == "protocol_error" or "protocol_error" in errors or "invalid_final_answer" in errors:
            answer_status = "protocol_error"
        elif status == "ok":
            answer_status = "unscorable"
        else:
            answer_status = "unknown"
        if reason == "insufficient_evidence" or "insufficient_evidence" in errors:
            evidence_status = "insufficient"
        elif getattr(result, "observed_frames", []):
            evidence_status = "manual_review_required"
        else:
            evidence_status = "unknown"
        usage = getattr(result, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        total_tokens = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
        def cell(value: Any) -> str:
            return "" if value is None else str(value)
        values = {
            "run_status": run_status, "request_attempts": attempts,
            "http_statuses": json.dumps(statuses, separators=(",", ":")),
            "answer_status": answer_status, "answer_correct": "",
            "evidence_status": evidence_status,
            "input_tokens": cell(input_tokens), "output_tokens": cell(output_tokens),
            "total_tokens": cell(total_tokens),
            "cost_usd": cell(getattr(usage, "cost_usd", None)) if usage else "",
            "duration_seconds": cell(getattr(result, "latency_seconds", None)),
            "failure_reason": failure_reason,
            "manual_review_status": "pending" if evidence_status == "manual_review_required" else "unknown",
        }
        self._update_result(run_id, values)

    def attempts_for_run(self, run_id: str) -> list[tuple[int | None, str, str | None]]:
        try:
            with self._connection() as db:
                self._validate_identity(db)
                return db.execute("SELECT http_status,send_status,error_type FROM attempts WHERE run_id=? ORDER BY request_seq",
                                  (run_id,)).fetchall()
        except (OSError, sqlite3.Error) as exc:
            raise LedgerError(f"Could not read attempts for result: {exc}") from exc

    def _update_result(self, run_id: str, values: dict[str, Any]) -> None:
        allowed = {"run_status", "request_attempts", "http_statuses", "answer_status", "answer_correct",
                   "evidence_status", "timestamp_coverage", "input_tokens", "output_tokens", "total_tokens",
                   "cost_usd", "duration_seconds", "failure_reason", "manual_review_status", "evidence_reviewer",
                   "evidence_review_independent", "evidence_review_blinded"}
        if set(values) - allowed:
            raise LedgerError("Unknown result fields")
        if values.get("run_status", "unknown") not in RUN_STATUSES:
            raise LedgerError("Invalid run_status")
        if values.get("answer_status", "unknown") not in ANSWER_STATUSES:
            raise LedgerError("Invalid answer_status")
        if values.get("evidence_status", "unknown") not in EVIDENCE_STATUSES:
            raise LedgerError("Invalid evidence_status")
        if values.get("manual_review_status", "unknown") not in MANUAL_REVIEW_STATUSES:
            raise LedgerError("Invalid manual_review_status")
        answer_correct = values.get("answer_correct")
        if answer_correct not in (None, "", "true", "false"):
            raise LedgerError("answer_correct must be true, false, or unknown/empty")
        if values.get("answer_status") == "correct" and answer_correct != "true":
            raise LedgerError("correct answer_status requires answer_correct=true")
        if values.get("answer_status") == "incorrect" and answer_correct != "false":
            raise LedgerError("incorrect answer_status requires answer_correct=false")
        try:
            with self._connection() as db:
                db.execute("BEGIN IMMEDIATE")
                self._validate_identity(db)
                current = db.execute("SELECT evidence_reviewer,evidence_review_independent,evidence_review_blinded "
                                     "FROM results WHERE run_id=?", (run_id,)).fetchone()
                if not current:
                    raise LedgerError("Result run_id is not registered")
                reviewer = values.get("evidence_reviewer", current[0])
                independent = values.get("evidence_review_independent", current[1])
                blinded = values.get("evidence_review_blinded", current[2])
                if values.get("evidence_status") == "sufficient" and not (reviewer and independent == "true" and blinded == "true"):
                    raise LedgerError("evidence_status=sufficient requires an identified independent blinded reviewer")
                assigns = ",".join(f"{key}=?" for key in values)
                db.execute(f"UPDATE results SET {assigns} WHERE run_id=?", (*values.values(), run_id))
                if "run_status" in values:
                    db.execute("UPDATE runs SET run_status=? WHERE run_id=?", (values["run_status"], run_id))
                db.commit()
            self._export_results()
        except LedgerError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise LedgerError(f"Could not persist result; model requests will not be repeated: {exc}") from exc

    def record_answer_review(self, run_id: str, *, answer_status: str, answer_correct: bool | None) -> None:
        if answer_status not in ANSWER_STATUSES:
            raise LedgerError("Invalid answer_status")
        if answer_status == "correct" and answer_correct is not True:
            raise LedgerError("correct answer_status requires answer_correct=true")
        if answer_status == "incorrect" and answer_correct is not False:
            raise LedgerError("incorrect answer_status requires answer_correct=false")
        if answer_status in {"correct", "incorrect"}:
            self._update_result(run_id, {"answer_status": answer_status,
                                         "answer_correct": str(answer_correct).lower()})
        else:
            self._update_result(run_id, {"answer_status": answer_status, "answer_correct": ""})

    def record_evidence_review(self, run_id: str, *, evidence_status: str,
                               reviewer: str, independent: bool, blinded: bool) -> None:
        if evidence_status not in {"sufficient", "insufficient", "unknown", "manual_review_required"}:
            raise LedgerError("Invalid evidence_status")
        if evidence_status in {"sufficient", "insufficient"} and (
                not reviewer.strip() or not independent or not blinded):
            raise LedgerError("Evidence classification requires a named independent blinded reviewer")
        self._update_result(run_id, {
            "evidence_status": evidence_status,
            "evidence_reviewer": reviewer.strip() if reviewer else "",
            "evidence_review_independent": str(bool(independent)).lower(),
            "evidence_review_blinded": str(bool(blinded)).lower(),
            "manual_review_status": "completed" if reviewer.strip() else "pending",
        })

    def counts(self) -> dict[str, int]:
        try:
            with self._connection() as db:
                self._validate_identity(db)
                counts = {name: db.execute("SELECT COUNT(*) FROM attempts WHERE strategy=?", (name,)).fetchone()[0]
                          for name in STRATEGY_LIMITS}
                counts["global"] = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
                return counts
        except (OSError, sqlite3.Error) as exc:
            raise LedgerError(f"Could not read budget counts: {exc}") from exc


def initialize_experiment(path: str | Path, *, experiment_id: str, snapshot_id: str,
                          _registry_path: str | Path | None = None) -> ExperimentLedger:
    """Explicit one-time experiment creation used by the setup CLI."""
    return ExperimentLedger.initialize(path, experiment_id=experiment_id, snapshot_id=snapshot_id,
                                       _registry_path=_registry_path)
