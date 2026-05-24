"""
Shared Code Executor Module
============================
Handles code execution for Python, R, and STATA.

This module is the single source of truth for code execution,
used by both ResearchReproducerAgent and AgenticReplicationEngineV2.
"""

import io
import os
import logging
import re
import subprocess
import contextlib
import tempfile
import threading
import traceback
import textwrap
import sys
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from core.constants import (
    CODE_EXECUTION_TIMEOUT_SECONDS,
    R_EXECUTION_TIMEOUT_SECONDS,
    STATA_EXECUTION_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class CodeLanguage(str, Enum):
    """Supported programming languages for code execution."""
    PYTHON = "python"
    R = "r"
    STATA = "stata"


@dataclass
class ExecutionResult:
    """Result of a code execution operation."""
    success: bool
    output: str
    error: Optional[str] = None
    figures: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    traceback_str: Optional[str] = None


class PersistentRSession:
    """Persistent R session using subprocess with stdin/stdout pipes.

    Keeps a single R process alive across multiple code executions,
    avoiding the overhead of reloading packages and data on each call.
    Communication uses sentinel markers to delimit output boundaries.
    """

    START_SENTINEL = "___EXEC_START___"
    END_SENTINEL = "___EXEC_END___"
    ERROR_SENTINEL = "___EXEC_ERROR___"

    def __init__(self, working_dir: str) -> None:
        self.working_dir = working_dir
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _start(self) -> None:
        """Start (or restart) the R process."""
        if self._process is not None:
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except Exception:
                pass

        self._process = subprocess.Popen(
            ["R", "--vanilla", "--quiet", "--no-save"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=self.working_dir,
            bufsize=1,
        )

        # Load common packages once at startup
        setup_code = f'''\
setwd("{self.working_dir}")
suppressPackageStartupMessages({{
  if(require(estimatr)) library(estimatr)
  if(require(tidyverse)) library(tidyverse)
  if(require(haven)) library(haven)
  if(require(lmtest)) library(lmtest)
  if(require(sandwich)) library(sandwich)
}})
cat("{self.START_SENTINEL}\\n")
cat("{self.END_SENTINEL}\\n")
'''
        self._process.stdin.write(setup_code)
        self._process.stdin.flush()
        # Consume the startup output
        self._read_until_sentinel()
        logger.info("Persistent R session started (pid=%s)", self._process.pid)

    def _is_alive(self) -> bool:
        """Check if the R process is still running."""
        return self._process is not None and self._process.poll() is None

    def _read_until_sentinel(self) -> tuple:
        """Read stdout until END or ERROR sentinel is found.

        Returns:
            (output_text, had_error) tuple
        """
        lines = []
        had_error = False
        capturing = False

        while True:
            line = self._process.stdout.readline()
            if not line:
                # Process died
                break
            stripped = line.rstrip("\n").rstrip("\r")

            if stripped == self.START_SENTINEL:
                capturing = True
                continue
            if stripped == self.END_SENTINEL:
                break
            if stripped == self.ERROR_SENTINEL:
                had_error = True
                break

            if capturing:
                lines.append(line.rstrip("\n"))

        return "\n".join(lines), had_error

    def execute(self, code: str, timeout: int = R_EXECUTION_TIMEOUT_SECONDS) -> ExecutionResult:
        """Execute R code in the persistent session.

        Args:
            code: R code to execute.
            timeout: Timeout in seconds.

        Returns:
            ExecutionResult with output and error info.
        """
        with self._lock:
            if not self._is_alive():
                self._start()

            # Wrap user code in tryCatch with sentinels
            wrapped = f'''\
cat("{self.START_SENTINEL}\\n")
tryCatch({{
{code}
  cat("\\n{self.END_SENTINEL}\\n")
}}, error = function(e) {{
  cat(paste0("\\nR Error: ", conditionMessage(e), "\\n"))
  cat("{self.ERROR_SENTINEL}\\n")
}})
'''
            try:
                self._process.stdin.write(wrapped)
                self._process.stdin.flush()
            except (BrokenPipeError, OSError):
                # Process died, restart and retry once
                self._start()
                self._process.stdin.write(wrapped)
                self._process.stdin.flush()

            # Read output with timeout
            result_container = [None]

            def _reader():
                result_container[0] = self._read_until_sentinel()

            reader_thread = threading.Thread(target=_reader, daemon=True)
            reader_thread.start()
            reader_thread.join(timeout=timeout)

            if reader_thread.is_alive():
                # Timed out — kill and restart
                self._process.kill()
                self._process = None
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"R execution timed out ({timeout}s limit)",
                )

            if result_container[0] is None:
                # Process died during execution
                self._process = None
                return ExecutionResult(
                    success=False,
                    output="",
                    error="R process died during execution",
                )

            output, had_error = result_container[0]

            if had_error:
                return ExecutionResult(
                    success=False,
                    output=output,
                    error=output.strip().split("\n")[-1] if output.strip() else "Unknown R error",
                )

            return ExecutionResult(success=True, output=output)

    def shutdown(self) -> None:
        """Shut down the persistent R session."""
        if self._process is not None:
            try:
                self._process.stdin.write("q(save='no')\n")
                self._process.stdin.flush()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            finally:
                self._process = None
            logger.info("Persistent R session shut down")


class CodeExecutor:
    """Handles code execution for Python, R, and STATA.

    Manages working directories, runtime detection, and
    language-specific execution with output capture.

    Args:
        working_dir: Base directory for execution. Created if absent.
    """

    def __init__(
        self,
        working_dir: Optional[str] = None,
        figures_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        source_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.working_dir = os.path.abspath(
            working_dir or tempfile.mkdtemp(prefix="research_reproducer_")
        )
        self.figures_dir = os.path.abspath(
            figures_dir or os.path.join(self.working_dir, "figures")
        )
        self.data_dir = os.path.abspath(
            data_dir or os.path.join(self.working_dir, "data")
        )
        self.source_dir = os.path.abspath(source_dir or self.data_dir)
        self.output_dir = os.path.abspath(output_dir or self.working_dir)
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        self.stata_batch_command = self._find_stata_batch_command()
        self.runtime_bin_dir = self._prepare_runtime_bin_dir()
        self.runtimes = self._check_runtimes()
        self._persistent_r: Optional[PersistentRSession] = None
        logger.info("Working directory: %s", self.working_dir)
        logger.info("Data directory: %s", self.data_dir)
        logger.info("Source directory: %s", self.source_dir)
        logger.info("Output directory: %s", self.output_dir)
        logger.info("Available runtimes: %s", self.runtimes)

    # Known Stata installation paths (macOS)
    _STATA_PYSTATA_PATHS = [
        "/Applications/StataNow/utilities",
        "/Applications/Stata/utilities",
        "/Applications/StataMP/utilities",
        "/Applications/StataSE/utilities",
        "/Applications/StataBE/utilities",
    ]
    _STATA_BATCH_COMMAND_CANDIDATES = [
        "stata",
        "stata-mp",
        "stata-se",
        "StataMP",
        "StataSE",
        "StataNow",
        "/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se",
        "/Applications/StataNow/StataSE.app/Contents/MacOS/StataSE",
        "/Applications/StataMP/StataMP.app/Contents/MacOS/stata-mp",
        "/Applications/StataSE/StataSE.app/Contents/MacOS/stata-se",
        "/Applications/Stata/Stata.app/Contents/MacOS/stata",
    ]

    def _find_stata_batch_command(self) -> Optional[str]:
        """Locate a batch-capable Stata executable."""
        for candidate in self._STATA_BATCH_COMMAND_CANDIDATES:
            resolved = shutil.which(candidate) if os.sep not in candidate else candidate
            if resolved and os.path.isfile(resolved) and os.access(resolved, os.X_OK):
                return resolved
        return None

    def _prepare_runtime_bin_dir(self) -> str:
        """Expose discovered runtimes under common executable names for probes."""
        runtime_bin_dir = os.path.join(self.working_dir, "runtime_bin")
        os.makedirs(runtime_bin_dir, exist_ok=True)
        if not self.stata_batch_command:
            return runtime_bin_dir
        for executable_name in ("stata", "stata-mp", "stata-se"):
            wrapper_path = os.path.join(runtime_bin_dir, executable_name)
            if os.path.exists(wrapper_path):
                continue
            with open(wrapper_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "#!/bin/sh\n"
                    f'exec "{self.stata_batch_command}" "$@"\n'
                )
            os.chmod(wrapper_path, 0o755)
        return runtime_bin_dir

    def _check_runtimes(self) -> Dict[str, bool]:
        """Detect which code execution runtimes are available."""
        runtimes: Dict[str, bool] = {"python": True, "r": False, "stata": False}

        try:
            result = subprocess.run(
                ["Rscript", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            runtimes["r"] = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try importing pystata; if not on sys.path, look in known Stata dirs
        try:
            import pystata  # noqa: F401
            runtimes["stata"] = True
        except ImportError:
            import sys as _sys
            for p in self._STATA_PYSTATA_PATHS:
                if os.path.isdir(os.path.join(p, "pystata")):
                    if p not in _sys.path:
                        _sys.path.insert(0, p)
                    try:
                        import pystata  # noqa: F401
                        runtimes["stata"] = True
                        logger.info("Found pystata at %s", p)
                        break
                    except ImportError:
                        pass

        if self.stata_batch_command:
            runtimes["stata"] = True

        return runtimes

    def execute(self, code: str, language: str) -> ExecutionResult:
        """Execute code in the specified language.

        Args:
            code: Source code to execute.
            language: One of 'python', 'r', or 'stata'.

        Returns:
            ExecutionResult with output, errors, and figures.
        """
        lang = language.lower()
        if lang == "python":
            return self.execute_python(code)
        elif lang == "r":
            return self.execute_r(code)
        elif lang in ("stata", "do"):
            return self.execute_stata(code)
        else:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Unsupported language: {language}. Use 'python', 'r', or 'stata'.",
            )

    def execute_python(self, code: str) -> ExecutionResult:
        """Execute Python code in a subprocess with hard timeout."""
        figures_before = (
            set(os.listdir(self.figures_dir)) if os.path.exists(self.figures_dir) else set()
        )
        script_path = os.path.join(
            self.working_dir,
            f"python_exec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.py",
        )
        bootstrap = f"""\
import os
import warnings

os.chdir(r"{self.working_dir}")
working_dir = r"{self.working_dir}"
figures_dir = r"{self.figures_dir}"
data_dir = r"{self.data_dir}"
source_dir = r"{self.source_dir}"
output_dir = r"{self.output_dir}"
runtime_bin_dir = r"{self.runtime_bin_dir}"
os.environ["WORKING_DIR"] = working_dir
os.environ["FIGURES_DIR"] = figures_dir
os.environ["DATA_DIR"] = data_dir
os.environ["SOURCE_DIR"] = source_dir
os.environ["OUTPUT_DIR"] = output_dir
if runtime_bin_dir and os.path.isdir(runtime_bin_dir):
    os.environ["PATH"] = runtime_bin_dir + os.pathsep + os.environ.get("PATH", "")
warnings.filterwarnings("ignore")
try:
    import numpy as np
except ImportError:
    np = None
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import statsmodels.api as sm
except ImportError:
    sm = None
try:
    from scipy import stats
except ImportError:
    stats = None
"""
        wrapped_script = (
            bootstrap
            + "\ntry:\n"
            + textwrap.indent(code, "    ")
            + "\nfinally:\n"
            + textwrap.indent(
                """
try:
    import matplotlib.pyplot as plt
    for fig_num in plt.get_fignums():
        fig_path = os.path.join(
            figures_dir,
            f"figure_{fig_num}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png",
        )
        plt.figure(fig_num).savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close("all")
except Exception:
    pass
""",
                "    ",
            )
        )
        try:
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("from datetime import datetime\n")
                handle.write(wrapped_script)

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=self.working_dir,
                env={
                    **os.environ,
                    "PATH": (
                        self.runtime_bin_dir
                        + os.pathsep
                        + os.environ.get("PATH", "")
                    ),
                },
                timeout=CODE_EXECUTION_TIMEOUT_SECONDS,
            )
            figures_after = (
                set(os.listdir(self.figures_dir))
                if os.path.exists(self.figures_dir)
                else set()
            )
            new_figures = [
                os.path.join(self.figures_dir, name)
                for name in sorted(figures_after - figures_before)
            ]

            if result.returncode == 0:
                return ExecutionResult(
                    success=True,
                    output=result.stdout,
                    error=result.stderr or None,
                    figures=new_figures,
                )

            return ExecutionResult(
                success=False,
                output=result.stdout,
                error=result.stderr.strip() or "Python execution failed",
                figures=new_figures,
                traceback_str=result.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                success=False,
                output=exc.stdout or "",
                error=f"Python execution timed out ({CODE_EXECUTION_TIMEOUT_SECONDS}s limit)",
                traceback_str=exc.stderr,
            )
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

    def execute_r(self, code: str) -> ExecutionResult:
        """Execute R code via persistent R session.

        Uses a long-running R process so that variables, loaded data,
        and packages persist across calls. Falls back to Rscript
        subprocess if the persistent session cannot be started.
        """
        if not self.runtimes.get("r"):
            return ExecutionResult(
                success=False, output="", error="R is not available on this system"
            )

        # Lazy-init persistent session on first call
        if self._persistent_r is None:
            self._persistent_r = PersistentRSession(self.working_dir)
            try:
                self._persistent_r.execute(
                    textwrap.dedent(
                        f"""
                        source_dir <- "{self.source_dir}"
                        output_dir <- "{self.output_dir}"
                        data_dir <- "{self.data_dir}"
                        setwd("{self.working_dir}")
                        """
                    )
                )
            except Exception:
                pass

        return self._persistent_r.execute(code)

    def shutdown(self) -> None:
        """Shut down persistent sessions."""
        if self._persistent_r is not None:
            self._persistent_r.shutdown()
            self._persistent_r = None

    _stata_initialized: bool = False

    def _ensure_stata_initialized(self) -> None:
        """Initialize Stata via pystata.config.init() once per process."""
        if CodeExecutor._stata_initialized:
            return
        from pystata import config as stata_config
        # Detect edition from the installation (prefer SE, fall back to MP/BE)
        for edition in ("se", "mp", "be"):
            try:
                stata_config.init(edition, splash=False)
                CodeExecutor._stata_initialized = True
                logger.info("Stata initialized (edition=%s)", edition)
                return
            except (FileNotFoundError, SystemError, ValueError):
                continue
        raise RuntimeError("Failed to initialize Stata — no valid edition found")

    def execute_stata(self, code: str) -> ExecutionResult:
        """Execute STATA code via pystata integration."""
        if not self.runtimes.get("stata"):
            return ExecutionResult(
                success=False,
                output="",
                error="STATA/pystata is not available. Install pystata and configure STATA path.",
            )

        if "pystata" not in sys.modules:
            return self.execute_stata_batch(code)

        result_container: Dict[str, ExecutionResult] = {}

        def _run_stata() -> None:
            try:
                self._ensure_stata_initialized()
                from pystata import stata
                import sys as _sys

                old_stdout = _sys.stdout
                _sys.stdout = captured_output = io.StringIO()
                try:
                    stata.run(f'cd "{self.working_dir}"', quietly=True)
                    preamble = (
                        f'global SOURCE_DIR "{self.source_dir}"\n'
                        f'global DATA_DIR "{self.data_dir}"\n'
                        f'global OUTPUT_DIR "{self.output_dir}"\n'
                    )
                    stata.run(preamble + code)
                finally:
                    _sys.stdout = old_stdout

                output = captured_output.getvalue()
                result_container["result"] = ExecutionResult(
                    success=True,
                    output=output or "STATA execution completed successfully.",
                )
            except Exception as exc:  # pragma: no cover - depends on Stata runtime
                result_container["result"] = ExecutionResult(
                    success=False,
                    output="",
                    error=f"STATA execution error: {exc}",
                    traceback_str=traceback.format_exc(),
                )

        worker = threading.Thread(target=_run_stata, daemon=True)
        worker.start()
        worker.join(timeout=STATA_EXECUTION_TIMEOUT_SECONDS)
        if worker.is_alive():
            return ExecutionResult(
                success=False,
                output="",
                error=f"STATA execution timed out ({STATA_EXECUTION_TIMEOUT_SECONDS}s limit)",
            )
        return result_container["result"]

    def execute_stata_batch(
        self,
        code: str,
        wrapper_path: Optional[str] = None,
        timeout: int = STATA_EXECUTION_TIMEOUT_SECONDS,
    ) -> ExecutionResult:
        """Execute STATA code in an isolated batch process.

        This is the default safe path for substantive STATA work because it
        avoids leaking `preserve`, logs, globals, or temp files across calls.
        """
        if not self.stata_batch_command:
            return ExecutionResult(
                success=False,
                output="",
                error="No batch-capable Stata executable was found.",
            )

        def _ensure_batch_exit(payload: str) -> str:
            normalized = payload.rstrip()
            if re.search(r"(?im)^\s*exit\s*,\s*clear\s+STATA\s*$", normalized[-2000:]):
                return normalized + "\n"
            return (
                normalized
                + "\n\n#delimit cr\n"
                + "capture log close _all\n"
                + "exit, clear STATA\n"
            )

        figures_before = self._collect_figure_paths()
        script_path = wrapper_path or os.path.join(
            self.working_dir,
            f"stata_exec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.do",
        )
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        if not (wrapper_path and os.path.exists(script_path)):
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write(_ensure_batch_exit(code))
        try:
            result = subprocess.run(
                [self.stata_batch_command, "-q", "do", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                success=False,
                output=exc.stdout or "",
                error=f"STATA execution timed out ({timeout}s limit)",
                traceback_str=exc.stderr,
            )
        except OSError as exc:
            return ExecutionResult(
                success=False,
                output="",
                error=f"STATA batch execution failed: {exc}",
                traceback_str=traceback.format_exc(),
            )

        figures_after = self._collect_figure_paths()
        new_figures = sorted(figures_after - figures_before)
        output = (result.stdout or "") + ((("\n" + result.stderr) if result.stderr else ""))
        stata_error_match = re.search(r"(?im)^\s*r\((\d+)\);?\s*$", output)
        success = result.returncode == 0 and stata_error_match is None
        if success:
            return ExecutionResult(
                success=True,
                output=output or "STATA batch execution completed successfully.",
                figures=new_figures,
            )
        return ExecutionResult(
            success=False,
            output=output,
            error=(output.strip() or "STATA batch execution failed"),
            figures=new_figures,
            traceback_str=result.stderr or result.stdout,
        )

    def _collect_figure_paths(self) -> set[str]:
        figure_paths: set[str] = set()
        for root in (self.figures_dir, self.output_dir):
            if not os.path.isdir(root):
                continue
            for base, _dirs, files in os.walk(root):
                for name in files:
                    if name.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg", ".eps", ".gph")):
                        figure_paths.add(os.path.join(base, name))
        return figure_paths
