"""Small desktop UI for the standalone V06 video-preparation runner."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import yaml


HERE = Path(__file__).resolve().parent


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FLUID-Parabola Stabilization")
        self.geometry("900x620")
        self.messages: queue.Queue[str | None] = queue.Queue()
        config = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
        self.video = tk.StringVar(value=str((HERE / config["input_video"]).resolve()))
        self.output = tk.StringVar(value=str((HERE / config["output_root"]).resolve()))
        self.status = tk.StringVar(value="Ready")
        self._build()
        self.after(100, self._poll)

    def _build(self) -> None:
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="FLUID-Parabola Stabilization", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Reproduce the retained N21 4+4+2 geometry and WR000001 correction.",
        ).pack(anchor="w", pady=(2, 16))
        self._path_row(frame, "Input camera-calibrated video", self.video, self._choose_video)
        self._path_row(frame, "Output directory", self.output, self._choose_output)
        self.run_button = ttk.Button(frame, text="Run V06 preparation", command=self._run)
        self.run_button.pack(anchor="w", pady=(10, 8))
        ttk.Label(frame, textvariable=self.status).pack(anchor="w")
        self.log = tk.Text(frame, height=24, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, pady=(8, 0))

    def _path_row(self, parent, label: str, variable: tk.StringVar, command) -> None:
        ttk.Label(parent, text=label).pack(anchor="w")
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(2, 10))
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse...", command=command).pack(side="left", padx=(8, 0))

    def _choose_video(self) -> None:
        value = filedialog.askopenfilename(
            title="Select N21 camera-calibrated video",
            filetypes=[("Video", "*.avi *.mp4 *.mov *.mkv"), ("All files", "*.*")],
        )
        if value:
            self.video.set(value)

    def _choose_output(self) -> None:
        value = filedialog.askdirectory(title="Select output directory")
        if value:
            self.output.set(value)

    def _append(self, value: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", value + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _run(self) -> None:
        video = Path(self.video.get()).expanduser()
        if not video.is_file():
            messagebox.showerror("Missing video", f"Input video not found:\n{video}")
            return
        output = Path(self.output.get()).expanduser()
        self.run_button.configure(state="disabled")
        self.status.set("Running")
        command = [
            sys.executable,
            "-u",
            str(HERE / "reproduce_v06.py"),
            "--config",
            str(HERE / "config.yaml"),
            "--input-video",
            str(video.resolve()),
            "--output-root",
            str(output.resolve()),
        ]

        def worker() -> None:
            process = subprocess.Popen(
                command,
                cwd=HERE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                self.messages.put(line.rstrip())
            code = process.wait()
            self.messages.put(None if code == 0 else f"ERROR: runner exited with code {code}")

        threading.Thread(target=worker, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                value = self.messages.get_nowait()
                if value is None:
                    self.status.set("Completed")
                    self.run_button.configure(state="normal")
                    messagebox.showinfo("Completed", "V06 video preparation completed.")
                elif value.startswith("ERROR:"):
                    self._append(value)
                    self.status.set("Failed")
                    self.run_button.configure(state="normal")
                    messagebox.showerror("Failed", value)
                else:
                    self._append(value)
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    App().mainloop()

