"""
LUT Extractor — Extract cinematic color grades from any video.
Single-file GUI app. Package with: pyinstaller --onefile --windowed lut_extractor_app.py
"""

import json
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path

import cv2
import numpy as np

# ── Config ──
NUM_FRAMES = 5
LUT_SIZE = 33
APP_TITLE = "LUT Extractor"
APP_VERSION = "1.0"

PROMPT = """You are a professional film colorist analyzing cinematic frames.

Examine these frames and determine the EXACT color grading parameters that would recreate this look when applied to neutral/flat Rec.709 footage.

Return ONLY a JSON object with these parameters (no markdown, no explanation, just the JSON):

{
  "shadows_r": 0.0,
  "shadows_g": 0.0,
  "shadows_b": 0.0,
  "midtones_r": 1.0,
  "midtones_g": 1.0,
  "midtones_b": 1.0,
  "highlights_r": 1.0,
  "highlights_g": 1.0,
  "highlights_b": 1.0,
  "contrast": 0.0,
  "saturation": 1.0,
  "temperature": 0,
  "tint": 0,
  "curve_blacks": 0.0,
  "curve_shadows": 0.25,
  "curve_midpoint": 0.5,
  "curve_highlights": 0.75,
  "curve_whites": 1.0,
  "grade_name": "Name of the look"
}

Parameter ranges:
- shadows_r/g/b: -0.15 to 0.15 (color offset in dark areas. Example: teal shadows = r:-0.03, g:0.02, b:0.06)
- midtones_r/g/b: 0.7 to 1.3 (multiplier for mid-range. 1.0 = no change)
- highlights_r/g/b: 0.7 to 1.3 (multiplier for bright areas. 1.0 = no change)
- contrast: -0.5 to 0.5 (0 = no change)
- saturation: 0.3 to 1.5 (1.0 = no change, <1 = desaturated)
- temperature: -50 to 50 (positive = warmer/orange, negative = cooler/blue)
- tint: -30 to 30 (positive = magenta, negative = green)
- curve_blacks: 0.0 to 0.15 (0 = true black, higher = lifted/milky blacks)
- curve_shadows: 0.15 to 0.35
- curve_midpoint: 0.4 to 0.6
- curve_highlights: 0.65 to 0.85
- curve_whites: 0.85 to 1.0 (lower = rolled-off highlights)
- grade_name: short descriptive name

Be precise. These numbers will directly generate a LUT file."""

REQUIRED_KEYS = [
    "shadows_r", "shadows_g", "shadows_b",
    "midtones_r", "midtones_g", "midtones_b",
    "highlights_r", "highlights_g", "highlights_b",
    "contrast", "saturation", "temperature", "tint",
    "curve_blacks", "curve_shadows", "curve_midpoint",
    "curve_highlights", "curve_whites",
]


# ── LUT Math ──

def apply_curve(x, p):
    pts_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    pts_y = np.array([p["curve_blacks"], p["curve_shadows"], p["curve_midpoint"],
                       p["curve_highlights"], p["curve_whites"]])
    return float(np.interp(x, pts_x, pts_y))


def apply_lgg(r, g, b, p):
    def ch(val, shadow, mid, hi):
        sw = 1.0 - val
        hw = val
        mw = 1.0 - abs(val - 0.5) * 2
        val = val + shadow * sw * 0.5
        val = val * (mid ** (mw * 0.3))
        val = val * (hi ** hw)
        return val
    return (ch(r, p["shadows_r"], p["midtones_r"], p["highlights_r"]),
            ch(g, p["shadows_g"], p["midtones_g"], p["highlights_g"]),
            ch(b, p["shadows_b"], p["midtones_b"], p["highlights_b"]))


def apply_temperature(r, g, b, p):
    t = p["temperature"] / 100.0
    ti = p["tint"] / 100.0
    return r + t * 0.05 + ti * 0.015, g - ti * 0.03, b - t * 0.05 + ti * 0.015


def apply_saturation(r, g, b, p):
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    s = p["saturation"]
    return luma + (r - luma) * s, luma + (g - luma) * s, luma + (b - luma) * s


def apply_contrast(r, g, b, p):
    c = p["contrast"]
    cc = lambda v: 0.5 + (v - 0.5) * (1.0 + c)
    return cc(r), cc(g), cc(b)


def generate_cube(params, size, progress_cb=None):
    lines = [f'TITLE "{params.get("grade_name", "Custom Grade")}"', f"LUT_3D_SIZE {size}", ""]
    total = size ** 3
    count = 0
    for bi in range(size):
        for gi in range(size):
            for ri in range(size):
                r, g, b = ri / (size - 1), gi / (size - 1), bi / (size - 1)
                r, g, b = apply_curve(r, params), apply_curve(g, params), apply_curve(b, params)
                r, g, b = apply_contrast(r, g, b, params)
                r, g, b = apply_lgg(r, g, b, params)
                r, g, b = apply_temperature(r, g, b, params)
                r, g, b = apply_saturation(r, g, b, params)
                lines.append(f"{max(0,min(1,r)):.6f} {max(0,min(1,g)):.6f} {max(0,min(1,b)):.6f}")
                count += 1
                if progress_cb and count % 5000 == 0:
                    progress_cb(count / total)
    if progress_cb:
        progress_cb(1.0)
    return "\n".join(lines)


def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = np.linspace(total * 0.1, total * 0.9, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (1280, int(h * scale)))
            frames.append(frame)
    cap.release()
    return frames


# ── GUI ──

class LUTExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_TITLE} v{APP_VERSION}")
        self.root.geometry("750x820")
        self.root.resizable(True, True)

        self.video_path = None
        self.frames_dir = None
        self.params = None

        self._build_ui()

    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg="#1a1a2e", height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="LUT Extractor", font=("Helvetica", 18, "bold"),
                 bg="#1a1a2e", fg="white").pack(side="left", padx=20, pady=10)
        tk.Label(header, text="Extract cinematic LUTs from any video",
                 font=("Helvetica", 10), bg="#1a1a2e", fg="#888").pack(side="left", pady=10)

        main = tk.Frame(self.root, padx=20, pady=10)
        main.pack(fill="both", expand=True)

        # ── Step 1: Select Video ──
        s1 = tk.LabelFrame(main, text="  Step 1: Select Video  ", font=("Helvetica", 11, "bold"), padx=15, pady=10)
        s1.pack(fill="x", pady=(0, 10))

        btn_frame = tk.Frame(s1)
        btn_frame.pack(fill="x")
        self.btn_select = tk.Button(btn_frame, text="Select Video File", font=("Helvetica", 11),
                                     bg="#0066cc", fg="white", padx=20, pady=8, cursor="hand2",
                                     command=self._select_video)
        self.btn_select.pack(side="left")
        self.lbl_video = tk.Label(btn_frame, text="No video selected", font=("Helvetica", 10), fg="#666")
        self.lbl_video.pack(side="left", padx=15)

        self.lbl_frames_status = tk.Label(s1, text="", font=("Helvetica", 10), fg="#228B22")
        self.lbl_frames_status.pack(anchor="w", pady=(5, 0))

        # ── Step 2: Get AI Analysis ──
        s2 = tk.LabelFrame(main, text="  Step 2: Get AI Analysis (free, no API key needed)  ",
                           font=("Helvetica", 11, "bold"), padx=15, pady=10)
        s2.pack(fill="x", pady=(0, 10))

        instructions = tk.Label(s2, justify="left", font=("Helvetica", 10), fg="#333",
                                wraplength=680, text=(
            "1. Click 'Open Gemini' below (opens in your browser)\n"
            "2. Upload ALL frame images from the frames folder\n"
            "3. Click 'Copy Prompt' and paste it into Gemini\n"
            "4. Copy Gemini's JSON response and paste it below"
        ))
        instructions.pack(anchor="w")

        btn_frame2 = tk.Frame(s2)
        btn_frame2.pack(fill="x", pady=(8, 5))
        self.btn_open_gemini = tk.Button(btn_frame2, text="Open Gemini", font=("Helvetica", 10),
                                          bg="#4285F4", fg="white", padx=12, pady=5, cursor="hand2",
                                          command=self._open_gemini, state="disabled")
        self.btn_open_gemini.pack(side="left", padx=(0, 8))
        self.btn_copy_prompt = tk.Button(btn_frame2, text="Copy Prompt to Clipboard", font=("Helvetica", 10),
                                          bg="#34A853", fg="white", padx=12, pady=5, cursor="hand2",
                                          command=self._copy_prompt, state="disabled")
        self.btn_copy_prompt.pack(side="left", padx=(0, 8))
        self.btn_open_folder = tk.Button(btn_frame2, text="Open Frames Folder", font=("Helvetica", 10),
                                          padx=12, pady=5, cursor="hand2",
                                          command=self._open_frames_folder, state="disabled")
        self.btn_open_folder.pack(side="left")

        # ── Step 3: Paste Response & Generate ──
        s3 = tk.LabelFrame(main, text="  Step 3: Paste Gemini's Response & Generate LUT  ",
                           font=("Helvetica", 11, "bold"), padx=15, pady=10)
        s3.pack(fill="both", expand=True, pady=(0, 10))

        tk.Label(s3, text="Paste the JSON response from Gemini here:", font=("Helvetica", 10)).pack(anchor="w")

        self.txt_json = scrolledtext.ScrolledText(s3, height=10, font=("Consolas", 10), wrap="word",
                                                   bg="#f8f8f8", relief="solid", borderwidth=1)
        self.txt_json.pack(fill="both", expand=True, pady=(5, 8))

        bottom = tk.Frame(s3)
        bottom.pack(fill="x")

        self.btn_generate = tk.Button(bottom, text="Generate .cube LUT", font=("Helvetica", 12, "bold"),
                                       bg="#EA4335", fg="white", padx=25, pady=10, cursor="hand2",
                                       command=self._generate_lut, state="disabled")
        self.btn_generate.pack(side="left")

        self.progress = ttk.Progressbar(bottom, length=200, mode="determinate")
        self.progress.pack(side="left", padx=15)

        self.lbl_status = tk.Label(bottom, text="", font=("Helvetica", 10), fg="#228B22")
        self.lbl_status.pack(side="left")

        # ── Footer ──
        footer = tk.Frame(self.root, bg="#f0f0f0", height=35)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        tk.Label(footer, text="Powered by Gemini AI  |  .cube files work in Premiere Pro, DaVinci Resolve, Final Cut Pro",
                 font=("Helvetica", 9), bg="#f0f0f0", fg="#888").pack(pady=8)

        # Enable generate button when text is pasted
        self.txt_json.bind("<KeyRelease>", self._on_json_change)

    def _select_video(self):
        path = filedialog.askopenfilename(
            title="Select a video or image file",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.avi *.webm"),
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return

        self.video_path = Path(path)
        self.lbl_video.config(text=self.video_path.name, fg="#333")
        self.lbl_frames_status.config(text="Extracting frames...", fg="#cc6600")
        self.root.update()

        # Extract frames in background
        threading.Thread(target=self._extract_frames_thread, daemon=True).start()

    def _extract_frames_thread(self):
        vp = self.video_path
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # Create frames directory
        self.frames_dir = vp.parent / f"{vp.stem}_frames"
        self.frames_dir.mkdir(exist_ok=True)

        if vp.suffix.lower() in image_exts:
            # Just copy the image
            import shutil
            dest = self.frames_dir / f"frame_1{vp.suffix}"
            shutil.copy2(str(vp), str(dest))
            count = 1
        else:
            frames = extract_frames(vp, NUM_FRAMES)
            count = 0
            for i, frame in enumerate(frames):
                fname = self.frames_dir / f"frame_{i+1}.jpg"
                cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                count += 1

        self.root.after(0, self._frames_done, count)

    def _frames_done(self, count):
        self.lbl_frames_status.config(
            text=f"{count} frames extracted to: {self.frames_dir.name}/",
            fg="#228B22"
        )
        self.btn_open_gemini.config(state="normal")
        self.btn_copy_prompt.config(state="normal")
        self.btn_open_folder.config(state="normal")
        self.btn_generate.config(state="normal")

    def _open_gemini(self):
        import webbrowser
        webbrowser.open("https://gemini.google.com/app")

    def _copy_prompt(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(PROMPT)
        self.btn_copy_prompt.config(text="Copied!", bg="#228B22")
        self.root.after(2000, lambda: self.btn_copy_prompt.config(text="Copy Prompt to Clipboard", bg="#34A853"))

    def _open_frames_folder(self):
        if self.frames_dir and self.frames_dir.exists():
            if sys.platform == "win32":
                os.startfile(str(self.frames_dir))
            elif sys.platform == "darwin":
                os.system(f'open "{self.frames_dir}"')
            else:
                os.system(f'xdg-open "{self.frames_dir}"')

    def _on_json_change(self, event=None):
        text = self.txt_json.get("1.0", "end").strip()
        if text and len(text) > 20:
            self.btn_generate.config(state="normal")

    def _generate_lut(self):
        text = self.txt_json.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("No Data", "Please paste Gemini's JSON response first.")
            return

        # Strip markdown code fences
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
        text = text.strip()

        # Try to extract JSON if there's extra text around it
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]

        try:
            self.params = json.loads(text)
        except json.JSONDecodeError as e:
            messagebox.showerror("Invalid JSON", f"Could not parse the response:\n\n{e}\n\nMake sure you copied the complete JSON from Gemini.")
            return

        missing = [k for k in REQUIRED_KEYS if k not in self.params]
        if missing:
            messagebox.showerror("Missing Parameters", f"The JSON is missing these keys:\n\n{', '.join(missing)}\n\nAsk Gemini to include all parameters.")
            return

        if "grade_name" not in self.params:
            self.params["grade_name"] = "Custom Grade"

        # Generate in background
        self.btn_generate.config(state="disabled", text="Generating...")
        self.lbl_status.config(text="Generating LUT...", fg="#cc6600")
        self.progress["value"] = 0
        threading.Thread(target=self._generate_thread, daemon=True).start()

    def _generate_thread(self):
        def progress_cb(pct):
            self.root.after(0, lambda: self.progress.configure(value=pct * 100))

        cube = generate_cube(self.params, LUT_SIZE, progress_cb)

        safe_name = self.params["grade_name"].replace(" ", "_").replace("/", "-").replace("&", "and")[:40]

        # Save next to original video
        if self.video_path:
            out_dir = self.video_path.parent
        else:
            out_dir = Path(".")

        out_path = out_dir / f"{safe_name}.cube"
        with open(out_path, "w") as f:
            f.write(cube)

        # Save params too
        json_path = out_dir / f"{safe_name}_params.json"
        with open(json_path, "w") as f:
            json.dump(self.params, f, indent=2)

        self.root.after(0, self._generate_done, out_path)

    def _generate_done(self, out_path):
        self.progress["value"] = 100
        self.btn_generate.config(state="normal", text="Generate .cube LUT")
        self.lbl_status.config(text=f"Saved: {out_path.name}", fg="#228B22")

        result = messagebox.askquestion(
            "LUT Generated!",
            f"LUT saved to:\n{out_path}\n\nGrade: {self.params['grade_name']}\n\n"
            f"To use in Premiere Pro:\nLumetri Color > Creative > Look > Browse\n\n"
            f"Open the folder?",
            icon="info"
        )
        if result == "yes":
            folder = out_path.parent
            if sys.platform == "win32":
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                os.system(f'open "{folder}"')
            else:
                os.system(f'xdg-open "{folder}"')


def main():
    root = tk.Tk()

    # Set icon if available
    try:
        if sys.platform == "win32":
            root.iconbitmap(default="")
    except Exception:
        pass

    # Style
    style = ttk.Style()
    style.configure("TProgressbar", thickness=20)

    app = LUTExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
