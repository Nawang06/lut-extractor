"""
LUT Extractor Pro — Extract, create, and customize cinematic LUTs.
Single-file GUI app. Package with: pyinstaller --onefile --windowed lut_extractor_pro.py

Features:
  Tab 1: Extract from Media — analyze video/image frames via Gemini AI
  Tab 2: Create from Text — describe a look, get a LUT
  Tab 3: Preset Library — 22 built-in cinematic presets (no AI needed)
  Tab 4: Editor — 17 sliders with real-time before/after preview
"""

import copy
import json
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: Constants & Configuration
# ═══════════════════════════════════════════════════════════════════

APP_TITLE = "LUT Extractor Pro"
APP_VERSION = "2.0"
WINDOW_SIZE = "1100x800"
MIN_SIZE = (950, 700)
PREVIEW_W, PREVIEW_H = 280, 200
LUT_SIZE = 33
NUM_FRAMES = 5
DEBOUNCE_MS = 80
INTENSITY_DEBOUNCE_MS = 30

DEFAULT_PARAMS = {
    "curve_blacks": 0.00, "curve_shadows": 0.25, "curve_midpoint": 0.50,
    "curve_highlights": 0.75, "curve_whites": 1.00,
    "shadows_r": 0.0, "shadows_g": 0.0, "shadows_b": 0.0,
    "midtones_r": 1.0, "midtones_g": 1.0, "midtones_b": 1.0,
    "highlights_r": 0.0, "highlights_g": 0.0, "highlights_b": 0.0,
    "contrast": 1.0, "saturation": 1.0, "temperature": 0.0, "tint": 0.0,
    "grade_name": "Untitled",
}

SLIDER_CONFIG = {
    "curve_blacks":     {"from": 0.00, "to": 0.15, "res": 0.005, "section": "Tone Curve"},
    "curve_shadows":    {"from": 0.10, "to": 0.40, "res": 0.005, "section": "Tone Curve"},
    "curve_midpoint":   {"from": 0.30, "to": 0.70, "res": 0.005, "section": "Tone Curve"},
    "curve_highlights": {"from": 0.60, "to": 0.90, "res": 0.005, "section": "Tone Curve"},
    "curve_whites":     {"from": 0.85, "to": 1.00, "res": 0.005, "section": "Tone Curve"},
    "shadows_r":    {"from": -0.20, "to": 0.20, "res": 0.005, "section": "Shadows"},
    "shadows_g":    {"from": -0.20, "to": 0.20, "res": 0.005, "section": "Shadows"},
    "shadows_b":    {"from": -0.20, "to": 0.20, "res": 0.005, "section": "Shadows"},
    "midtones_r":   {"from": 0.50, "to": 1.50, "res": 0.01,  "section": "Midtones"},
    "midtones_g":   {"from": 0.50, "to": 1.50, "res": 0.01,  "section": "Midtones"},
    "midtones_b":   {"from": 0.50, "to": 1.50, "res": 0.01,  "section": "Midtones"},
    "highlights_r": {"from": -0.20, "to": 0.20, "res": 0.005, "section": "Highlights"},
    "highlights_g": {"from": -0.20, "to": 0.20, "res": 0.005, "section": "Highlights"},
    "highlights_b": {"from": -0.20, "to": 0.20, "res": 0.005, "section": "Highlights"},
    "contrast":     {"from": 0.50, "to": 2.00, "res": 0.01,  "section": "Global"},
    "saturation":   {"from": 0.00, "to": 2.00, "res": 0.01,  "section": "Global"},
    "temperature":  {"from": -0.30, "to": 0.30, "res": 0.01,  "section": "Global"},
    "tint":         {"from": -0.30, "to": 0.30, "res": 0.01,  "section": "Global"},
}

# ═══════════════════════════════════════════════════════════════════
# SECTION 2: Preset Library (22 presets)
# ═══════════════════════════════════════════════════════════════════

PRESETS = {
    "Cinematic": {
        "Teal & Orange": {
            "desc": "Blockbuster look with warm highlights and cool teal shadows. Think Transformers, Mad Max.",
            "params": {
                "curve_blacks": 0.03, "curve_shadows": 0.22, "curve_midpoint": 0.50,
                "curve_highlights": 0.78, "curve_whites": 0.97,
                "shadows_r": -0.08, "shadows_g": 0.04, "shadows_b": 0.12,
                "midtones_r": 1.05, "midtones_g": 0.97, "midtones_b": 0.90,
                "highlights_r": 0.10, "highlights_g": 0.02, "highlights_b": -0.08,
                "contrast": 1.15, "saturation": 1.20, "temperature": 0.04, "tint": 0.0,
                "grade_name": "Teal & Orange",
            },
        },
        "Bleach Bypass": {
            "desc": "High contrast, desaturated. Saving Private Ryan, Se7en look.",
            "params": {
                "curve_blacks": 0.02, "curve_shadows": 0.20, "curve_midpoint": 0.48,
                "curve_highlights": 0.80, "curve_whites": 0.98,
                "shadows_r": 0.0, "shadows_g": 0.0, "shadows_b": 0.0,
                "midtones_r": 1.0, "midtones_g": 1.0, "midtones_b": 1.0,
                "highlights_r": 0.02, "highlights_g": 0.02, "highlights_b": 0.02,
                "contrast": 1.50, "saturation": 0.45, "temperature": 0.0, "tint": 0.0,
                "grade_name": "Bleach Bypass",
            },
        },
        "Vintage Film": {
            "desc": "Warm, faded look with lifted blacks and muted colors. Indie film feel.",
            "params": {
                "curve_blacks": 0.06, "curve_shadows": 0.24, "curve_midpoint": 0.50,
                "curve_highlights": 0.76, "curve_whites": 0.94,
                "shadows_r": 0.04, "shadows_g": 0.02, "shadows_b": -0.02,
                "midtones_r": 1.03, "midtones_g": 1.01, "midtones_b": 0.95,
                "highlights_r": 0.05, "highlights_g": 0.03, "highlights_b": -0.03,
                "contrast": 0.90, "saturation": 0.80, "temperature": 0.06, "tint": 0.01,
                "grade_name": "Vintage Film",
            },
        },
        "Moody Desaturated": {
            "desc": "Dark, muted look with cool undertones. David Fincher style.",
            "params": {
                "curve_blacks": 0.04, "curve_shadows": 0.22, "curve_midpoint": 0.46,
                "curve_highlights": 0.74, "curve_whites": 0.96,
                "shadows_r": 0.02, "shadows_g": 0.02, "shadows_b": 0.04,
                "midtones_r": 0.98, "midtones_g": 0.98, "midtones_b": 1.02,
                "highlights_r": -0.02, "highlights_g": -0.01, "highlights_b": 0.01,
                "contrast": 1.05, "saturation": 0.55, "temperature": -0.02, "tint": 0.0,
                "grade_name": "Moody Desaturated",
            },
        },
        "Golden Hour": {
            "desc": "Warm, golden sunlight. Rich amber tones. Terrence Malick vibes.",
            "params": {
                "curve_blacks": 0.02, "curve_shadows": 0.24, "curve_midpoint": 0.52,
                "curve_highlights": 0.78, "curve_whites": 0.98,
                "shadows_r": 0.05, "shadows_g": 0.02, "shadows_b": -0.04,
                "midtones_r": 1.08, "midtones_g": 1.02, "midtones_b": 0.88,
                "highlights_r": 0.08, "highlights_g": 0.04, "highlights_b": -0.06,
                "contrast": 1.05, "saturation": 1.15, "temperature": 0.12, "tint": 0.02,
                "grade_name": "Golden Hour",
            },
        },
        "Noir": {
            "desc": "Dark, contrasty, near-monochrome. Classic film noir mood.",
            "params": {
                "curve_blacks": 0.0, "curve_shadows": 0.18, "curve_midpoint": 0.42,
                "curve_highlights": 0.72, "curve_whites": 0.95,
                "shadows_r": 0.0, "shadows_g": 0.0, "shadows_b": 0.02,
                "midtones_r": 1.0, "midtones_g": 1.0, "midtones_b": 1.0,
                "highlights_r": 0.0, "highlights_g": 0.0, "highlights_b": 0.0,
                "contrast": 1.45, "saturation": 0.15, "temperature": -0.02, "tint": 0.0,
                "grade_name": "Noir",
            },
        },
        "Pastel Faded": {
            "desc": "Light, airy, lifted everything. Instagram/Pinterest aesthetic.",
            "params": {
                "curve_blacks": 0.08, "curve_shadows": 0.28, "curve_midpoint": 0.52,
                "curve_highlights": 0.74, "curve_whites": 0.92,
                "shadows_r": 0.03, "shadows_g": 0.02, "shadows_b": 0.04,
                "midtones_r": 1.02, "midtones_g": 1.01, "midtones_b": 1.03,
                "highlights_r": 0.02, "highlights_g": 0.01, "highlights_b": 0.02,
                "contrast": 0.80, "saturation": 0.65, "temperature": 0.02, "tint": 0.01,
                "grade_name": "Pastel Faded",
            },
        },
        "High Contrast Drama": {
            "desc": "Punchy, dramatic, deep blacks and bright whites.",
            "params": {
                "curve_blacks": 0.0, "curve_shadows": 0.15, "curve_midpoint": 0.50,
                "curve_highlights": 0.85, "curve_whites": 1.0,
                "shadows_r": -0.02, "shadows_g": -0.01, "shadows_b": 0.0,
                "midtones_r": 1.02, "midtones_g": 1.0, "midtones_b": 0.98,
                "highlights_r": 0.03, "highlights_g": 0.02, "highlights_b": 0.0,
                "contrast": 1.60, "saturation": 1.10, "temperature": 0.02, "tint": 0.0,
                "grade_name": "High Contrast Drama",
            },
        },
    },
    "Film Stocks": {
        "Kodak Vision3 500T": {
            "desc": "Tungsten motion picture stock. Warm highlights, blue shadows under daylight.",
            "params": {
                "curve_blacks": 0.03, "curve_shadows": 0.23, "curve_midpoint": 0.50,
                "curve_highlights": 0.77, "curve_whites": 0.97,
                "shadows_r": -0.02, "shadows_g": 0.0, "shadows_b": 0.06,
                "midtones_r": 1.04, "midtones_g": 1.0, "midtones_b": 0.94,
                "highlights_r": 0.06, "highlights_g": 0.03, "highlights_b": -0.04,
                "contrast": 1.12, "saturation": 1.05, "temperature": 0.05, "tint": 0.01,
                "grade_name": "Kodak Vision3 500T",
            },
        },
        "Kodak Portra 400": {
            "desc": "Portrait film. Low contrast, warm skin tones, muted greens.",
            "params": {
                "curve_blacks": 0.04, "curve_shadows": 0.25, "curve_midpoint": 0.51,
                "curve_highlights": 0.76, "curve_whites": 0.96,
                "shadows_r": 0.03, "shadows_g": 0.01, "shadows_b": -0.01,
                "midtones_r": 1.03, "midtones_g": 0.99, "midtones_b": 0.96,
                "highlights_r": 0.04, "highlights_g": 0.02, "highlights_b": -0.02,
                "contrast": 0.92, "saturation": 0.88, "temperature": 0.05, "tint": 0.02,
                "grade_name": "Kodak Portra 400",
            },
        },
        "Fuji Velvia 50": {
            "desc": "Slide film. Extremely saturated, deep blacks, vivid greens and blues.",
            "params": {
                "curve_blacks": 0.0, "curve_shadows": 0.20, "curve_midpoint": 0.50,
                "curve_highlights": 0.80, "curve_whites": 1.0,
                "shadows_r": -0.01, "shadows_g": 0.02, "shadows_b": 0.01,
                "midtones_r": 0.98, "midtones_g": 1.04, "midtones_b": 1.02,
                "highlights_r": 0.02, "highlights_g": 0.03, "highlights_b": 0.01,
                "contrast": 1.30, "saturation": 1.45, "temperature": -0.02, "tint": -0.01,
                "grade_name": "Fuji Velvia 50",
            },
        },
        "CineStill 800T": {
            "desc": "Tungsten cinema stock. Halation glow, amber cast, lifted shadows.",
            "params": {
                "curve_blacks": 0.05, "curve_shadows": 0.24, "curve_midpoint": 0.50,
                "curve_highlights": 0.76, "curve_whites": 0.96,
                "shadows_r": 0.06, "shadows_g": 0.0, "shadows_b": -0.03,
                "midtones_r": 1.06, "midtones_g": 0.98, "midtones_b": 0.92,
                "highlights_r": 0.08, "highlights_g": 0.02, "highlights_b": -0.05,
                "contrast": 1.05, "saturation": 1.10, "temperature": 0.10, "tint": 0.03,
                "grade_name": "CineStill 800T",
            },
        },
        "Kodachrome 64": {
            "desc": "Legendary slide film. Deep reds, strong cyan-blue, warm midtones.",
            "params": {
                "curve_blacks": 0.01, "curve_shadows": 0.22, "curve_midpoint": 0.50,
                "curve_highlights": 0.79, "curve_whites": 0.99,
                "shadows_r": 0.02, "shadows_g": -0.01, "shadows_b": -0.03,
                "midtones_r": 1.06, "midtones_g": 1.0, "midtones_b": 0.94,
                "highlights_r": 0.05, "highlights_g": 0.01, "highlights_b": -0.04,
                "contrast": 1.20, "saturation": 1.25, "temperature": 0.06, "tint": 0.01,
                "grade_name": "Kodachrome 64",
            },
        },
    },
    "Technical": {
        "S-Log3 to Rec.709": {
            "desc": "Sony S-Log3 to standard display. Adds contrast and saturation to flat footage.",
            "params": {
                "curve_blacks": 0.0, "curve_shadows": 0.12, "curve_midpoint": 0.45,
                "curve_highlights": 0.82, "curve_whites": 1.0,
                "shadows_r": 0.0, "shadows_g": 0.0, "shadows_b": 0.0,
                "midtones_r": 1.0, "midtones_g": 1.0, "midtones_b": 1.0,
                "highlights_r": 0.0, "highlights_g": 0.0, "highlights_b": 0.0,
                "contrast": 1.65, "saturation": 1.30, "temperature": 0.0, "tint": 0.0,
                "grade_name": "S-Log3 to Rec709",
            },
        },
        "ARRI Log-C to Rec.709": {
            "desc": "ARRI Alexa Log-C to standard display. Smooth highlight rolloff.",
            "params": {
                "curve_blacks": 0.0, "curve_shadows": 0.14, "curve_midpoint": 0.47,
                "curve_highlights": 0.83, "curve_whites": 1.0,
                "shadows_r": 0.0, "shadows_g": 0.0, "shadows_b": 0.0,
                "midtones_r": 1.0, "midtones_g": 1.0, "midtones_b": 1.0,
                "highlights_r": 0.0, "highlights_g": 0.0, "highlights_b": 0.0,
                "contrast": 1.55, "saturation": 1.25, "temperature": 0.0, "tint": 0.0,
                "grade_name": "ARRI LogC to Rec709",
            },
        },
        "Canon C-Log to Rec.709": {
            "desc": "Canon C-Log to standard display. Good starting point for Canon cinema cameras.",
            "params": {
                "curve_blacks": 0.0, "curve_shadows": 0.13, "curve_midpoint": 0.46,
                "curve_highlights": 0.81, "curve_whites": 1.0,
                "shadows_r": 0.0, "shadows_g": 0.0, "shadows_b": 0.0,
                "midtones_r": 1.0, "midtones_g": 1.0, "midtones_b": 1.0,
                "highlights_r": 0.0, "highlights_g": 0.0, "highlights_b": 0.0,
                "contrast": 1.60, "saturation": 1.28, "temperature": 0.0, "tint": 0.0,
                "grade_name": "Canon CLog to Rec709",
            },
        },
    },
    "Moods": {
        "Horror Green": {
            "desc": "Sickly green tint, dark, unsettling. Classic horror film look.",
            "params": {
                "curve_blacks": 0.02, "curve_shadows": 0.18, "curve_midpoint": 0.44,
                "curve_highlights": 0.72, "curve_whites": 0.95,
                "shadows_r": -0.04, "shadows_g": 0.06, "shadows_b": -0.02,
                "midtones_r": 0.92, "midtones_g": 1.08, "midtones_b": 0.94,
                "highlights_r": -0.04, "highlights_g": 0.04, "highlights_b": -0.02,
                "contrast": 1.25, "saturation": 0.70, "temperature": -0.06, "tint": -0.08,
                "grade_name": "Horror Green",
            },
        },
        "Romantic Warm": {
            "desc": "Soft, warm, dreamy. Wedding films and love stories.",
            "params": {
                "curve_blacks": 0.04, "curve_shadows": 0.26, "curve_midpoint": 0.52,
                "curve_highlights": 0.77, "curve_whites": 0.96,
                "shadows_r": 0.04, "shadows_g": 0.01, "shadows_b": -0.02,
                "midtones_r": 1.05, "midtones_g": 1.01, "midtones_b": 0.94,
                "highlights_r": 0.06, "highlights_g": 0.03, "highlights_b": -0.03,
                "contrast": 0.90, "saturation": 0.95, "temperature": 0.08, "tint": 0.03,
                "grade_name": "Romantic Warm",
            },
        },
        "Sci-Fi Blue": {
            "desc": "Cold, clinical blue. Futuristic and sterile. Blade Runner vibes.",
            "params": {
                "curve_blacks": 0.03, "curve_shadows": 0.20, "curve_midpoint": 0.47,
                "curve_highlights": 0.76, "curve_whites": 0.97,
                "shadows_r": -0.06, "shadows_g": -0.02, "shadows_b": 0.08,
                "midtones_r": 0.90, "midtones_g": 0.96, "midtones_b": 1.10,
                "highlights_r": -0.04, "highlights_g": 0.0, "highlights_b": 0.08,
                "contrast": 1.20, "saturation": 0.85, "temperature": -0.10, "tint": -0.02,
                "grade_name": "Sci-Fi Blue",
            },
        },
        "Western Dusty": {
            "desc": "Warm, desaturated, dusty. Spaghetti western, desert landscapes.",
            "params": {
                "curve_blacks": 0.05, "curve_shadows": 0.26, "curve_midpoint": 0.52,
                "curve_highlights": 0.76, "curve_whites": 0.94,
                "shadows_r": 0.04, "shadows_g": 0.03, "shadows_b": -0.02,
                "midtones_r": 1.06, "midtones_g": 1.02, "midtones_b": 0.90,
                "highlights_r": 0.05, "highlights_g": 0.03, "highlights_b": -0.04,
                "contrast": 0.95, "saturation": 0.75, "temperature": 0.08, "tint": 0.02,
                "grade_name": "Western Dusty",
            },
        },
        "Matrix Green": {
            "desc": "Heavy green tint with desaturation. The Matrix digital rain look.",
            "params": {
                "curve_blacks": 0.01, "curve_shadows": 0.19, "curve_midpoint": 0.46,
                "curve_highlights": 0.75, "curve_whites": 0.97,
                "shadows_r": -0.06, "shadows_g": 0.08, "shadows_b": -0.04,
                "midtones_r": 0.88, "midtones_g": 1.12, "midtones_b": 0.90,
                "highlights_r": -0.04, "highlights_g": 0.06, "highlights_b": -0.03,
                "contrast": 1.30, "saturation": 0.60, "temperature": -0.05, "tint": -0.10,
                "grade_name": "Matrix Green",
            },
        },
        "Cyberpunk Neon": {
            "desc": "Vivid magenta/cyan, high saturation. Neon-lit nightlife aesthetic.",
            "params": {
                "curve_blacks": 0.03, "curve_shadows": 0.21, "curve_midpoint": 0.48,
                "curve_highlights": 0.78, "curve_whites": 0.97,
                "shadows_r": 0.04, "shadows_g": -0.04, "shadows_b": 0.08,
                "midtones_r": 1.06, "midtones_g": 0.92, "midtones_b": 1.08,
                "highlights_r": 0.06, "highlights_g": -0.02, "highlights_b": 0.06,
                "contrast": 1.25, "saturation": 1.30, "temperature": -0.03, "tint": 0.05,
                "grade_name": "Cyberpunk Neon",
            },
        },
    },
}


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: LUT Math Engine
# ═══════════════════════════════════════════════════════════════════

class LUTEngine:
    def __init__(self, size=LUT_SIZE):
        self.size = size
        self.identity = self._generate_identity()

    def _generate_identity(self):
        s = self.size
        r = np.linspace(0, 1, s)
        lut = np.zeros((s, s, s, 3), dtype=np.float32)
        for ri in range(s):
            for gi in range(s):
                for bi in range(s):
                    lut[ri, gi, bi] = [r[ri], r[gi], r[bi]]
        return lut

    def generate_lut(self, params):
        lut = self.identity.copy()
        s = self.size
        # Step 1: Tone curve
        pts_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        pts_y = np.array([
            params["curve_blacks"], params["curve_shadows"], params["curve_midpoint"],
            params["curve_highlights"], params["curve_whites"],
        ], dtype=np.float32)
        flat = lut.reshape(-1)
        mapped = np.interp(flat, pts_x, pts_y).astype(np.float32)
        lut = mapped.reshape(s, s, s, 3)

        # Step 2: Contrast
        c = params["contrast"]
        lut = np.clip((lut - 0.5) * c + 0.5, 0, 1).astype(np.float32)

        # Step 3: Lift/Gamma/Gain
        for ch, suffix in enumerate(["_r", "_g", "_b"]):
            shadow = params[f"shadows{suffix}"]
            mid = params[f"midtones{suffix}"]
            hi = params[f"highlights{suffix}"]
            ch_data = lut[:, :, :, ch]
            ch_data = ch_data + shadow * (1.0 - ch_data)
            gamma = 1.0 / max(mid, 0.01)
            ch_data = np.power(np.clip(ch_data, 1e-6, 1.0), gamma)
            ch_data = ch_data * (1.0 + hi)
            lut[:, :, :, ch] = ch_data
        lut = np.clip(lut, 0, 1).astype(np.float32)

        # Step 4: Temperature/Tint
        t = params["temperature"]
        ti = params["tint"]
        lut[:, :, :, 0] += t * 0.5 + ti * 0.15
        lut[:, :, :, 1] += ti * 0.3
        lut[:, :, :, 2] -= t * 0.5 - ti * 0.15
        lut = np.clip(lut, 0, 1).astype(np.float32)

        # Step 5: Saturation
        sat = params["saturation"]
        gray = 0.2126 * lut[:, :, :, 0] + 0.7152 * lut[:, :, :, 1] + 0.0722 * lut[:, :, :, 2]
        for ch in range(3):
            lut[:, :, :, ch] = gray + sat * (lut[:, :, :, ch] - gray)
        lut = np.clip(lut, 0, 1).astype(np.float32)
        return lut

    def interpolate_intensity(self, lut, intensity):
        return (self.identity * (1.0 - intensity) + lut * intensity).astype(np.float32)

    def apply_to_image(self, image_bgr, lut_3d):
        h, w = image_bgr.shape[:2]
        img = image_bgr.astype(np.float32) / 255.0
        s = lut_3d.shape[0]
        b_ch, g_ch, r_ch = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        rf = np.clip(r_ch * (s - 1), 0, s - 2)
        gf = np.clip(g_ch * (s - 1), 0, s - 2)
        bf = np.clip(b_ch * (s - 1), 0, s - 2)
        r0, g0, b0 = rf.astype(int), gf.astype(int), bf.astype(int)
        r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1
        rd = (rf - r0)[..., np.newaxis]
        gd = (gf - g0)[..., np.newaxis]
        bd = (bf - b0)[..., np.newaxis]
        c000 = lut_3d[r0, g0, b0]; c001 = lut_3d[r0, g0, b1]
        c010 = lut_3d[r0, g1, b0]; c011 = lut_3d[r0, g1, b1]
        c100 = lut_3d[r1, g0, b0]; c101 = lut_3d[r1, g0, b1]
        c110 = lut_3d[r1, g1, b0]; c111 = lut_3d[r1, g1, b1]
        c00 = c000 * (1 - bd) + c001 * bd; c01 = c010 * (1 - bd) + c011 * bd
        c10 = c100 * (1 - bd) + c101 * bd; c11 = c110 * (1 - bd) + c111 * bd
        c0 = c00 * (1 - gd) + c01 * gd; c1 = c10 * (1 - gd) + c11 * gd
        mapped = c0 * (1 - rd) + c1 * rd
        result = np.zeros_like(img)
        result[:, :, 0] = mapped[:, :, 2]  # B
        result[:, :, 1] = mapped[:, :, 1]  # G
        result[:, :, 2] = mapped[:, :, 0]  # R
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def export_cube(self, lut_3d, filepath, title="Untitled"):
        s = lut_3d.shape[0]
        with open(filepath, "w") as f:
            f.write(f'TITLE "{title}"\n')
            f.write(f"# Created by {APP_TITLE} v{APP_VERSION}\n")
            f.write(f"LUT_3D_SIZE {s}\n\n")
            for ri in range(s):
                for gi in range(s):
                    for bi in range(s):
                        v = lut_3d[ri, gi, bi]
                        f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: Preview Engine
# ═══════════════════════════════════════════════════════════════════

class PreviewEngine:
    def __init__(self):
        self.has_image = False
        self.original_bgr = self._generate_default_image()
        self.preview_bgr = self._downscale(self.original_bgr)

    def _generate_default_image(self):
        """Simple gray placeholder prompting user to load their own image."""
        img = np.full((PREVIEW_H, PREVIEW_W, 3), 60, dtype=np.uint8)
        return img

    def _downscale(self, img):
        h, w = img.shape[:2]
        scale = min(PREVIEW_W / w, PREVIEW_H / h)
        if scale < 1.0:
            return cv2.resize(img, (int(w * scale), int(h * scale)))
        return img.copy()

    def load_image(self, filepath):
        img = cv2.imread(str(filepath))
        if img is not None:
            self.original_bgr = img
            self.preview_bgr = self._downscale(img)
            self.has_image = True
            return True
        return False

    def get_before_photo(self):
        return self._bgr_to_photo(self.preview_bgr)

    def get_after_photo(self, lut_engine, lut_3d):
        graded = lut_engine.apply_to_image(self.preview_bgr, lut_3d)
        return self._bgr_to_photo(graded)

    def _bgr_to_photo(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return ImageTk.PhotoImage(pil)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: Prompt Generator
# ═══════════════════════════════════════════════════════════════════

EXTRACT_PROMPT = """You are a professional film colorist analyzing cinematic frames.

Examine these frames and determine the EXACT color grading parameters that would recreate this look when applied to neutral/flat Rec.709 footage.

Return ONLY a JSON object (no markdown, no explanation) with these keys:

{"shadows_r": 0.0, "shadows_g": 0.0, "shadows_b": 0.0, "midtones_r": 1.0, "midtones_g": 1.0, "midtones_b": 1.0, "highlights_r": 0.0, "highlights_g": 0.0, "highlights_b": 0.0, "contrast": 1.0, "saturation": 1.0, "temperature": 0.0, "tint": 0.0, "curve_blacks": 0.0, "curve_shadows": 0.25, "curve_midpoint": 0.5, "curve_highlights": 0.75, "curve_whites": 1.0, "grade_name": "Name"}

Ranges: shadows/highlights_r/g/b: -0.20 to 0.20 | midtones_r/g/b: 0.5 to 1.5 | contrast: 0.5 to 2.0 | saturation: 0.0 to 2.0 | temperature/tint: -0.30 to 0.30 | curve points: 0.0 to 1.0

Be precise. These numbers directly generate a LUT file."""

TEXT_CREATE_PROMPT = """You are a professional colorist. Based on the description below, generate color grading parameters as a JSON object.

DESCRIPTION: {description}

Return ONLY a valid JSON object with these keys and numeric values:

{{"grade_name": "<descriptive name>", "curve_blacks": <0.00-0.15>, "curve_shadows": <0.10-0.40>, "curve_midpoint": <0.30-0.70>, "curve_highlights": <0.60-0.90>, "curve_whites": <0.85-1.00>, "shadows_r": <-0.20 to 0.20>, "shadows_g": <-0.20 to 0.20>, "shadows_b": <-0.20 to 0.20>, "midtones_r": <0.5-1.5>, "midtones_g": <0.5-1.5>, "midtones_b": <0.5-1.5>, "highlights_r": <-0.20 to 0.20>, "highlights_g": <-0.20 to 0.20>, "highlights_b": <-0.20 to 0.20>, "contrast": <0.5-2.0, 1.0=neutral>, "saturation": <0.0-2.0, 1.0=neutral>, "temperature": <-0.30 to 0.30, negative=cool, positive=warm>, "tint": <-0.30 to 0.30, negative=green, positive=magenta>}}

Think about what makes this look distinctive. For film stocks, consider spectral response. For movies, consider the actual DI grade. For moods, translate emotion into color science.
Return ONLY the JSON, no other text."""

REQUIRED_KEYS = [
    "shadows_r", "shadows_g", "shadows_b", "midtones_r", "midtones_g", "midtones_b",
    "highlights_r", "highlights_g", "highlights_b", "contrast", "saturation",
    "temperature", "tint", "curve_blacks", "curve_shadows", "curve_midpoint",
    "curve_highlights", "curve_whites",
]


def parse_gemini_response(text):
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in response")
    data = json.loads(text[start:end + 1])
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"Missing keys: {', '.join(missing)}")
    for k in REQUIRED_KEYS:
        data[k] = float(data[k])
    if "grade_name" not in data:
        data["grade_name"] = "AI Generated"
    return data


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: Frame Extraction
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: Tab 1 — Extract from Media
# ═══════════════════════════════════════════════════════════════════

THUMB_W, THUMB_H = 90, 60  # Thumbnail size for frame preview


class ExtractTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.file_path = None
        self.frames_dir = None
        self.is_video = False
        self.video_total_frames = 0
        self.selected_frames = []  # list of (frame_index, bgr_array)
        self._thumb_photos = []  # keep references to prevent GC
        self._scrub_photo = None
        self._build()

    def _build(self):
        # Step 1: Select file + frame selection (fixed height, not expandable)
        s1 = ttk.LabelFrame(self, text="  Step 1: Select Video or Image & Pick Frames  ", padding=8)
        s1.pack(fill="x", padx=10, pady=(5, 3))

        # Top row: file selection
        row = ttk.Frame(s1)
        row.pack(fill="x")
        ttk.Button(row, text="Select File", command=self._select_file).pack(side="left")
        self.lbl_file = ttk.Label(row, text="No file selected")
        self.lbl_file.pack(side="left", padx=10)

        # Auto-extract button + manual add
        row_mode = ttk.Frame(s1)
        row_mode.pack(fill="x", pady=(8, 0))
        self.btn_auto = ttk.Button(row_mode, text="Auto-Extract 5 Frames", command=self._auto_extract, state="disabled")
        self.btn_auto.pack(side="left", padx=(0, 8))
        self.btn_add_current = ttk.Button(row_mode, text="Add Current Frame", command=self._add_current_frame, state="disabled")
        self.btn_add_current.pack(side="left", padx=(0, 8))
        self.btn_clear = ttk.Button(row_mode, text="Clear All", command=self._clear_frames, state="disabled")
        self.btn_clear.pack(side="left")
        self.lbl_count = ttk.Label(row_mode, text="", foreground="gray")
        self.lbl_count.pack(side="right")

        # Video scrubber: preview + slider (only for videos)
        self.scrub_frame = ttk.Frame(s1)
        # Not packed yet — shown only when a video is loaded

        self.scrub_canvas = tk.Canvas(self.scrub_frame, width=160, height=100, bg="#1a1a1a",
                                       highlightthickness=1, highlightbackground="#444")
        self.scrub_canvas.pack(side="left", padx=(0, 10))
        self.scrub_canvas.create_text(80, 50, text="Scrub to preview", fill="#666", font=("Helvetica", 9))

        scrub_right = ttk.Frame(self.scrub_frame)
        scrub_right.pack(side="left", fill="x", expand=True)
        ttk.Label(scrub_right, text="Scrub through video:", font=("Helvetica", 9)).pack(anchor="w")
        self.scrub_slider = ttk.Scale(scrub_right, from_=0, to=100, orient="horizontal",
                                       command=self._on_scrub)
        self.scrub_slider.pack(fill="x", pady=(3, 2))
        self.lbl_timecode = ttk.Label(scrub_right, text="00:00 / 00:00", foreground="gray", font=("Helvetica", 9))
        self.lbl_timecode.pack(anchor="w")

        # Selected frames thumbnail strip
        self.thumb_label = ttk.Label(s1, text="Selected frames (click X to remove):", foreground="#333")
        # Not packed yet — shown after first frame added

        self.thumb_strip = ttk.Frame(s1)
        # Not packed yet

        # Step 2: Gemini buttons + Step 3: Response (combined to save space)
        s23 = ttk.LabelFrame(self, text="  Step 2: Upload frames to Gemini & paste response  ", padding=8)
        s23.pack(fill="both", expand=True, padx=10, pady=(3, 5))
        row2 = ttk.Frame(s23)
        row2.pack(fill="x")
        self.btn_folder = ttk.Button(row2, text="Open Frames Folder", command=self._open_folder, state="disabled")
        self.btn_folder.pack(side="left", padx=(0, 5))
        self.btn_gemini = ttk.Button(row2, text="Open Gemini", command=self._open_gemini, state="disabled")
        self.btn_gemini.pack(side="left", padx=(0, 5))
        self.btn_copy = ttk.Button(row2, text="Copy Prompt", command=self._copy_prompt, state="disabled")
        self.btn_copy.pack(side="left")

        self.txt = scrolledtext.ScrolledText(s23, height=3, font=("Consolas", 9), wrap="word")
        self.txt.pack(fill="both", expand=True, pady=(5, 3))
        ttk.Button(s23, text="Generate LUT", command=self._generate).pack(anchor="w")

    # ── File selection ──

    def _select_file(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Media files", "*.mp4 *.mov *.mkv *.avi *.webm *.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*"),
        ])
        if not path:
            return
        self.file_path = Path(path)
        self.lbl_file.config(text=self.file_path.name)
        self._clear_frames()

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        if self.file_path.suffix.lower() in image_exts:
            self.is_video = False
            self.scrub_frame.pack_forget()
            img = cv2.imread(str(self.file_path))
            if img is not None:
                h, w = img.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    img = cv2.resize(img, (1280, int(h * scale)))
                self._add_frame(0, img)
                self._save_frames_to_disk()
            self.btn_auto.config(state="disabled")
            self.btn_add_current.config(state="disabled")
        else:
            self.is_video = True
            cap = cv2.VideoCapture(str(self.file_path))
            self.video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            cap.release()
            self.scrub_slider.config(from_=0, to=max(1, self.video_total_frames - 1))
            self.scrub_slider.set(self.video_total_frames // 2)
            duration = self.video_total_frames / self.video_fps
            self.lbl_timecode.config(text=f"00:00 / {int(duration//60):02d}:{int(duration%60):02d}")
            self.scrub_frame.pack(fill="x", pady=(8, 0), before=self.thumb_label if self.thumb_label.winfo_manager() else None)
            self.btn_auto.config(state="normal")
            self.btn_add_current.config(state="normal")
            self._on_scrub(self.video_total_frames // 2)

    # ── Video scrubber ──

    def _on_scrub(self, val):
        if not self.is_video or not self.file_path:
            return
        frame_idx = int(float(val))
        secs = frame_idx / self.video_fps
        duration = self.video_total_frames / self.video_fps
        self.lbl_timecode.config(text=f"{int(secs//60):02d}:{int(secs%60):02d} / {int(duration//60):02d}:{int(duration%60):02d}")
        threading.Thread(target=self._load_scrub_frame, args=(frame_idx,), daemon=True).start()

    def _load_scrub_frame(self, frame_idx):
        cap = cv2.VideoCapture(str(self.file_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return
        h, w = frame.shape[:2]
        scale = min(160 / w, 100 / h)
        thumb = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        self._current_scrub_bgr = frame  # store full frame for "Add"
        self._current_scrub_idx = frame_idx
        pil = Image.fromarray(rgb)
        self.after(0, lambda p=pil: self._update_scrub_preview(p))

    def _update_scrub_preview(self, pil_img):
        self._scrub_photo = ImageTk.PhotoImage(pil_img)
        self.scrub_canvas.delete("all")
        self.scrub_canvas.create_image(80, 50, image=self._scrub_photo)

    # ── Frame management ──

    def _auto_extract(self):
        if not self.is_video:
            return
        self._clear_frames()
        self.lbl_count.config(text="Extracting...")
        threading.Thread(target=self._do_auto_extract, daemon=True).start()

    def _do_auto_extract(self):
        frames = extract_frames(self.file_path, NUM_FRAMES)
        total = self.video_total_frames
        indices = np.linspace(total * 0.1, total * 0.9, NUM_FRAMES, dtype=int)
        for i, (idx, frame) in enumerate(zip(indices, frames)):
            self.after(0, lambda ii=int(idx), ff=frame: self._add_frame(ii, ff))
        self.after(50 * len(frames), self._save_frames_to_disk)

    def _add_current_frame(self):
        if hasattr(self, '_current_scrub_bgr') and self._current_scrub_bgr is not None:
            frame = self._current_scrub_bgr
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (1280, int(h * scale)))
            self._add_frame(self._current_scrub_idx, frame)
            self._save_frames_to_disk()

    def _add_frame(self, frame_idx, bgr):
        self.selected_frames.append((frame_idx, bgr))
        self._refresh_thumbnails()

    def _remove_frame(self, index):
        if 0 <= index < len(self.selected_frames):
            self.selected_frames.pop(index)
            self._refresh_thumbnails()
            self._save_frames_to_disk()

    def _clear_frames(self):
        self.selected_frames = []
        self._thumb_photos = []
        self._refresh_thumbnails()
        self.btn_folder.config(state="disabled")
        self.btn_gemini.config(state="disabled")
        self.btn_copy.config(state="disabled")
        self.lbl_count.config(text="")
        self.btn_clear.config(state="disabled")

    def _refresh_thumbnails(self):
        for w in self.thumb_strip.winfo_children():
            w.destroy()
        self._thumb_photos = []

        count = len(self.selected_frames)
        self.lbl_count.config(text=f"{count} frame{'s' if count != 1 else ''} selected")

        if count > 0:
            self.thumb_label.pack(fill="x", pady=(8, 2))
            self.thumb_strip.pack(fill="x", pady=(0, 5))
            self.btn_clear.config(state="normal")
        else:
            self.thumb_label.pack_forget()
            self.thumb_strip.pack_forget()
            self.btn_clear.config(state="disabled")

        for i, (idx, bgr) in enumerate(self.selected_frames):
            card = ttk.Frame(self.thumb_strip)
            card.pack(side="left", padx=3)

            # Thumbnail
            h, w = bgr.shape[:2]
            scale = min(THUMB_W / w, THUMB_H / h)
            small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil)
            self._thumb_photos.append(photo)

            canvas = tk.Canvas(card, width=THUMB_W, height=THUMB_H, bg="#222", highlightthickness=1, highlightbackground="#555")
            canvas.pack()
            canvas.create_image(THUMB_W // 2, THUMB_H // 2, image=photo)

            # Timecode label
            if self.is_video and self.video_fps > 0:
                secs = idx / self.video_fps
                tc = f"{int(secs//60):02d}:{int(secs%60):02d}"
            else:
                tc = "Image"
            lbl_row = ttk.Frame(card)
            lbl_row.pack(fill="x")
            ttk.Label(lbl_row, text=tc, font=("Helvetica", 8), foreground="gray").pack(side="left")
            # Remove button
            btn_x = tk.Button(lbl_row, text="X", font=("Helvetica", 7, "bold"), fg="red",
                              relief="flat", cursor="hand2", padx=2, pady=0,
                              command=lambda ii=i: self._remove_frame(ii))
            btn_x.pack(side="right")

    def _save_frames_to_disk(self):
        if not self.file_path:
            return
        self.frames_dir = self.file_path.parent / f"{self.file_path.stem}_frames"
        self.frames_dir.mkdir(exist_ok=True)
        # Clear old frames
        for f in self.frames_dir.glob("frame_*.jpg"):
            f.unlink()
        for i, (idx, bgr) in enumerate(self.selected_frames):
            cv2.imwrite(str(self.frames_dir / f"frame_{i+1}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        # Enable Gemini buttons
        if len(self.selected_frames) > 0:
            self.btn_folder.config(state="normal")
            self.btn_gemini.config(state="normal")
            self.btn_copy.config(state="normal")

    # ── Gemini ──

    def _open_gemini(self):
        import webbrowser
        webbrowser.open("https://gemini.google.com/app")

    def _copy_prompt(self):
        self.winfo_toplevel().clipboard_clear()
        self.winfo_toplevel().clipboard_append(EXTRACT_PROMPT)
        self.btn_copy.config(text="Copied!")
        self.after(2000, lambda: self.btn_copy.config(text="Copy Prompt"))

    def _open_folder(self):
        if self.frames_dir:
            if sys.platform == "win32":
                os.startfile(str(self.frames_dir))
            elif sys.platform == "darwin":
                os.system(f'open "{self.frames_dir}"')

    def _generate(self):
        text = self.txt.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Empty", "Paste Gemini's JSON response first.")
            return
        try:
            params = parse_gemini_response(text)
            self.app.set_params(params, f"Extracted: {params['grade_name']}")
        except (json.JSONDecodeError, ValueError) as e:
            messagebox.showerror("Parse Error", str(e))


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: Tab 2 — Create from Text
# ═══════════════════════════════════════════════════════════════════

class TextCreateTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        s1 = ttk.LabelFrame(self, text="  Describe the look you want  ", padding=10)
        s1.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Label(s1, text='Examples: "warm vintage 70s film", "Blade Runner 2049", "Kodak Portra with faded blacks"',
                  foreground="gray").pack(anchor="w")
        self.txt_desc = tk.Text(s1, height=3, font=("Helvetica", 11), wrap="word")
        self.txt_desc.pack(fill="x", pady=(5, 5))
        ttk.Button(s1, text="Generate Prompt", command=self._gen_prompt).pack(anchor="w")

        s2 = ttk.LabelFrame(self, text="  Copy this prompt to Gemini  ", padding=10)
        s2.pack(fill="x", padx=10, pady=5)
        self.txt_prompt = scrolledtext.ScrolledText(s2, height=5, font=("Consolas", 9), wrap="word", state="disabled")
        self.txt_prompt.pack(fill="x", pady=(0, 5))
        row = ttk.Frame(s2)
        row.pack(fill="x")
        self.btn_copy = ttk.Button(row, text="Copy to Clipboard", command=self._copy, state="disabled")
        self.btn_copy.pack(side="left", padx=(0, 5))
        ttk.Button(row, text="Open Gemini", command=lambda: __import__("webbrowser").open("https://gemini.google.com/app")).pack(side="left")

        s3 = ttk.LabelFrame(self, text="  Paste Gemini's Response  ", padding=10)
        s3.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        self.txt_resp = scrolledtext.ScrolledText(s3, height=6, font=("Consolas", 9), wrap="word")
        self.txt_resp.pack(fill="both", expand=True, pady=(0, 5))
        ttk.Button(s3, text="Generate LUT", command=self._generate).pack(anchor="w")

    def _gen_prompt(self):
        desc = self.txt_desc.get("1.0", "end").strip()
        if not desc:
            messagebox.showwarning("Empty", "Enter a description first.")
            return
        prompt = TEXT_CREATE_PROMPT.format(description=desc)
        self.txt_prompt.config(state="normal")
        self.txt_prompt.delete("1.0", "end")
        self.txt_prompt.insert("1.0", prompt)
        self.txt_prompt.config(state="disabled")
        self.btn_copy.config(state="normal")

    def _copy(self):
        self.winfo_toplevel().clipboard_clear()
        self.winfo_toplevel().clipboard_append(self.txt_prompt.get("1.0", "end"))
        self.btn_copy.config(text="Copied!")
        self.after(2000, lambda: self.btn_copy.config(text="Copy to Clipboard"))

    def _generate(self):
        text = self.txt_resp.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Empty", "Paste Gemini's JSON response first.")
            return
        try:
            params = parse_gemini_response(text)
            self.app.set_params(params, f"Created: {params['grade_name']}")
        except (json.JSONDecodeError, ValueError) as e:
            messagebox.showerror("Parse Error", str(e))


# ═══════════════════════════════════════════════════════════════════
# SECTION 9: Tab 3 — Preset Library
# ═══════════════════════════════════════════════════════════════════

class PresetLibraryTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.selected_preset = None
        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 5))

        # Category selector
        self.category_var = tk.StringVar(value=list(PRESETS.keys())[0])
        for cat in PRESETS:
            ttk.Radiobutton(top, text=cat, variable=self.category_var, value=cat,
                            command=self._refresh_grid).pack(side="left", padx=(0, 15))

        # Grid frame with scrollbar
        grid_frame = ttk.LabelFrame(self, text="  Select a Preset  ", padding=10)
        grid_frame.pack(fill="both", expand=True, padx=10, pady=5)

        canvas = tk.Canvas(grid_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(grid_frame, orient="vertical", command=canvas.yview)
        self.grid_inner = ttk.Frame(canvas)
        self.grid_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.grid_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Description + buttons
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=(0, 10))
        self.lbl_desc = ttk.Label(bottom, text="Click a preset to select it.", wraplength=700, foreground="gray")
        self.lbl_desc.pack(anchor="w", pady=(5, 8))
        btn_row = ttk.Frame(bottom)
        btn_row.pack(fill="x")
        self.btn_apply = ttk.Button(btn_row, text="Apply Preset", command=self._apply, state="disabled")
        self.btn_apply.pack(side="left", padx=(0, 8))
        self.btn_edit = ttk.Button(btn_row, text="Apply & Edit", command=self._apply_edit, state="disabled")
        self.btn_edit.pack(side="left")

        self._refresh_grid()

    def _refresh_grid(self):
        for w in self.grid_inner.winfo_children():
            w.destroy()
        cat = self.category_var.get()
        presets = PRESETS.get(cat, {})
        col = 0
        row = 0
        for name, data in presets.items():
            btn = tk.Button(
                self.grid_inner, text=name, width=22, height=2,
                font=("Helvetica", 10), relief="groove", cursor="hand2",
                command=lambda n=name, d=data: self._select(n, d),
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            col += 1
            if col >= 3:
                col = 0
                row += 1

    def _select(self, name, data):
        self.selected_preset = data["params"]
        self.lbl_desc.config(text=f"{name}: {data['desc']}", foreground="#333")
        self.btn_apply.config(state="normal")
        self.btn_edit.config(state="normal")

    def _apply(self):
        if self.selected_preset:
            self.app.set_params(copy.deepcopy(self.selected_preset),
                                f"Preset: {self.selected_preset['grade_name']}")

    def _apply_edit(self):
        self._apply()
        self.app.notebook.select(3)  # Switch to Editor tab


# ═══════════════════════════════════════════════════════════════════
# SECTION 10: Tab 4 — Editor / Tweaker
# ═══════════════════════════════════════════════════════════════════

class EditorTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.sliders = {}
        self.labels = {}
        self._preview_timer = None
        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 5))
        self.lbl_source = ttk.Label(top, text="Source: None", font=("Helvetica", 10, "bold"))
        self.lbl_source.pack(side="left")
        ttk.Button(top, text="Reset to Default", command=self._reset).pack(side="right")

        # Scrollable area for sliders
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.slider_frame = ttk.Frame(canvas)
        self.slider_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.slider_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0))
        scrollbar.pack(side="right", fill="y")

        # Build sliders by section
        current_section = None
        row = 0
        for param, cfg in SLIDER_CONFIG.items():
            if cfg["section"] != current_section:
                current_section = cfg["section"]
                lf = ttk.LabelFrame(self.slider_frame, text=f"  {current_section}  ", padding=5)
                lf.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=(8, 2))
                row += 1

            label_name = param.replace("_", " ").title()
            ttk.Label(self.slider_frame, text=label_name, width=18, anchor="e").grid(
                row=row, column=0, padx=(5, 2), pady=2, sticky="e")

            slider = ttk.Scale(
                self.slider_frame, from_=cfg["from"], to=cfg["to"],
                orient="horizontal", length=350,
                command=lambda val, p=param: self._on_change(p, val),
            )
            slider.set(DEFAULT_PARAMS[param])
            slider.grid(row=row, column=1, padx=5, pady=2, sticky="ew")
            slider.bind("<ButtonRelease-1>", lambda e: self._on_release())
            self.sliders[param] = slider

            val_label = ttk.Label(self.slider_frame, text=f"{DEFAULT_PARAMS[param]:.3f}", width=8)
            val_label.grid(row=row, column=2, padx=(2, 10), pady=2)
            self.labels[param] = val_label
            row += 1

        self.slider_frame.columnconfigure(1, weight=1)

        # Grade name
        name_frame = ttk.Frame(self.slider_frame)
        name_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=(10, 5))
        ttk.Label(name_frame, text="Grade Name:").pack(side="left", padx=(0, 5))
        self.name_entry = ttk.Entry(name_frame, width=40)
        self.name_entry.insert(0, "Untitled")
        self.name_entry.pack(side="left", fill="x", expand=True)

    def set_params(self, params):
        for param, slider in self.sliders.items():
            if param in params:
                slider.set(params[param])
                self.labels[param].config(text=f"{params[param]:.3f}")
        if "grade_name" in params:
            self.name_entry.delete(0, "end")
            self.name_entry.insert(0, params["grade_name"])

    def get_params(self):
        params = {}
        for param, slider in self.sliders.items():
            params[param] = float(slider.get())
        params["grade_name"] = self.name_entry.get() or "Untitled"
        return params

    def _on_change(self, param, val):
        self.labels[param].config(text=f"{float(val):.3f}")
        if self._preview_timer:
            self.after_cancel(self._preview_timer)
        self._preview_timer = self.after(DEBOUNCE_MS, self._update_preview)

    def _on_release(self):
        if self._preview_timer:
            self.after_cancel(self._preview_timer)
            self._preview_timer = None
        self._update_preview()

    def _update_preview(self):
        self._preview_timer = None
        params = self.get_params()
        self.app.update_from_editor(params)

    def _reset(self):
        self.set_params(DEFAULT_PARAMS)
        self.app.update_from_editor(DEFAULT_PARAMS)


# ═══════════════════════════════════════════════════════════════════
# SECTION 11: Bottom Panel — Preview + Export
# ═══════════════════════════════════════════════════════════════════

class BottomPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._before_photo = None
        self._after_photo = None
        self._intensity_timer = None
        self._build()

    def _build(self):
        # Preview canvases
        preview_frame = ttk.LabelFrame(self, text="  Preview  ", padding=5)
        preview_frame.pack(fill="x", padx=10, pady=(5, 3))

        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack()

        ttk.Label(canvas_frame, text="BEFORE", font=("Helvetica", 9, "bold")).grid(row=0, column=0)
        ttk.Label(canvas_frame, text="AFTER", font=("Helvetica", 9, "bold")).grid(row=0, column=2)

        self.canvas_before = tk.Canvas(canvas_frame, width=PREVIEW_W, height=PREVIEW_H,
                                        bg="#1a1a1a", highlightthickness=1, highlightbackground="#444")
        self.canvas_before.grid(row=1, column=0, padx=(0, 3))

        ttk.Separator(canvas_frame, orient="vertical").grid(row=1, column=1, sticky="ns", padx=3)

        self.canvas_after = tk.Canvas(canvas_frame, width=PREVIEW_W, height=PREVIEW_H,
                                       bg="#1a1a1a", highlightthickness=1, highlightbackground="#444")
        self.canvas_after.grid(row=1, column=2, padx=(3, 0))

        # Controls row
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", padx=10, pady=(3, 5))

        ttk.Label(ctrl, text="Intensity:").pack(side="left", padx=(0, 5))
        self.intensity_var = tk.DoubleVar(value=100.0)
        self.intensity_slider = ttk.Scale(ctrl, from_=0, to=100, orient="horizontal",
                                           variable=self.intensity_var, length=150,
                                           command=self._on_intensity)
        self.intensity_slider.pack(side="left", padx=(0, 3))
        self.lbl_intensity = ttk.Label(ctrl, text="100%", width=5)
        self.lbl_intensity.pack(side="left", padx=(0, 15))

        ttk.Button(ctrl, text="Load Image", command=self._load_image).pack(side="left", padx=(0, 8))
        ttk.Button(ctrl, text="Export .cube", command=self._export).pack(side="left", padx=(0, 8))

        self.lbl_status = ttk.Label(ctrl, text="Ready", foreground="gray")
        self.lbl_status.pack(side="right")

    def update_preview(self, lut_3d):
        engine = self.app.engine
        preview = self.app.preview_engine
        intensity = self.intensity_var.get() / 100.0
        if intensity < 1.0:
            lut_3d = engine.interpolate_intensity(lut_3d, intensity)
        self._before_photo = preview.get_before_photo()
        self._after_photo = preview.get_after_photo(engine, lut_3d)
        self.canvas_before.delete("all")
        self.canvas_after.delete("all")
        self.canvas_before.create_image(PREVIEW_W // 2, PREVIEW_H // 2, image=self._before_photo)
        self.canvas_after.create_image(PREVIEW_W // 2, PREVIEW_H // 2, image=self._after_photo)
        if not self.app.preview_engine.has_image:
            self.canvas_before.create_text(PREVIEW_W // 2, PREVIEW_H // 2, text="Load an image\nto preview", fill="#888", font=("Helvetica", 11), justify="center")
            self.canvas_after.create_text(PREVIEW_W // 2, PREVIEW_H // 2, text="Load an image\nto preview", fill="#888", font=("Helvetica", 11), justify="center")

    def _on_intensity(self, val):
        self.lbl_intensity.config(text=f"{int(float(val))}%")
        if self._intensity_timer:
            self.after_cancel(self._intensity_timer)
        self._intensity_timer = self.after(INTENSITY_DEBOUNCE_MS, self._intensity_update)

    def _intensity_update(self):
        self._intensity_timer = None
        if self.app.current_lut is not None:
            self.update_preview(self.app.current_lut)

    def _load_image(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All", "*.*"),
        ])
        if path and self.app.preview_engine.load_image(path):
            self.lbl_status.config(text=f"Loaded: {Path(path).name}", foreground="#228B22")
            if self.app.current_lut is not None:
                self.update_preview(self.app.current_lut)

    def _export(self):
        if self.app.current_lut is None:
            messagebox.showinfo("No LUT", "Generate or select a LUT first.")
            return
        name = self.app.current_params.get("grade_name", "Untitled")
        safe = name.replace(" ", "_").replace("/", "-").replace("&", "and")[:40]
        path = filedialog.asksaveasfilename(
            defaultextension=".cube",
            initialfile=f"{safe}.cube",
            filetypes=[("Cube LUT", "*.cube"), ("All", "*.*")],
        )
        if path:
            intensity = self.intensity_var.get() / 100.0
            lut = self.app.current_lut
            if intensity < 1.0:
                lut = self.app.engine.interpolate_intensity(lut, intensity)
            self.app.engine.export_cube(lut, path, title=name)
            self.lbl_status.config(text=f"Exported: {Path(path).name}", foreground="#228B22")


# ═══════════════════════════════════════════════════════════════════
# SECTION 12: Main Application
# ═══════════════════════════════════════════════════════════════════

class LUTExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_TITLE} v{APP_VERSION}")
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(*MIN_SIZE)

        self.engine = LUTEngine()
        self.preview_engine = PreviewEngine()
        self.current_params = copy.deepcopy(DEFAULT_PARAMS)
        self.current_lut = None

        self._build()

    def _build(self):
        # Header
        header = tk.Frame(self.root, bg="#1a1a2e", height=50)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text=APP_TITLE, font=("Helvetica", 16, "bold"),
                 bg="#1a1a2e", fg="white").pack(side="left", padx=15, pady=8)
        tk.Label(header, text=f"v{APP_VERSION}  |  Extract, create & customize cinematic LUTs",
                 font=("Helvetica", 9), bg="#1a1a2e", fg="#888").pack(side="left", pady=8)

        # Main area: tabs + bottom panel
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        # Tabs
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=(5, 0))

        self.tab_extract = ExtractTab(self.notebook, self)
        self.tab_text = TextCreateTab(self.notebook, self)
        self.tab_presets = PresetLibraryTab(self.notebook, self)
        self.tab_editor = EditorTab(self.notebook, self)

        self.notebook.add(self.tab_extract, text="  Extract from Media  ")
        self.notebook.add(self.tab_text, text="  Create from Text  ")
        self.notebook.add(self.tab_presets, text="  Preset Library  ")
        self.notebook.add(self.tab_editor, text="  Editor  ")

        # Bottom panel
        self.bottom = BottomPanel(main, self)
        self.bottom.pack(fill="x", side="bottom")

        # Initial preview
        self.root.after(100, self._initial_preview)

    def _initial_preview(self):
        self.current_lut = self.engine.generate_lut(DEFAULT_PARAMS)
        self.bottom.update_preview(self.current_lut)

    def set_params(self, params, source_name=""):
        self.current_params = copy.deepcopy(params)
        self.current_lut = self.engine.generate_lut(params)
        self.tab_editor.set_params(params)
        self.tab_editor.lbl_source.config(text=f"Source: {source_name}")
        self.bottom.update_preview(self.current_lut)
        self.bottom.lbl_status.config(text=f"LUT ready: {params.get('grade_name', '?')}", foreground="#228B22")

    def update_from_editor(self, params):
        self.current_params = copy.deepcopy(params)
        self.current_lut = self.engine.generate_lut(params)
        self.bottom.update_preview(self.current_lut)


# ═══════════════════════════════════════════════════════════════════
# SECTION 13: Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TLabelframe.Label", font=("Helvetica", 10, "bold"))
    app = LUTExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
