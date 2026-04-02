"""
LUT Extractor Pro — Extract, create, and customize cinematic LUTs.
Single-file GUI app. Package with: pyinstaller --onefile --windowed lut_extractor_pro.py

V5 Engine: per-channel R/G/B tone curves (15) + dual shadow/highlight 3x3 color matrices (18) + saturation (1) = 34 parameters.
Shadow and highlight matrices are blended by luminance (smoothstep), capturing different color mixing in darks vs brights.

Features:
  Tab 1: Extract from Media — analyze video/image frames via Gemini AI
  Tab 2: Create from Text — describe a look, get a LUT
  Tab 3: Preset Library — 22 built-in cinematic presets (no AI needed)
  Tab 4: Editor — 34 sliders with real-time before/after preview
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
APP_VERSION = "3.0"  # V5 engine
WINDOW_SIZE = "1100x800"
MIN_SIZE = (950, 700)
PREVIEW_W, PREVIEW_H = 200, 130
LUT_SIZE = 33
NUM_FRAMES = 5
DEBOUNCE_MS = 80
INTENSITY_DEBOUNCE_MS = 30

DEFAULT_PARAMS = {
    # Per-channel tone curves (5 control points each, mapping input → output brightness)
    "r_blacks": 0.00, "r_shadows": 0.25, "r_mid": 0.50, "r_highlights": 0.75, "r_whites": 1.00,
    "g_blacks": 0.00, "g_shadows": 0.25, "g_mid": 0.50, "g_highlights": 0.75, "g_whites": 1.00,
    "b_blacks": 0.00, "b_shadows": 0.25, "b_mid": 0.50, "b_highlights": 0.75, "b_whites": 1.00,
    # Shadow color matrix (applied to dark pixels, blended by luminance)
    "smat_rr": 1.0, "smat_rg": 0.0, "smat_rb": 0.0,
    "smat_gr": 0.0, "smat_gg": 1.0, "smat_gb": 0.0,
    "smat_br": 0.0, "smat_bg": 0.0, "smat_bb": 1.0,
    # Highlight color matrix (applied to bright pixels, blended by luminance)
    "hmat_rr": 1.0, "hmat_rg": 0.0, "hmat_rb": 0.0,
    "hmat_gr": 0.0, "hmat_gg": 1.0, "hmat_gb": 0.0,
    "hmat_br": 0.0, "hmat_bg": 0.0, "hmat_bb": 1.0,
    # Global
    "saturation": 1.0,
    "grade_name": "Untitled",
}

SLIDER_CONFIG = {
    # Red channel curve
    "r_blacks":     {"from": 0.00, "to": 0.20, "res": 0.005, "section": "Red Curve"},
    "r_shadows":    {"from": 0.05, "to": 0.45, "res": 0.005, "section": "Red Curve"},
    "r_mid":        {"from": 0.20, "to": 0.80, "res": 0.005, "section": "Red Curve"},
    "r_highlights": {"from": 0.55, "to": 0.95, "res": 0.005, "section": "Red Curve"},
    "r_whites":     {"from": 0.80, "to": 1.00, "res": 0.005, "section": "Red Curve"},
    # Green channel curve
    "g_blacks":     {"from": 0.00, "to": 0.20, "res": 0.005, "section": "Green Curve"},
    "g_shadows":    {"from": 0.05, "to": 0.45, "res": 0.005, "section": "Green Curve"},
    "g_mid":        {"from": 0.20, "to": 0.80, "res": 0.005, "section": "Green Curve"},
    "g_highlights": {"from": 0.55, "to": 0.95, "res": 0.005, "section": "Green Curve"},
    "g_whites":     {"from": 0.80, "to": 1.00, "res": 0.005, "section": "Green Curve"},
    # Blue channel curve
    "b_blacks":     {"from": 0.00, "to": 0.20, "res": 0.005, "section": "Blue Curve"},
    "b_shadows":    {"from": 0.05, "to": 0.45, "res": 0.005, "section": "Blue Curve"},
    "b_mid":        {"from": 0.20, "to": 0.80, "res": 0.005, "section": "Blue Curve"},
    "b_highlights": {"from": 0.55, "to": 0.95, "res": 0.005, "section": "Blue Curve"},
    "b_whites":     {"from": 0.80, "to": 1.00, "res": 0.005, "section": "Blue Curve"},
    # Shadow matrix (color mixing in dark areas)
    "smat_rr": {"from": -0.50, "to": 2.00, "res": 0.01, "section": "Shadow Matrix — Red Out"},
    "smat_rg": {"from": -1.00, "to": 1.00, "res": 0.01, "section": "Shadow Matrix — Red Out"},
    "smat_rb": {"from": -1.00, "to": 1.50, "res": 0.01, "section": "Shadow Matrix — Red Out"},
    "smat_gr": {"from": -1.00, "to": 1.00, "res": 0.01, "section": "Shadow Matrix — Green Out"},
    "smat_gg": {"from": -0.50, "to": 2.00, "res": 0.01, "section": "Shadow Matrix — Green Out"},
    "smat_gb": {"from": -1.00, "to": 1.00, "res": 0.01, "section": "Shadow Matrix — Green Out"},
    "smat_br": {"from": -1.00, "to": 1.50, "res": 0.01, "section": "Shadow Matrix — Blue Out"},
    "smat_bg": {"from": -1.00, "to": 1.00, "res": 0.01, "section": "Shadow Matrix — Blue Out"},
    "smat_bb": {"from": -0.50, "to": 2.00, "res": 0.01, "section": "Shadow Matrix — Blue Out"},
    # Highlight matrix (color mixing in bright areas)
    "hmat_rr": {"from": -0.50, "to": 2.00, "res": 0.01, "section": "Highlight Matrix — Red Out"},
    "hmat_rg": {"from": -1.00, "to": 1.00, "res": 0.01, "section": "Highlight Matrix — Red Out"},
    "hmat_rb": {"from": -1.00, "to": 1.50, "res": 0.01, "section": "Highlight Matrix — Red Out"},
    "hmat_gr": {"from": -1.00, "to": 1.00, "res": 0.01, "section": "Highlight Matrix — Green Out"},
    "hmat_gg": {"from": -0.50, "to": 2.00, "res": 0.01, "section": "Highlight Matrix — Green Out"},
    "hmat_gb": {"from": -1.00, "to": 1.00, "res": 0.01, "section": "Highlight Matrix — Green Out"},
    "hmat_br": {"from": -1.00, "to": 1.50, "res": 0.01, "section": "Highlight Matrix — Blue Out"},
    "hmat_bg": {"from": -1.00, "to": 1.00, "res": 0.01, "section": "Highlight Matrix — Blue Out"},
    "hmat_bb": {"from": -0.50, "to": 2.00, "res": 0.01, "section": "Highlight Matrix — Blue Out"},
    # Global
    "saturation": {"from": 0.00, "to": 2.00, "res": 0.01, "section": "Global"},
}

# ═══════════════════════════════════════════════════════════════════
# SECTION 2: Preset Library (22 presets)
# ═══════════════════════════════════════════════════════════════════

PRESETS = {
    "Cinematic": {
        "Teal & Orange": {
            "desc": "Blockbuster look with warm highlights and cool teal shadows. Think Transformers, Mad Max.",
            "params": {
                "r_blacks": 0.017, "r_shadows": 0.150, "r_mid": 0.550, "r_highlights": 0.931, "r_whites": 1.000,
                "g_blacks": 0.037, "g_shadows": 0.207, "g_mid": 0.520, "g_highlights": 0.839, "g_whites": 1.000,
                "b_blacks": 0.074, "b_shadows": 0.202, "b_mid": 0.451, "b_highlights": 0.720, "b_whites": 0.881,
                "smat_rr": 1.091, "smat_rg": -0.112, "smat_rb": -0.015,
                "smat_gr": -0.057, "smat_gg": 1.066, "smat_gb": -0.028,
                "smat_br": -0.058, "smat_bg": -0.186, "smat_bb": 1.244,
                "hmat_rr": 1.104, "hmat_rg": -0.101, "hmat_rb": -0.011,
                "hmat_gr": -0.035, "hmat_gg": 1.059, "hmat_gb": -0.011,
                "hmat_br": -0.040, "hmat_bg": -0.150, "hmat_bb": 1.235,
                "saturation": 1.20, "grade_name": "Teal & Orange",
            },
        },
        "Bleach Bypass": {
            "desc": "High contrast, desaturated. Saving Private Ryan, Se7en look.",
            "params": {
                "r_blacks": 0.000, "r_shadows": 0.051, "r_mid": 0.479, "r_highlights": 0.969, "r_whites": 1.000,
                "g_blacks": 0.000, "g_shadows": 0.051, "g_mid": 0.479, "g_highlights": 0.969, "g_whites": 1.000,
                "b_blacks": 0.000, "b_shadows": 0.051, "b_mid": 0.479, "b_highlights": 0.969, "b_whites": 1.000,
                "smat_rr": 0.570, "smat_rg": 0.381, "smat_rb": 0.031,
                "smat_gr": 0.108, "smat_gg": 0.849, "smat_gb": 0.031,
                "smat_br": 0.109, "smat_bg": 0.381, "smat_bb": 0.493,
                "hmat_rr": 0.580, "hmat_rg": 0.389, "hmat_rb": 0.041,
                "hmat_gr": 0.120, "hmat_gg": 0.849, "hmat_gb": 0.041,
                "hmat_br": 0.119, "hmat_bg": 0.389, "hmat_bb": 0.503,
                "saturation": 0.45, "grade_name": "Bleach Bypass",
            },
        },
        "Vintage Film": {
            "desc": "Warm, faded look with lifted blacks and muted colors. Indie film feel.",
            "params": {
                "r_blacks": 0.177, "r_shadows": 0.342, "r_mid": 0.578, "r_highlights": 0.810, "r_whites": 0.970,
                "g_blacks": 0.132, "g_shadows": 0.297, "g_mid": 0.533, "g_highlights": 0.768, "g_whites": 0.930,
                "b_blacks": 0.063, "b_shadows": 0.219, "b_mid": 0.451, "b_highlights": 0.687, "b_whites": 0.853,
                "smat_rr": 0.848, "smat_rg": 0.160, "smat_rb": 0.018,
                "smat_gr": 0.042, "smat_gg": 0.941, "smat_gb": 0.014,
                "smat_br": 0.030, "smat_bg": 0.110, "smat_bb": 0.807,
                "hmat_rr": 0.843, "hmat_rg": 0.152, "hmat_rb": 0.014,
                "hmat_gr": 0.042, "hmat_gg": 0.942, "hmat_gb": 0.014,
                "hmat_br": 0.040, "hmat_bg": 0.126, "hmat_bb": 0.815,
                "saturation": 0.80, "grade_name": "Vintage Film",
            },
        },
        "Moody Desaturated": {
            "desc": "Dark, muted look with cool undertones. David Fincher style.",
            "params": {
                "r_blacks": 0.028, "r_shadows": 0.207, "r_mid": 0.449, "r_highlights": 0.735, "r_whites": 0.961,
                "g_blacks": 0.034, "g_shadows": 0.213, "g_mid": 0.457, "g_highlights": 0.745, "g_whites": 0.972,
                "b_blacks": 0.054, "b_shadows": 0.237, "b_mid": 0.482, "b_highlights": 0.766, "b_whites": 0.987,
                "smat_rr": 0.642, "smat_rg": 0.316, "smat_rb": 0.031,
                "smat_gr": 0.095, "smat_gg": 0.871, "smat_gb": 0.032,
                "smat_br": 0.104, "smat_bg": 0.342, "smat_bb": 0.588,
                "hmat_rr": 0.644, "hmat_rg": 0.319, "hmat_rb": 0.033,
                "hmat_gr": 0.095, "hmat_gg": 0.873, "hmat_gb": 0.032,
                "hmat_br": 0.097, "hmat_bg": 0.334, "hmat_bb": 0.580,
                "saturation": 0.55, "grade_name": "Moody Desaturated",
            },
        },
        "Golden Hour": {
            "desc": "Warm, golden sunlight. Rich amber tones. Terrence Malick vibes.",
            "params": {
                "r_blacks": 0.143, "r_shadows": 0.395, "r_mid": 0.694, "r_highlights": 0.960, "r_whites": 1.000,
                "g_blacks": 0.026, "g_shadows": 0.264, "g_mid": 0.563, "g_highlights": 0.838, "g_whites": 1.000,
                "b_blacks": 0.000, "b_shadows": 0.063, "b_mid": 0.342, "b_highlights": 0.629, "b_whites": 0.867,
                "smat_rr": 1.057, "smat_rg": -0.108, "smat_rb": -0.004,
                "smat_gr": -0.030, "smat_gg": 1.038, "smat_gb": -0.009,
                "smat_br": -0.017, "smat_bg": -0.046, "smat_bb": 1.164,
                "hmat_rr": 1.115, "hmat_rg": -0.120, "hmat_rb": -0.015,
                "hmat_gr": -0.032, "hmat_gg": 1.054, "hmat_gb": -0.012,
                "hmat_br": -0.015, "hmat_bg": -0.049, "hmat_bb": 1.083,
                "saturation": 1.15, "grade_name": "Golden Hour",
            },
        },
        "Noir": {
            "desc": "Dark, contrasty, near-monochrome. Classic film noir mood.",
            "params": {
                "r_blacks": 0.002, "r_shadows": 0.034, "r_mid": 0.382, "r_highlights": 0.817, "r_whites": 0.997,
                "g_blacks": 0.002, "g_shadows": 0.036, "g_mid": 0.384, "g_highlights": 0.818, "g_whites": 0.998,
                "b_blacks": 0.006, "b_shadows": 0.040, "b_mid": 0.387, "b_highlights": 0.820, "b_whites": 0.998,
                "smat_rr": 0.335, "smat_rg": 0.594, "smat_rb": 0.055,
                "smat_gr": 0.179, "smat_gg": 0.747, "smat_gb": 0.056,
                "smat_br": 0.182, "smat_bg": 0.602, "smat_bb": 0.212,
                "hmat_rr": 0.339, "hmat_rg": 0.630, "hmat_rb": 0.061,
                "hmat_gr": 0.181, "hmat_gg": 0.790, "hmat_gb": 0.060,
                "hmat_br": 0.182, "hmat_bg": 0.635, "hmat_bb": 0.215,
                "saturation": 0.15, "grade_name": "Noir",
            },
        },
        "Pastel Faded": {
            "desc": "Light, airy, lifted everything. Instagram/Pinterest aesthetic.",
            "params": {
                "r_blacks": 0.205, "r_shadows": 0.364, "r_mid": 0.554, "r_highlights": 0.726, "r_whites": 0.867,
                "g_blacks": 0.191, "g_shadows": 0.350, "g_mid": 0.539, "g_highlights": 0.712, "g_whites": 0.854,
                "b_blacks": 0.200, "b_shadows": 0.358, "b_mid": 0.546, "b_highlights": 0.717, "b_whites": 0.856,
                "smat_rr": 0.726, "smat_rg": 0.258, "smat_rb": 0.027,
                "smat_gr": 0.073, "smat_gg": 0.899, "smat_gb": 0.024,
                "smat_br": 0.076, "smat_bg": 0.256, "smat_bb": 0.673,
                "hmat_rr": 0.724, "hmat_rg": 0.255, "hmat_rb": 0.025,
                "hmat_gr": 0.074, "hmat_gg": 0.899, "hmat_gb": 0.025,
                "hmat_br": 0.075, "hmat_bg": 0.254, "hmat_bb": 0.671,
                "saturation": 0.65, "grade_name": "Pastel Faded",
            },
        },
        "High Contrast Drama": {
            "desc": "Punchy, dramatic, deep blacks and bright whites.",
            "params": {
                "r_blacks": 0.011, "r_shadows": 0.011, "r_mid": 0.523, "r_highlights": 1.000, "r_whites": 1.000,
                "g_blacks": 0.000, "g_shadows": 0.000, "g_mid": 0.505, "g_highlights": 1.000, "g_whites": 1.000,
                "b_blacks": 0.000, "b_shadows": 0.000, "b_mid": 0.481, "b_highlights": 0.989, "b_whites": 0.989,
                "smat_rr": 1.043, "smat_rg": -0.050, "smat_rb": -0.010,
                "smat_gr": -0.018, "smat_gg": 0.987, "smat_gb": -0.009,
                "smat_br": -0.011, "smat_bg": -0.033, "smat_bb": 1.048,
                "hmat_rr": 1.052, "hmat_rg": -0.041, "hmat_rb": -0.004,
                "hmat_gr": -0.007, "hmat_gg": 1.022, "hmat_gb": 0.001,
                "hmat_br": -0.011, "hmat_bg": -0.032, "hmat_bb": 1.048,
                "saturation": 1.10, "grade_name": "High Contrast Drama",
            },
        },
    },
    "Film Stocks": {
        "Kodak Vision3 500T": {
            "desc": "Tungsten motion picture stock. Warm highlights, blue shadows under daylight.",
            "params": {
                "r_blacks": 0.027, "r_shadows": 0.233, "r_mid": 0.562, "r_highlights": 0.883, "r_whites": 1.000,
                "g_blacks": 0.003, "g_shadows": 0.206, "g_mid": 0.518, "g_highlights": 0.829, "g_whites": 1.000,
                "b_blacks": 0.025, "b_shadows": 0.191, "b_mid": 0.462, "b_highlights": 0.744, "b_whites": 0.934,
                "smat_rr": 1.052, "smat_rg": -0.073, "smat_rb": -0.014,
                "smat_gr": -0.018, "smat_gg": 1.013, "smat_gb": -0.010,
                "smat_br": -0.020, "smat_bg": -0.058, "smat_bb": 1.069,
                "hmat_rr": 1.075, "hmat_rg": -0.056, "hmat_rb": -0.004,
                "hmat_gr": -0.014, "hmat_gg": 1.034, "hmat_gb": -0.005,
                "hmat_br": -0.012, "hmat_bg": -0.046, "hmat_bb": 1.073,
                "saturation": 1.05, "grade_name": "Kodak Vision3 500T",
            },
        },
        "Kodak Portra 400": {
            "desc": "Portrait film. Low contrast, warm skin tones, muted greens.",
            "params": {
                "r_blacks": 0.139, "r_shadows": 0.337, "r_mid": 0.577, "r_highlights": 0.806, "r_whites": 0.988,
                "g_blacks": 0.092, "g_shadows": 0.286, "g_mid": 0.528, "g_highlights": 0.761, "g_whites": 0.948,
                "b_blacks": 0.045, "b_shadows": 0.230, "b_mid": 0.467, "b_highlights": 0.700, "b_whites": 0.888,
                "smat_rr": 0.909, "smat_rg": 0.096, "smat_rb": 0.011,
                "smat_gr": 0.025, "smat_gg": 0.963, "smat_gb": 0.009,
                "smat_br": 0.020, "smat_bg": 0.071, "smat_bb": 0.884,
                "hmat_rr": 0.905, "hmat_rg": 0.092, "hmat_rb": 0.009,
                "hmat_gr": 0.025, "hmat_gg": 0.966, "hmat_gb": 0.008,
                "hmat_br": 0.024, "hmat_bg": 0.078, "hmat_bb": 0.888,
                "saturation": 0.88, "grade_name": "Kodak Portra 400",
            },
        },
        "Fuji Velvia 50": {
            "desc": "Slide film. Extremely saturated, deep blacks, vivid greens and blues.",
            "params": {
                "r_blacks": 0.000, "r_shadows": 0.068, "r_mid": 0.469, "r_highlights": 0.884, "r_whites": 0.985,
                "g_blacks": 0.023, "g_shadows": 0.145, "g_mid": 0.541, "g_highlights": 0.923, "g_whites": 0.998,
                "b_blacks": 0.021, "b_shadows": 0.136, "b_mid": 0.526, "b_highlights": 0.909, "b_whites": 1.000,
                "smat_rr": 1.157, "smat_rg": -0.069, "smat_rb": 0.006,
                "smat_gr": -0.087, "smat_gg": 1.107, "smat_gb": -0.044,
                "smat_br": -0.035, "smat_bg": -0.129, "smat_bb": 1.161,
                "hmat_rr": 1.123, "hmat_rg": -0.111, "hmat_rb": -0.018,
                "hmat_gr": -0.052, "hmat_gg": 1.072, "hmat_gb": -0.016,
                "hmat_br": -0.045, "hmat_bg": -0.143, "hmat_bb": 1.165,
                "saturation": 1.45, "grade_name": "Fuji Velvia 50",
            },
        },
        "CineStill 800T": {
            "desc": "Tungsten cinema stock. Halation glow, amber cast, lifted shadows.",
            "params": {
                "r_blacks": 0.171, "r_shadows": 0.384, "r_mid": 0.659, "r_highlights": 0.926, "r_whites": 1.000,
                "g_blacks": 0.033, "g_shadows": 0.231, "g_mid": 0.510, "g_highlights": 0.792, "g_whites": 1.000,
                "b_blacks": 0.000, "b_shadows": 0.110, "b_mid": 0.373, "b_highlights": 0.651, "b_whites": 0.876,
                "smat_rr": 1.052, "smat_rg": -0.086, "smat_rb": -0.006,
                "smat_gr": -0.020, "smat_gg": 1.031, "smat_gb": -0.006,
                "smat_br": -0.021, "smat_bg": -0.056, "smat_bb": 1.113,
                "hmat_rr": 1.092, "hmat_rg": -0.087, "hmat_rb": -0.010,
                "hmat_gr": -0.021, "hmat_gg": 1.032, "hmat_gb": -0.007,
                "hmat_br": -0.014, "hmat_bg": -0.050, "hmat_bb": 1.077,
                "saturation": 1.10, "grade_name": "CineStill 800T",
            },
        },
        "Kodachrome 64": {
            "desc": "Legendary slide film. Deep reds, strong cyan-blue, warm midtones.",
            "params": {
                "r_blacks": 0.069, "r_shadows": 0.258, "r_mid": 0.606, "r_highlights": 0.950, "r_whites": 1.000,
                "g_blacks": 0.000, "g_shadows": 0.157, "g_mid": 0.500, "g_highlights": 0.855, "g_whites": 1.000,
                "b_blacks": 0.000, "b_shadows": 0.068, "b_mid": 0.391, "b_highlights": 0.749, "b_whites": 0.916,
                "smat_rr": 1.098, "smat_rg": -0.207, "smat_rb": -0.021,
                "smat_gr": -0.054, "smat_gg": 1.067, "smat_gb": -0.023,
                "smat_br": -0.025, "smat_bg": -0.075, "smat_bb": 1.221,
                "hmat_rr": 1.168, "hmat_rg": -0.174, "hmat_rb": -0.018,
                "hmat_gr": -0.041, "hmat_gg": 1.073, "hmat_gb": -0.014,
                "hmat_br": -0.027, "hmat_bg": -0.081, "hmat_bb": 1.146,
                "saturation": 1.25, "grade_name": "Kodachrome 64",
            },
        },
    },
    "Technical": {
        "S-Log3 to Rec.709": {
            "desc": "Sony S-Log3 to standard display. Adds contrast and saturation to flat footage.",
            "params": {
                "r_blacks": 0.000, "r_shadows": 0.000, "r_mid": 0.417, "r_highlights": 1.000, "r_whites": 1.000,
                "g_blacks": 0.000, "g_shadows": 0.000, "g_mid": 0.417, "g_highlights": 1.000, "g_whites": 1.000,
                "b_blacks": 0.000, "b_shadows": 0.000, "b_mid": 0.417, "b_highlights": 1.000, "b_whites": 1.000,
                "smat_rr": 1.060, "smat_rg": -0.048, "smat_rb": -0.004,
                "smat_gr": -0.032, "smat_gg": 0.985, "smat_gb": -0.014,
                "smat_br": -0.014, "smat_bg": -0.049, "smat_bb": 1.065,
                "hmat_rr": 1.049, "hmat_rg": -0.066, "hmat_rb": -0.008,
                "hmat_gr": -0.028, "hmat_gg": 1.033, "hmat_gb": -0.008,
                "hmat_br": -0.020, "hmat_bg": -0.060, "hmat_bb": 1.053,
                "saturation": 1.30, "grade_name": "S-Log3 to Rec709",
            },
        },
        "ARRI Log-C to Rec.709": {
            "desc": "ARRI Alexa Log-C to standard display. Smooth highlight rolloff.",
            "params": {
                "r_blacks": 0.000, "r_shadows": 0.000, "r_mid": 0.454, "r_highlights": 1.000, "r_whites": 1.000,
                "g_blacks": 0.000, "g_shadows": 0.000, "g_mid": 0.454, "g_highlights": 1.000, "g_whites": 1.000,
                "b_blacks": 0.000, "b_shadows": 0.000, "b_mid": 0.454, "b_highlights": 1.000, "b_whites": 1.000,
                "smat_rr": 1.051, "smat_rg": -0.035, "smat_rb": 0.001,
                "smat_gr": -0.024, "smat_gg": 1.003, "smat_gb": -0.008,
                "smat_br": -0.008, "smat_bg": -0.034, "smat_bb": 1.056,
                "hmat_rr": 1.040, "hmat_rg": -0.056, "hmat_rb": -0.008,
                "hmat_gr": -0.025, "hmat_gg": 1.027, "hmat_gb": -0.008,
                "hmat_br": -0.018, "hmat_bg": -0.050, "hmat_bb": 1.042,
                "saturation": 1.25, "grade_name": "ARRI LogC to Rec709",
            },
        },
        "Canon C-Log to Rec.709": {
            "desc": "Canon C-Log to standard display. Good starting point for Canon cinema cameras.",
            "params": {
                "r_blacks": 0.000, "r_shadows": 0.000, "r_mid": 0.436, "r_highlights": 0.996, "r_whites": 1.000,
                "g_blacks": 0.000, "g_shadows": 0.000, "g_mid": 0.436, "g_highlights": 0.996, "g_whites": 1.000,
                "b_blacks": 0.000, "b_shadows": 0.000, "b_mid": 0.436, "b_highlights": 0.996, "b_whites": 1.000,
                "smat_rr": 1.056, "smat_rg": -0.043, "smat_rb": -0.002,
                "smat_gr": -0.028, "smat_gg": 0.983, "smat_gb": -0.011,
                "smat_br": -0.012, "smat_bg": -0.043, "smat_bb": 1.061,
                "hmat_rr": 1.043, "hmat_rg": -0.063, "hmat_rb": -0.008,
                "hmat_gr": -0.030, "hmat_gg": 1.031, "hmat_gb": -0.010,
                "hmat_br": -0.019, "hmat_bg": -0.057, "hmat_bb": 1.046,
                "saturation": 1.28, "grade_name": "Canon CLog to Rec709",
            },
        },
    },
    "Moods": {
        "Horror Green": {
            "desc": "Sickly green tint, dark, unsettling. Classic horror film look.",
            "params": {
                "r_blacks": 0.012, "r_shadows": 0.041, "r_mid": 0.352, "r_highlights": 0.707, "r_whites": 0.932,
                "g_blacks": 0.049, "g_shadows": 0.149, "g_mid": 0.470, "g_highlights": 0.801, "g_whites": 0.973,
                "b_blacks": 0.024, "b_shadows": 0.097, "b_mid": 0.413, "b_highlights": 0.766, "b_whites": 0.988,
                "smat_rr": 0.761, "smat_rg": 0.165, "smat_rb": 0.002,
                "smat_gr": 0.051, "smat_gg": 0.942, "smat_gb": 0.010,
                "smat_br": 0.044, "smat_bg": 0.171, "smat_bb": 0.744,
                "hmat_rr": 0.792, "hmat_rg": 0.188, "hmat_rb": 0.026,
                "hmat_gr": 0.066, "hmat_gg": 0.942, "hmat_gb": 0.025,
                "hmat_br": 0.062, "hmat_bg": 0.196, "hmat_bb": 0.766,
                "saturation": 0.70, "grade_name": "Horror Green",
            },
        },
        "Romantic Warm": {
            "desc": "Soft, warm, dreamy. Wedding films and love stories.",
            "params": {
                "r_blacks": 0.185, "r_shadows": 0.391, "r_mid": 0.628, "r_highlights": 0.850, "r_whites": 0.998,
                "g_blacks": 0.110, "g_shadows": 0.313, "g_mid": 0.551, "g_highlights": 0.780, "g_whites": 0.952,
                "b_blacks": 0.025, "b_shadows": 0.211, "b_mid": 0.443, "b_highlights": 0.672, "b_whites": 0.850,
                "smat_rr": 0.968, "smat_rg": 0.037, "smat_rb": 0.004,
                "smat_gr": 0.010, "smat_gg": 0.986, "smat_gb": 0.003,
                "smat_br": 0.006, "smat_bg": 0.023, "smat_bb": 0.952,
                "hmat_rr": 0.971, "hmat_rg": 0.035, "hmat_rb": 0.003,
                "hmat_gr": 0.011, "hmat_gg": 0.985, "hmat_gb": 0.004,
                "hmat_br": 0.010, "hmat_bg": 0.029, "hmat_bb": 0.955,
                "saturation": 0.95, "grade_name": "Romantic Warm",
            },
        },
        "Sci-Fi Blue": {
            "desc": "Cold, clinical blue. Futuristic and sterile. Blade Runner vibes.",
            "params": {
                "r_blacks": 0.002, "r_shadows": 0.025, "r_mid": 0.340, "r_highlights": 0.710, "r_whites": 0.917,
                "g_blacks": 0.002, "g_shadows": 0.106, "g_mid": 0.431, "g_highlights": 0.794, "g_whites": 0.991,
                "b_blacks": 0.134, "b_shadows": 0.276, "b_mid": 0.598, "b_highlights": 0.930, "b_whites": 0.996,
                "smat_rr": 0.893, "smat_rg": 0.082, "smat_rb": -0.004,
                "smat_gr": 0.021, "smat_gg": 0.967, "smat_gb": -0.005,
                "smat_br": 0.036, "smat_bg": 0.120, "smat_bb": 0.888,
                "hmat_rr": 0.908, "hmat_rg": 0.090, "hmat_rb": 0.013,
                "hmat_gr": 0.027, "hmat_gg": 0.984, "hmat_gb": 0.010,
                "hmat_br": 0.034, "hmat_bg": 0.119, "hmat_bb": 0.881,
                "saturation": 0.85, "grade_name": "Sci-Fi Blue",
            },
        },
        "Western Dusty": {
            "desc": "Warm, desaturated, dusty. Spaghetti western, desert landscapes.",
            "params": {
                "r_blacks": 0.160, "r_shadows": 0.366, "r_mid": 0.613, "r_highlights": 0.836, "r_whites": 0.989,
                "g_blacks": 0.115, "g_shadows": 0.317, "g_mid": 0.563, "g_highlights": 0.788, "g_whites": 0.956,
                "b_blacks": 0.030, "b_shadows": 0.212, "b_mid": 0.454, "b_highlights": 0.687, "b_whites": 0.865,
                "smat_rr": 0.814, "smat_rg": 0.199, "smat_rb": 0.022,
                "smat_gr": 0.053, "smat_gg": 0.928, "smat_gb": 0.018,
                "smat_br": 0.033, "smat_bg": 0.124, "smat_bb": 0.758,
                "hmat_rr": 0.809, "hmat_rg": 0.189, "hmat_rb": 0.018,
                "hmat_gr": 0.053, "hmat_gg": 0.929, "hmat_gb": 0.018,
                "hmat_br": 0.049, "hmat_bg": 0.150, "hmat_bb": 0.771,
                "saturation": 0.75, "grade_name": "Western Dusty",
            },
        },
        "Matrix Green": {
            "desc": "Heavy green tint with desaturation. The Matrix digital rain look.",
            "params": {
                "r_blacks": 0.024, "r_shadows": 0.055, "r_mid": 0.378, "r_highlights": 0.767, "r_whites": 0.936,
                "g_blacks": 0.072, "g_shadows": 0.167, "g_mid": 0.510, "g_highlights": 0.860, "g_whites": 0.966,
                "b_blacks": 0.030, "b_shadows": 0.087, "b_mid": 0.422, "b_highlights": 0.806, "b_whites": 0.972,
                "smat_rr": 0.681, "smat_rg": 0.215, "smat_rb": 0.003,
                "smat_gr": 0.072, "smat_gg": 0.935, "smat_gb": 0.016,
                "smat_br": 0.056, "smat_bg": 0.221, "smat_bb": 0.650,
                "hmat_rr": 0.727, "hmat_rg": 0.244, "hmat_rb": 0.035,
                "hmat_gr": 0.097, "hmat_gg": 0.903, "hmat_gb": 0.036,
                "hmat_br": 0.087, "hmat_bg": 0.253, "hmat_bb": 0.682,
                "saturation": 0.60, "grade_name": "Matrix Green",
            },
        },
        "Cyberpunk Neon": {
            "desc": "Vivid magenta/cyan, high saturation. Neon-lit nightlife aesthetic.",
            "params": {
                "r_blacks": 0.048, "r_shadows": 0.213, "r_mid": 0.562, "r_highlights": 0.924, "r_whites": 0.992,
                "g_blacks": 0.011, "g_shadows": 0.088, "g_mid": 0.420, "g_highlights": 0.823, "g_whites": 0.995,
                "b_blacks": 0.153, "b_shadows": 0.310, "b_mid": 0.638, "b_highlights": 0.974, "b_whites": 1.000,
                "smat_rr": 1.120, "smat_rg": -0.203, "smat_rb": -0.024,
                "smat_gr": -0.047, "smat_gg": 1.086, "smat_gb": -0.021,
                "smat_br": -0.068, "smat_bg": -0.268, "smat_bb": 1.121,
                "hmat_rr": 1.174, "hmat_rg": -0.177, "hmat_rb": -0.019,
                "hmat_gr": -0.052, "hmat_gg": 1.093, "hmat_gb": -0.019,
                "hmat_br": -0.068, "hmat_bg": -0.259, "hmat_bb": 1.274,
                "saturation": 1.30, "grade_name": "Cyberpunk Neon",
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
        pts_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

        # Step 1: Per-channel tone curves (R, G, B each get their own 5-point curve)
        for ch, prefix in enumerate(["r", "g", "b"]):
            pts_y = np.array([
                params[f"{prefix}_blacks"], params[f"{prefix}_shadows"],
                params[f"{prefix}_mid"], params[f"{prefix}_highlights"],
                params[f"{prefix}_whites"],
            ], dtype=np.float32)
            flat = lut[:, :, :, ch].reshape(-1)
            lut[:, :, :, ch] = np.interp(flat, pts_x, pts_y).reshape(s, s, s).astype(np.float32)

        # Step 2: Dual shadow/highlight color matrices, blended by luminance
        shadow_mat = np.array([
            [params["smat_rr"], params["smat_rg"], params["smat_rb"]],
            [params["smat_gr"], params["smat_gg"], params["smat_gb"]],
            [params["smat_br"], params["smat_bg"], params["smat_bb"]],
        ], dtype=np.float32)
        highlight_mat = np.array([
            [params["hmat_rr"], params["hmat_rg"], params["hmat_rb"]],
            [params["hmat_gr"], params["hmat_gg"], params["hmat_gb"]],
            [params["hmat_br"], params["hmat_bg"], params["hmat_bb"]],
        ], dtype=np.float32)

        flat_pixels = lut.reshape(-1, 3)  # (N, 3)
        lum = 0.2126 * flat_pixels[:, 0] + 0.7152 * flat_pixels[:, 1] + 0.0722 * flat_pixels[:, 2]

        # Smoothstep blend: 0.0 at lum≤0.2 (full shadow), 1.0 at lum≥0.8 (full highlight)
        t = np.clip((lum - 0.2) / 0.6, 0, 1).astype(np.float32)
        blend = t * t * (3.0 - 2.0 * t)  # smoothstep

        shadow_result = flat_pixels @ shadow_mat.T    # (N, 3)
        highlight_result = flat_pixels @ highlight_mat.T  # (N, 3)
        mixed = shadow_result * (1.0 - blend[:, np.newaxis]) + highlight_result * blend[:, np.newaxis]
        lut = np.clip(mixed.reshape(s, s, s, 3), 0, 1).astype(np.float32)

        # Step 3: Saturation
        sat = params["saturation"]
        if sat != 1.0:
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

EXTRACT_PROMPT = """You are an elite Hollywood colorist reverse-engineering a color grade from video frames. Your output will directly generate a 3D LUT (.cube file) for DaVinci Resolve / Premiere Pro.

The engine uses 34 parameters:
- 15 tone curve values: separate R, G, B curves (5 control points each)
- 9 shadow matrix values: 3x3 color mixing applied to dark areas
- 9 highlight matrix values: 3x3 color mixing applied to bright areas
- 1 saturation value

The two matrices are blended by luminance — shadows use smat_*, highlights use hmat_*.
This is how real color grading works: "teal shadows + orange highlights" = different matrices for darks vs brights.

ANALYZE THE FRAMES:
1. Per channel (R, G, B): Are blacks lifted or crushed? Midpoint shifted? Whites rolled off?
2. Shadow color mixing: What color cast is in the DARK areas? Teal? Blue? Warm? Green?
3. Highlight color mixing: What color cast is in the BRIGHT areas? Orange? Golden? Cool?
4. Saturation: Vivid, muted, or near-monochrome?

Return ONLY a JSON object with these 35 fields:

{
  "r_blacks": <float>, "r_shadows": <float>, "r_mid": <float>, "r_highlights": <float>, "r_whites": <float>,
  "g_blacks": <float>, "g_shadows": <float>, "g_mid": <float>, "g_highlights": <float>, "g_whites": <float>,
  "b_blacks": <float>, "b_shadows": <float>, "b_mid": <float>, "b_highlights": <float>, "b_whites": <float>,
  "smat_rr": <float>, "smat_rg": <float>, "smat_rb": <float>,
  "smat_gr": <float>, "smat_gg": <float>, "smat_gb": <float>,
  "smat_br": <float>, "smat_bg": <float>, "smat_bb": <float>,
  "hmat_rr": <float>, "hmat_rg": <float>, "hmat_rb": <float>,
  "hmat_gr": <float>, "hmat_gg": <float>, "hmat_gb": <float>,
  "hmat_br": <float>, "hmat_bg": <float>, "hmat_bb": <float>,
  "saturation": <float>,
  "grade_name": "<short name>"
}

PARAMETER GUIDE:

PER-CHANNEL TONE CURVES (0.0 to 1.0 each):
  5 control points at input 0%, 25%, 50%, 75%, 100%. Identity: 0.0, 0.25, 0.50, 0.75, 1.0
  - Lifted blacks (>0.0): faded/milky. R lifted + B not = warm faded blacks.
  - Crushed blacks (=0.0): deep/dramatic.
  - Rolled-off whites (<1.0): soft highlights. B rolled off = warm highlights.
  - Pushed midpoint (>0.5): brighter midtones in that channel.
  - S-curve (shadows<0.25, highlights>0.75): more contrast.

SHADOW MATRIX (smat_*, applied to dark pixels):
  Controls color mixing in shadows/dark areas. Identity = [[1,0,0],[0,1,0],[0,0,1]]
  Each row sums to ~1.0. smat_rr/rg/rb = how R,G,B inputs contribute to RED output in shadows.

  What to look for in DARK areas of the image:
  - Teal/cyan shadows: smat_rb=0.6-1.0, smat_rr=0.2 (blue input → red output, creating teal)
  - Warm/amber shadows: smat_rr=1.05, smat_rg=0.03 (slight warm push)
  - Green shadows: smat_gg=1.05, smat_gr=0.03
  - Blue shadows: smat_br=0.1, smat_bb=0.9, smat_bg=0.05
  - Neutral shadows: identity matrix

HIGHLIGHT MATRIX (hmat_*, applied to bright pixels):
  Controls color mixing in highlights/bright areas. Same format as shadow matrix.

  What to look for in BRIGHT areas of the image:
  - Orange/warm highlights: hmat_rr=1.05, hmat_rg=0.03, or hmat_br=0.6, hmat_bb=0.3
  - Cool/blue highlights: hmat_bb=1.05, hmat_bg=0.03
  - Golden highlights: hmat_rr=0.6, hmat_rb=0.4 (red from blue input)
  - Neutral highlights: identity matrix

CLASSIC LOOKS with shadow + highlight matrices:
  - Teal & Orange: shadow teal (smat_rb=0.95, smat_br=0.65) + highlight warm (hmat_rr=1.05, hmat_rg=0.03)
  - Bleach Bypass: both matrices = identity. Saturation 0.3-0.5.
  - Cross-process: different channel swaps in shadows vs highlights
  - Fincher cool: shadow slightly blue (smat_bb=1.05) + highlight neutral

  IMPORTANT: Be bold with matrix values. Real cinematic LUTs have off-diagonals of 0.5-1.0
  for strong looks. If shadows are clearly teal and highlights clearly orange, the shadow and
  highlight matrices should be VERY different from each other. Keep each row summing to ~0.9-1.1.

SATURATION (0.0 to 2.0, 1.0 = no change):
  Near monochrome: 0.10-0.20 | Desaturated: 0.50-0.70 | Normal: 1.0 | Vivid: 1.15-1.30

Return ONLY the raw JSON. No markdown, no explanation."""

TEXT_CREATE_PROMPT = """You are an elite Hollywood colorist. Create precise color grading parameters for a 3D LUT.

DESCRIPTION: {description}

The engine uses per-channel R/G/B tone curves + DUAL shadow/highlight 3x3 color matrices + saturation.
The two matrices let you apply DIFFERENT color mixing to shadows vs highlights (like DaVinci Resolve color wheels).

Think about:
- Curves: Which channels are lifted/crushed in blacks? Boosted/rolled in highlights?
- Shadow color: What color cast should dark areas have? (Teal? Blue? Warm? Green?)
- Highlight color: What color cast should bright areas have? (Orange? Golden? Cool?)
- Saturation: Vivid, muted, or desaturated?

FILM STOCKS:
- Kodak Portra: R mid=0.54, G mid=0.50, B mid=0.46. Lifted blacks (~0.04). Both matrices near identity. Sat 0.85.
- Fuji Velvia: Deep blacks (0.00), punchy S-curves. Identity matrices. Sat 1.35.
- CineStill 800T: R boosted (mid=0.56). Shadow matrix warm (smat_rr=1.05, smat_rg=0.03). Sat 0.90.

MOVIE LOOKS:
- Teal & Orange: Shadow matrix: smat_rb=0.8, smat_br=0.6 (teal shadows). Highlight matrix: hmat_rr=1.05, hmat_rg=0.03 (warm highlights).
- The Matrix: G curve boosted. Shadow: smat_gg=1.08, smat_gr=0.04. Highlight similar. Sat 0.65.
- Blade Runner 2049: Heavy shadow teal + highlight orange via dual matrices. Sat 0.80.

MOODS:
- Warm: R curve up, B down. Highlight matrix warm. Shadow matrix near identity.
- Cool: B curve up, R down. Shadow matrix cool. Highlight near identity.
- Faded: All blacks lifted (0.04-0.08), whites rolled. Both matrices identity.

Return ONLY a JSON object:

{{
  "r_blacks": <0.0-0.15>, "r_shadows": <0.10-0.40>, "r_mid": <0.30-0.70>,
  "r_highlights": <0.60-0.90>, "r_whites": <0.85-1.00>,
  "g_blacks": <0.0-0.15>, "g_shadows": <0.10-0.40>, "g_mid": <0.30-0.70>,
  "g_highlights": <0.60-0.90>, "g_whites": <0.85-1.00>,
  "b_blacks": <0.0-0.15>, "b_shadows": <0.10-0.40>, "b_mid": <0.30-0.70>,
  "b_highlights": <0.60-0.90>, "b_whites": <0.85-1.00>,
  "smat_rr": <~1.0>, "smat_rg": <~0.0>, "smat_rb": <~0.0>,
  "smat_gr": <~0.0>, "smat_gg": <~1.0>, "smat_gb": <~0.0>,
  "smat_br": <~0.0>, "smat_bg": <~0.0>, "smat_bb": <~1.0>,
  "hmat_rr": <~1.0>, "hmat_rg": <~0.0>, "hmat_rb": <~0.0>,
  "hmat_gr": <~0.0>, "hmat_gg": <~1.0>, "hmat_gb": <~0.0>,
  "hmat_br": <~0.0>, "hmat_bg": <~0.0>, "hmat_bb": <~1.0>,
  "saturation": <0.0-2.0>,
  "grade_name": "<short name>"
}}

Be bold — subtle curve shifts (0.01-0.02) are invisible. Use 0.04-0.10 for visible changes.
For strong looks, use matrix off-diagonals of 0.5-1.0. Make shadow and highlight matrices
DIFFERENT from each other for cinematic split-tone looks. Keep each row summing to ~1.0.
Return ONLY the raw JSON."""

REQUIRED_KEYS = [
    "r_blacks", "r_shadows", "r_mid", "r_highlights", "r_whites",
    "g_blacks", "g_shadows", "g_mid", "g_highlights", "g_whites",
    "b_blacks", "b_shadows", "b_mid", "b_highlights", "b_whites",
    "smat_rr", "smat_rg", "smat_rb", "smat_gr", "smat_gg", "smat_gb",
    "smat_br", "smat_bg", "smat_bb",
    "hmat_rr", "hmat_rg", "hmat_rb", "hmat_gr", "hmat_gg", "hmat_gb",
    "hmat_br", "hmat_bg", "hmat_bb", "saturation",
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
            self.app.bottom.lbl_status.config(
                text=f"LUT ready: {params['grade_name']}", foreground="#228B22")
        except (json.JSONDecodeError, ValueError) as e:
            self.app.bottom.lbl_status.config(
                text=f"Error: {str(e)[:60]}", foreground="red")
            messagebox.showerror("Parse Error",
                f"{str(e)}\n\nMake sure you copied the COMPLETE JSON response "
                f"from Gemini, including the opening {{ and closing }} braces.")


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

            # Friendly slider labels
            _LABELS = {
                "r_blacks": "Blacks", "r_shadows": "Shadows", "r_mid": "Midtones",
                "r_highlights": "Highlights", "r_whites": "Whites",
                "g_blacks": "Blacks", "g_shadows": "Shadows", "g_mid": "Midtones",
                "g_highlights": "Highlights", "g_whites": "Whites",
                "b_blacks": "Blacks", "b_shadows": "Shadows", "b_mid": "Midtones",
                "b_highlights": "Highlights", "b_whites": "Whites",
                "smat_rr": "R \u2192 R (self)", "smat_rg": "G \u2192 R (mix)", "smat_rb": "B \u2192 R (mix)",
                "smat_gr": "R \u2192 G (mix)", "smat_gg": "G \u2192 G (self)", "smat_gb": "B \u2192 G (mix)",
                "smat_br": "R \u2192 B (mix)", "smat_bg": "G \u2192 B (mix)", "smat_bb": "B \u2192 B (self)",
                "hmat_rr": "R \u2192 R (self)", "hmat_rg": "G \u2192 R (mix)", "hmat_rb": "B \u2192 R (mix)",
                "hmat_gr": "R \u2192 G (mix)", "hmat_gg": "G \u2192 G (self)", "hmat_gb": "B \u2192 G (mix)",
                "hmat_br": "R \u2192 B (mix)", "hmat_bg": "G \u2192 B (mix)", "hmat_bb": "B \u2192 B (self)",
                "saturation": "Saturation",
            }
            label_name = _LABELS.get(param, param.replace("_", " ").title())
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
        # Single row: BEFORE canvas | AFTER canvas | controls column
        row = ttk.Frame(self)
        row.pack(fill="x", padx=10, pady=3)

        # Before
        before_col = ttk.Frame(row)
        before_col.pack(side="left", padx=(0, 3))
        ttk.Label(before_col, text="BEFORE", font=("Helvetica", 8, "bold")).pack()
        self.canvas_before = tk.Canvas(before_col, width=PREVIEW_W, height=PREVIEW_H,
                                        bg="#1a1a1a", highlightthickness=1, highlightbackground="#444")
        self.canvas_before.pack()

        # After
        after_col = ttk.Frame(row)
        after_col.pack(side="left", padx=(3, 10))
        ttk.Label(after_col, text="AFTER", font=("Helvetica", 8, "bold")).pack()
        self.canvas_after = tk.Canvas(after_col, width=PREVIEW_W, height=PREVIEW_H,
                                       bg="#1a1a1a", highlightthickness=1, highlightbackground="#444")
        self.canvas_after.pack()

        # Controls column (right of canvases)
        ctrl = ttk.Frame(row)
        ctrl.pack(side="left", fill="y", padx=5)

        ttk.Button(ctrl, text="Load Image", command=self._load_image).pack(fill="x", pady=(0, 5))
        ttk.Button(ctrl, text="Export .cube", command=self._export).pack(fill="x", pady=(0, 8))

        ttk.Label(ctrl, text="Intensity:").pack(anchor="w")
        self.intensity_var = tk.DoubleVar(value=100.0)
        self.intensity_slider = ttk.Scale(ctrl, from_=0, to=100, orient="horizontal",
                                           variable=self.intensity_var, length=120,
                                           command=self._on_intensity)
        self.intensity_slider.pack(fill="x")
        self.lbl_intensity = ttk.Label(ctrl, text="100%")
        self.lbl_intensity.pack(anchor="w")

        self.lbl_status = ttk.Label(ctrl, text="Ready", foreground="gray", wraplength=150)
        self.lbl_status.pack(anchor="w", pady=(5, 0))

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
