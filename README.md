# LUT Extractor Pro

Extract, create, and customize cinematic LUTs from any video, image, or text description.

## Features

| Tab | What it does | AI needed? |
|-----|-------------|------------|
| **Extract from Media** | Analyze video/image frames to reverse-engineer the color grade | Yes (free Gemini web app) |
| **Create from Text** | Describe a look ("Blade Runner 2049", "warm vintage film") | Yes (free Gemini web app) |
| **Preset Library** | 22 built-in presets across 4 categories | No |
| **Editor** | 17 sliders with real-time before/after preview | No |

## Presets (no AI, instant)

- **Cinematic:** Teal & Orange, Bleach Bypass, Vintage Film, Moody Desaturated, Golden Hour, Noir, Pastel Faded, High Contrast Drama
- **Film Stocks:** Kodak Vision3 500T, Portra 400, Fuji Velvia 50, CineStill 800T, Kodachrome 64
- **Technical:** Sony S-Log3, ARRI Log-C, Canon C-Log to Rec.709
- **Moods:** Horror Green, Romantic Warm, Sci-Fi Blue, Western Dusty, Matrix Green, Cyberpunk Neon

## Download

Get the latest .exe from [Releases](https://github.com/Nawang06/lut-extractor/releases). No Python, no install, no API keys.

## Using the .cube file

- **Premiere Pro:** Lumetri Color > Creative > Look > Browse
- **DaVinci Resolve:** Color > LUTs > Browse
- **Final Cut Pro:** Effects > Custom LUT > Browse

## Build from source

```bash
pip install -r requirements.txt pyinstaller
pyinstaller --onefile --windowed --name "LUT Extractor Pro" lut_extractor_pro.py
```
