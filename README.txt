================================================================
   LUT Extractor - Extract Cinematic LUTs from Any Video
================================================================

WHAT THIS DOES:
   Takes any video and extracts its "cinematic look" as a
   .cube LUT file that you can apply in Premiere Pro,
   DaVinci Resolve, or Final Cut Pro.

HOW TO USE:
   1. Double-click "LUT Extractor.exe" (or run lut_extractor_app.py)
   2. Click "Select Video File" - pick your video
   3. App extracts frames automatically
   4. Click "Open Gemini" - opens Google Gemini AI in your browser
   5. Upload the frame images from the frames folder
   6. Click "Copy Prompt" - pastes the analysis prompt
   7. Paste the prompt into Gemini chat
   8. Copy Gemini's JSON response
   9. Paste it into the app
  10. Click "Generate LUT" - done!

APPLY THE LUT:
   Premiere Pro:  Lumetri Color > Creative > Look > Browse
   DaVinci:       Color > LUTs > Browse
   Final Cut:     Effects > Custom LUT > Browse

BUILDING THE .EXE (for developers):
   1. Install Python 3.10+
   2. Double-click build_windows.bat
   3. Find the .exe in the dist/ folder

NO API KEY NEEDED:
   This uses the free Gemini web app (gemini.google.com)
   for AI analysis. No API key, no billing, no limits.

================================================================
