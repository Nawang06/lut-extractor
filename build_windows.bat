@echo off
echo ========================================
echo   Building LUT Extractor for Windows
echo ========================================
echo.
echo Installing dependencies...
pip install opencv-python numpy Pillow pyinstaller
echo.
echo Building executable...
pyinstaller --onefile --windowed --name "LUT Extractor" --clean lut_extractor_app.py
echo.
echo ========================================
echo   BUILD COMPLETE!
echo   Find your .exe at: dist\LUT Extractor.exe
echo ========================================
echo.
pause
