@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   LaTeX Compilation
echo ============================================
echo.

set TEXFILE=main
set FAILED=0

REM --- Step 1: XeLaTeX (generate aux) ---
echo [1/4] XeLaTeX - first pass...
xelatex -interaction=nonstopmode %TEXFILE%.tex
if %ERRORLEVEL% neq 0 (
    set FAILED=1
    echo [ERROR] XeLaTeX first pass failed! Check %TEXFILE%.log for details.
    goto cleanup
)
echo        Done.

REM --- Step 2: BibTeX ---
echo [2/4] BibTeX - processing bibliography...
bibtex %TEXFILE%
if %ERRORLEVEL% neq 0 (
    set FAILED=1
    echo [ERROR] BibTeX failed! Check %TEXFILE%.blg for details.
    goto cleanup
)
echo        Done.

REM --- Step 3: XeLaTeX (resolve refs) ---
echo [3/4] XeLaTeX - second pass...
xelatex -interaction=nonstopmode %TEXFILE%.tex
if %ERRORLEVEL% neq 0 (
    set FAILED=1
    echo [ERROR] XeLaTeX second pass failed! Check %TEXFILE%.log for details.
    goto cleanup
)
echo        Done.

REM --- Step 4: XeLaTeX (final) ---
echo [4/4] XeLaTeX - final pass...
xelatex -interaction=nonstopmode %TEXFILE%.tex
if %ERRORLEVEL% neq 0 (
    set FAILED=1
    echo [ERROR] XeLaTeX final pass failed! Check %TEXFILE%.log for details.
    goto cleanup
)
echo        Done.

REM --- Cleanup ---
:cleanup
echo.
echo Cleaning up temporary files...
del /q %TEXFILE%.aux %TEXFILE%.log %TEXFILE%.out %TEXFILE%.toc %TEXFILE%.lof %TEXFILE%.lot %TEXFILE%.fls %TEXFILE%.fdb_latexmk %TEXFILE%.synctex.gz %TEXFILE%.bbl %TEXFILE%.blg %TEXFILE%.bcf %TEXFILE%.run.xml 2>nul
echo        Done.

REM --- Result ---
echo.
if "!FAILED!"=="0" (
    echo ============================================
    echo   Compilation successful!
    echo   Output: %TEXFILE%.pdf
    echo ============================================
) else (
    echo ============================================
    echo   Compilation FAILED.
    echo   See log files for details.
    echo ============================================
)

endlocal
