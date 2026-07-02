param(
    [ValidateSet("--check", "--install")]
    [string]$Mode = "--check",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

function Test-PythonPackage {
    param([string]$Module)
    & $Python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$Module') else 1)" | Out-Null
    return $LASTEXITCODE -eq 0
}

function Find-Tesseract {
    $cmd = Get-Command tesseract -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    $candidates = @(
        "C:\Program Files\Tesseract-OCR\tesseract.exe",
        "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return $null
}

function Install-PythonPackages {
    & $Python -m pip install --upgrade pymupdf pillow numpy
    if ($LASTEXITCODE -ne 0) {
        throw "Python package installation failed."
    }
}

function Install-Tesseract {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id UB-Mannheim.TesseractOCR -e --accept-package-agreements --accept-source-agreements
        return
    }
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        choco install tesseract -y
        return
    }
    throw "Install Tesseract OCR manually, then rerun this script. Recommended Windows package: UB-Mannheim.TesseractOCR."
}

if ($Mode -eq "--install") {
    Install-PythonPackages
    Install-Tesseract
}

$missingPackages = @()
if (-not (Test-PythonPackage "fitz")) { $missingPackages += "pymupdf" }
if (-not (Test-PythonPackage "PIL")) { $missingPackages += "pillow" }
if (-not (Test-PythonPackage "numpy")) { $missingPackages += "numpy" }
if ($missingPackages.Count -gt 0) {
    throw "Missing Python packages: $($missingPackages -join ', '). Run: .\scripts\install_dependencies.ps1 --install"
}

$tesseract = Find-Tesseract
if (-not $tesseract) {
    throw "Missing dependency: tesseract. Run: .\scripts\install_dependencies.ps1 --install"
}

$langs = & $tesseract --list-langs 2>$null
$missingLangs = @()
foreach ($lang in @("eng", "osd", "chi_sim", "chi_tra")) {
    if ($langs -notcontains $lang) {
        $missingLangs += $lang
    }
}
if ($missingLangs.Count -gt 0) {
    throw "Missing Tesseract language data: $($missingLangs -join ', '). Install the missing traineddata files, then rerun this script."
}

Write-Host "Dependency check passed."
& $Python -c "import fitz, numpy, PIL; print('PyMuPDF ' + fitz.version[0]); print('NumPy ' + numpy.__version__); print('Pillow ' + PIL.__version__)"
& $tesseract --version | Select-Object -First 1
Write-Host "OCR languages available: eng, chi_sim, chi_tra"
