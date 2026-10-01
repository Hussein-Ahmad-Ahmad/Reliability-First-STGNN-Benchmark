@echo off
setlocal
cd /d "%~dp0\.."

set PYTHON=.venv\Scripts\python.exe
set PASSES=50
set DATASET=METR-LA
set SEED=43

for %%M in (D2STGNN MegaCRN MTGNN STNorm STGCNChebGraphConv STID STAEformer) do (
    echo Running %%M on %DATASET% with %PASSES% stochastic passes...
    "%PYTHON%" scripts\run_mc_dropout_inference.py --model %%M --dataset %DATASET% --seed %SEED% --passes %PASSES% --batch-size 64 --device cuda
    if errorlevel 1 exit /b 1
)

echo Completed MC Dropout generation for %DATASET%.
endlocal
