@echo off
REM run.bat - Windows equivalent of run.sh
REM Usage: run.bat [train|infer|all]

SET MODE=%1
IF "%MODE%"=="" SET MODE=all

echo ============================================
echo  CSCD618/DSCD604 Assignment 2 Pipeline
echo ============================================

REM ── Install dependencies ─────────────────────────────────────────────────
echo.
echo [Step 0] Installing dependencies ...
pip install -r requirements.txt -q

REM ── Train ────────────────────────────────────────────────────────────────
IF "%MODE%"=="train" GOTO TRAIN
IF "%MODE%"=="all"   GOTO TRAIN
GOTO INFER

:TRAIN
echo.
echo [Step 1] Training the model ...
python train.py
IF ERRORLEVEL 1 GOTO ERROR

REM ── Inference ────────────────────────────────────────────────────────────
:INFER
IF "%MODE%"=="infer" GOTO DO_INFER
IF "%MODE%"=="all"   GOTO DO_INFER
GOTO DONE

:DO_INFER
echo.
echo [Step 2] Running inference with TTA ...
python inference.py --tta --tta-steps 8
IF ERRORLEVEL 1 GOTO ERROR

:DONE
echo.
echo ============================================
echo  Done! Submission file: submission.csv
echo ============================================
GOTO END

:ERROR
echo.
echo [ERROR] Pipeline failed. Check the error messages above.
EXIT /B 1

:END
