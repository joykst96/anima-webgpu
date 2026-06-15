@echo off
REM ============================================================================
REM  Anima DiT full export pipeline (base + base+turbo)
REM  export_dit -> int8 quant(a4) -> fuse attention -> negpip -> shard -> lora slots
REM
REM  Usage (run from ComfyUI portable root):
REM    export\build_dit.bat <ckpt_path> <output_prefix>
REM  Example:
REM    export\build_dit.bat ComfyUI\models\diffusion_models\my-finetune.safetensors myft
REM      -> out\dit\myft_dit_dyn32_q8a4fns.onnx (+ _q8a4fnLs.onnx) and turbo versions
REM
REM  Defaults used if args omitted.
REM ============================================================================

setlocal enabledelayedexpansion

REM ---- Config (edit for your environment) ------------------------------------
set PY=python_embeded\python.exe
set COMFY=ComfyUI
set TURBO_LORA=ComfyUI\models\loras\anima-turbo-lora-v0.2.safetensors
set OUTDIR=out\dit

REM ---- Args ------------------------------------------------------------------
set CKPT=%~1
set PREFIX=%~2
if "%CKPT%"=="" set CKPT=ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors
if "%PREFIX%"=="" set PREFIX=anima

echo ============================================================
echo  CKPT   = %CKPT%
echo  PREFIX = %PREFIX%
echo  TURBO  = %TURBO_LORA%
echo  OUT    = %OUTDIR%
echo ============================================================
echo.

if not exist "%CKPT%" ( echo [ERROR] checkpoint not found: %CKPT% & exit /b 1 )
if /i not "%CKPT:~-12%"==".safetensors" ( echo [ERROR] ckpt must be a .safetensors model file ^(got: %CKPT%^). The .jpeg is just a preview image. & exit /b 1 )
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM ===========================================================================
REM  BASE pipeline
REM ===========================================================================
echo.
echo ###########  BASE  ###########
call :pipeline "%PREFIX%_dit_dyn32" ""
if errorlevel 1 ( echo [ERROR] BASE pipeline failed & exit /b 1 )

REM ===========================================================================
REM  TURBO pipeline (turbo LoRA merged at export time)
REM ===========================================================================
echo.
echo ###########  TURBO  ###########
if not exist "%TURBO_LORA%" ( echo [ERROR] turbo LoRA not found: %TURBO_LORA% & exit /b 1 )
call :pipeline "%PREFIX%_dit_turbo32" "--lora %TURBO_LORA%"
if errorlevel 1 ( echo [ERROR] TURBO pipeline failed & exit /b 1 )

echo.
echo ============================================================
echo  Done. Outputs (%OUTDIR%\):
echo    %PREFIX%_dit_dyn32_q8a4fns.onnx    (base non-slot)
echo    %PREFIX%_dit_dyn32_q8a4fnLs.onnx   (base slot)
echo    %PREFIX%_dit_turbo32_q8a4fns.onnx  (turbo non-slot)
echo    %PREFIX%_dit_turbo32_q8a4fnLs.onnx (turbo slot)
echo  Deploy each .onnx with its .N.bin shards + manifest + (slot) lora_manifest.
echo  Non-slot and slot share shards; upload both to the server.
echo ============================================================
endlocal
exit /b 0

REM ===========================================================================
REM  :pipeline <base_name> <export_extra_opts>
REM    <base_name> e.g. anima_dit_dyn32  ->  suffixes _q8a4 / _q8a4f / ...
REM ===========================================================================
:pipeline
set BASE=%~1
set EXTRA=%~2
set RAW=%OUTDIR%\%BASE%.onnx
set Q=%OUTDIR%\%BASE%_q8a4.onnx
set F=%OUTDIR%\%BASE%_q8a4f.onnx
set N=%OUTDIR%\%BASE%_q8a4fn.onnx
set S=%OUTDIR%\%BASE%_q8a4fns.onnx
set L=%OUTDIR%\%BASE%_q8a4fnLs.onnx

echo.
echo --- [1/6] export_dit (%BASE%) ---
%PY% export\export_dit.py --comfyui %COMFY% --ckpt "%CKPT%" %EXTRA% --out "%RAW%" --dynamic --fp32-act
if errorlevel 1 exit /b 1

echo.
echo --- [2/6] int8 quant (accuracy_level=4) ---
%PY% export\quantize_dit.py --src "%RAW%" --out "%Q%" --bits 8 --no-exclude --accuracy-level 4
if errorlevel 1 exit /b 1

echo.
echo --- [3/6] fuse attention (-> MultiHeadAttention, emits .fuse_meta.json) ---
%PY% export\fuse_attention.py --src "%Q%" --out "%F%"
if errorlevel 1 exit /b 1

echo.
echo --- [4/6] negpip patch (meta-based identification) ---
%PY% export\add_negpip.py --src "%F%" --out "%N%"
if errorlevel 1 exit /b 1

echo.
echo --- [5/6] shard (--shard-mb 400) ---
%PY% export\shard_onnx_data.py --src "%N%" --out "%S%" --shard-mb 400
if errorlevel 1 exit /b 1

echo.
echo --- [6/6] add lora slots (rank 48, explicit fuse_meta path) ---
%PY% export\add_lora_slots.py --src "%S%" --out "%L%" --rank 48 --fuse-meta "%F%.fuse_meta.json"
if errorlevel 1 exit /b 1

echo --- %BASE% done (non-slot %S% / slot %L%) ---
exit /b 0
