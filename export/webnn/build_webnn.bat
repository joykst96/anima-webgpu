@echo off
REM ============================================================================
REM  Anima WebNN graphs (experimental "WebNN" precision option of the page)
REM
REM  DiT (base and turbo), per variant:
REM    export_dit (static 1024x1024, fp16 load) -> webnn_fix -> shard
REM      -> add_lora_slots -> spectrum_split
REM    -> <prefix>_dit_{dyn32|turbo32}_q8a4fns_wnn1024.onnx   (+ .N.bin, .manifest.json)
REM    -> ..._q8a4fnLs_wnn1024.onnx     LoRA slot graph, same shards
REM                                     (+ .manifest.json, .lora_manifest.json)
REM    -> ..._wnn1024_sp.onnx (both)    same graphs + the Spectrum feature output
REM    -> ..._q8a4fns_wnn1024_head.onnx (+ .data)  Spectrum head: feature, timesteps -> v_pred
REM    -> ..._q8a4fns_wnn1024_headk.onnx           head with the weighted sum of 16 features (shares the .data)
REM    -> ..._q8a4fn{s,Ls}_wnn1024n.onnx (+ _sp)   the same four graphs with the NegPip input
REM                                     (negpip_mask, add_negpip --graph-only before the LoRA slots;
REM                                     same shards and head, + .manifest.json, .lora_manifest.json)
REM  VAE  : vae_fp16 (all) on the existing dynamic decoder graph
REM         -> <vae>_wnn16.onnx (graph only, shares the original <vae>.onnx.data)
REM
REM  There is no WebNN-specific exporter. The DiT step runs the same
REM  export\export_dit.py as the WebGPU build (build_dit.bat) with different
REM  options: no --dynamic / --fp32-act, so the checkpoint is loaded as fp16
REM  and the graph is static. Everything WebNN-specific lives in this folder.
REM
REM  Usage (run from ComfyUI portable root):
REM    <repo>\export\webnn\build_webnn.bat <ckpt_path> <output_prefix> [vae_onnx] [base|turbo|both]
REM  Example:
REM    C:\repo\export\webnn\build_webnn.bat ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors animav10
REM
REM  The last argument picks the variants (default: both). Each variant is
REM  about 3.6 GB of shards.
REM ============================================================================

setlocal enabledelayedexpansion

REM ---- Config (edit for your environment) ------------------------------------
set PY=python_embeded\python.exe
set COMFY=ComfyUI
set TURBO_LORA=ComfyUI\models\loras\anima-turbo-lora-v0.2.safetensors
set OUTDIR=out\dit
set WORKDIR=out\webnn-work
set SIZE=1024

REM ---- Script locations (this file is <repo>\export\webnn\build_webnn.bat) ------
set WN=%~dp0
for %%I in ("%~dp0..") do set EX=%%~fI

REM ---- Args ------------------------------------------------------------------
set CKPT=%~1
set PREFIX=%~2
set VAE=%~3
set WHICH=%~4
if "%CKPT%"=="" set CKPT=ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors
if "%PREFIX%"=="" set PREFIX=anima
if "%VAE%"=="" set VAE=out\vae\qwen_image_vae_decoder_dyn32_2dc.onnx
if "%WHICH%"=="" set WHICH=both

echo ============================================================
echo  CKPT     = %CKPT%
echo  PREFIX   = %PREFIX%
echo  TURBO    = %TURBO_LORA%
echo  VARIANTS = %WHICH%
echo  OUT      = %OUTDIR%
echo  VAE      = %VAE%
echo ============================================================
echo.

if not exist "%CKPT%" ( echo [ERROR] checkpoint not found: %CKPT% & exit /b 1 )
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
if not exist "%WORKDIR%" mkdir "%WORKDIR%"

if /i not "%WHICH%"=="turbo" (
  echo.
  echo ###########  BASE  ###########
  call :pipeline "%PREFIX%_dit_dyn32" ""
  if errorlevel 1 ( echo [ERROR] BASE pipeline failed & exit /b 1 )
)
if /i not "%WHICH%"=="base" (
  echo.
  echo ###########  TURBO  ###########
  if not exist "%TURBO_LORA%" ( echo [ERROR] turbo LoRA not found: %TURBO_LORA% & exit /b 1 )
  call :pipeline "%PREFIX%_dit_turbo32" "--lora %TURBO_LORA%"
  if errorlevel 1 ( echo [ERROR] TURBO pipeline failed & exit /b 1 )
)

echo.
if exist "%VAE%" (
  echo --- [VAE] vae_fp16 all ^(graph only^) ---
  %PY% -I "%WN%vae_fp16.py" "%VAE%" "%VAE:~0,-5%_wnn16.onnx" all latent
  if errorlevel 1 exit /b 1
) else (
  echo [SKIP] VAE graph not found: %VAE%
  echo        Build it with the WebGPU pipeline first, then rerun or run vae_fp16.py by hand.
)

echo.
echo ============================================================
echo  Done. Outputs in %OUTDIR%\ per variant ^(dyn32 = base, turbo32 = turbo^):
echo    ..._q8a4fns_wnn%SIZE%.onnx       + .N.bin shards + .manifest.json
echo    ..._q8a4fnLs_wnn%SIZE%.onnx      LoRA slot graph: + .manifest.json + .lora_manifest.json
echo    ..._wnn%SIZE%_sp.onnx ^(both^)     Spectrum main graphs ^(use the manifests of the graphs above^)
echo    ..._q8a4fns_wnn%SIZE%_head.onnx  + .data, ..._headk.onnx   Spectrum heads
echo    ..._wnn%SIZE%n.onnx, ..._wnn%SIZE%n_sp.onnx ^(both^)  the same graphs with the NegPip input ^(+ their manifests^)
echo    %VAE:~0,-5%_wnn16.onnx  ^(shares %VAE%.data^)
echo  The zero .lora.bin files are not deployed ^(the page builds them^).
echo  The intermediate export in %WORKDIR%\ ^(about 4GB per variant^) can be deleted.
echo ============================================================
endlocal
exit /b 0

REM ===========================================================================
REM  :pipeline <base_name> <export_extra_opts>
REM    <base_name> e.g. anima_dit_turbo32
REM ===========================================================================
:pipeline
set BASE=%~1
set EXTRA=%~2
set RAW=%WORKDIR%\%BASE%_s%SIZE%_f16.onnx
set FIX=%WORKDIR%\%BASE%_s%SIZE%_f16w.onnx
set OUT=%OUTDIR%\%BASE%_q8a4fns_wnn%SIZE%.onnx
set OUTL=%OUTDIR%\%BASE%_q8a4fnLs_wnn%SIZE%.onnx
set HEAD=%OUTDIR%\%BASE%_q8a4fns_wnn%SIZE%_head.onnx
set OUTN=%OUTDIR%\%BASE%_q8a4fns_wnn%SIZE%n.onnx
set OUTLN=%OUTDIR%\%BASE%_q8a4fnLs_wnn%SIZE%n.onnx

echo.
echo --- [1/6] export_dit (%BASE%, static %SIZE%, fp16 load) ---
%PY% "%EX%\export_dit.py" --comfyui %COMFY% --ckpt "%CKPT%" %EXTRA% --size %SIZE% --out "%RAW%"
if errorlevel 1 exit /b 1

echo.
echo --- [2/6] webnn_fix (5D MatMul, allowzero Reshape, fp16 LayerNorm) ---
%PY% -I "%WN%webnn_fix.py" "%RAW%" "%FIX%"
if errorlevel 1 exit /b 1

echo.
echo --- [3/6] shard (--shard-mb 400) ---
%PY% "%EX%\shard_onnx_data.py" --src "%FIX%" --out "%OUT%" --shard-mb 400
if errorlevel 1 exit /b 1

echo.
echo --- [4/6] add negpip (graph only: negpip_mask input on the cross-attention v_proj, same shards) ---
%PY% "%EX%\add_negpip.py" --graph-only --src "%OUT%" --out "%OUTN%"
if errorlevel 1 exit /b 1

echo.
echo --- [5/6] add lora slots (rank 48, fp16 branches, same lora.bin layout as the WebGPU slot graph) ---
REM  The NegPip slot graph is made from the NegPip graph so the v_proj LoRA branch also gets the masked context
%PY% "%EX%\add_lora_slots.py" --src "%OUT%" --out "%OUTL%" --rank 48
if errorlevel 1 exit /b 1
%PY% "%EX%\add_lora_slots.py" --src "%OUTN%" --out "%OUTLN%" --rank 48
if errorlevel 1 exit /b 1

echo.
echo --- [6/6] spectrum split (main graphs with the feature output + head) ---
%PY% -I "%WN%spectrum_split.py" "%OUT%" --head "%HEAD%"
if errorlevel 1 exit /b 1
%PY% -I "%WN%spectrum_split.py" "%OUTL%"
if errorlevel 1 exit /b 1
%PY% -I "%WN%spectrum_split.py" "%OUTN%"
if errorlevel 1 exit /b 1
%PY% -I "%WN%spectrum_split.py" "%OUTLN%"
if errorlevel 1 exit /b 1

echo --- %BASE% done ---
exit /b 0
