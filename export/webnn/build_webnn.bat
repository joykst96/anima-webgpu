@echo off
REM ============================================================================
REM  Anima WebNN graphs (experimental "WebNN" precision option of the page)
REM
REM  DiT (base and turbo), per variant - DYNAMIC graphs: the page fixes the latent size with
REM  freeDimensionOverrides when it creates the session and creates it again when the resolution changes.
REM    export_dit --dynamic (fp16 load) -> webnn_fix -> shard -> add_negpip -> add_lora_slots
REM      -> int8 ConvRot (webnn_q8) -> spectrum_split
REM    -> <prefix>_dit_{dyn32|turbo32}_q8a4fns_wnnd.onnx   fp16 weights (+ .N.bin shards, .manifest.json)
REM    -> ..._q8a4fns_wnndn.onnx                     + NegPip input (negpip_mask, add_negpip --graph-only, same shards)
REM    -> ..._q8a4fnLs_wnnd.onnx, ..._q8a4fnLs_wnndn.onnx   LoRA slot graphs, same shards (+ .lora_manifest.json)
REM    -> ..._q8a4fns_wnnd_head.onnx (+ .data), ..._headk.onnx   Spectrum heads (fp16, used by fp16 and int8)
REM    -> ..._wnndr*.onnx (the same four graphs)     int8 weights with ConvRot (webnn_q8.py --dq-cast --rot 64):
REM                                     own shards ..._q8a4fns_wnndr.onnx.N.bin (about 2 GB); the NegPip and
REM                                     slot graphs are rewritten graph-only with --like to point at them
REM    -> ..._sp.onnx (all eight)                    same graphs + the Spectrum feature output
REM  VAE  : vae_fp16 (all) on the existing dynamic decoder graph
REM         -> <vae>_wnn16.onnx (graph only, shares the original <vae>.onnx.data)
REM
REM  There is no WebNN-specific exporter. The DiT step runs the same
REM  export\export_dit.py as the WebGPU build (build_dit.bat) with different
REM  options: --dynamic without --fp32-act, so the checkpoint is loaded as fp16.
REM  Everything WebNN-specific lives in this folder.
REM
REM  Usage (run from ComfyUI portable root):
REM    <repo>\export\webnn\build_webnn.bat <ckpt_path> <output_prefix> [vae_onnx] [base|turbo|both]
REM  Example:
REM    C:\repo\export\webnn\build_webnn.bat ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors animav10
REM
REM  The last argument picks the variants (default: both). Each variant is
REM  about 3.6 GB (fp16) + 2.0 GB (int8) of shards.
REM ============================================================================

setlocal enabledelayedexpansion

REM ---- Config (edit for your environment) ------------------------------------
set PY=python_embeded\python.exe
set COMFY=ComfyUI
set TURBO_LORA=ComfyUI\models\loras\anima-turbo-lora-v0.2.safetensors
set OUTDIR=out\dit
set WORKDIR=out\webnn-work
set ROT=64

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
echo    ..._q8a4fns_wnnd.onnx       fp16 + .N.bin shards + .manifest.json
echo    ..._q8a4fns_wnndr.onnx      int8 ConvRot + its own .N.bin shards + .manifest.json
echo    ..._wnndn, ..._wnndrn       the same with the NegPip input ^(+ manifests^)
echo    ..._q8a4fnLs_wnnd*.onnx     LoRA slot graphs ^(+ .manifest.json + .lora_manifest.json^)
echo    ..._sp.onnx ^(all^)           Spectrum main graphs ^(use the manifests of the graphs above^)
echo    ..._q8a4fns_wnnd_head.onnx  + .data, ..._headk.onnx   Spectrum heads
echo    %VAE:~0,-5%_wnn16.onnx  ^(shares %VAE%.data^)
echo  The zero .lora.bin files are not deployed ^(the page builds them^).
echo  The intermediate export in %WORKDIR%\ ^(about 8GB per variant^) can be deleted.
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
set RAW=%WORKDIR%\%BASE%_dyn_f16.onnx
set FIX=%WORKDIR%\%BASE%_dyn_f16w.onnx
set Q8=%WORKDIR%\%BASE%_dyn_q8r.onnx
set OUT=%OUTDIR%\%BASE%_q8a4fns_wnnd.onnx
set OUTN=%OUTDIR%\%BASE%_q8a4fns_wnndn.onnx
set OUTL=%OUTDIR%\%BASE%_q8a4fnLs_wnnd.onnx
set OUTLN=%OUTDIR%\%BASE%_q8a4fnLs_wnndn.onnx
set HEAD=%OUTDIR%\%BASE%_q8a4fns_wnnd_head.onnx
set OUTR=%OUTDIR%\%BASE%_q8a4fns_wnndr.onnx

echo.
echo --- [1/7] export_dit (%BASE%, dynamic, fp16 load) ---
%PY% "%EX%\export_dit.py" --comfyui %COMFY% --ckpt "%CKPT%" %EXTRA% --size 1024 --dynamic --out "%RAW%"
if errorlevel 1 exit /b 1

echo.
echo --- [2/7] webnn_fix (5D MatMul, allowzero Reshape, fp16 LayerNorm) ---
%PY% -I "%WN%webnn_fix.py" "%RAW%" "%FIX%"
if errorlevel 1 exit /b 1

echo.
echo --- [3/7] shard fp16 (--shard-mb 400) ---
%PY% "%EX%\shard_onnx_data.py" --src "%FIX%" --out "%OUT%" --shard-mb 400
if errorlevel 1 exit /b 1

echo.
echo --- [4/7] add negpip (graph only: negpip_mask input on the cross-attention v_proj, same shards) ---
%PY% "%EX%\add_negpip.py" --graph-only --src "%OUT%" --out "%OUTN%"
if errorlevel 1 exit /b 1

echo.
echo --- [5/7] add lora slots (rank 48, fp16 branches, same lora.bin layout as the WebGPU slot graph) ---
REM  The NegPip slot graph is made from the NegPip graph so the v_proj LoRA branch also gets the masked context
%PY% "%EX%\add_lora_slots.py" --src "%OUT%" --out "%OUTL%" --rank 48
if errorlevel 1 exit /b 1
%PY% "%EX%\add_lora_slots.py" --src "%OUTN%" --out "%OUTLN%" --rank 48
if errorlevel 1 exit /b 1

echo.
echo --- [6/7] int8 ConvRot (webnn_q8 --dq-cast --rot %ROT%, own shards; NegPip and slot graphs graph-only with --like) ---
%PY% -I "%WN%webnn_q8.py" "%OUT%" "%Q8%" --dq-cast --rot %ROT%
if errorlevel 1 exit /b 1
%PY% "%EX%\shard_onnx_data.py" --src "%Q8%" --out "%OUTR%" --shard-mb 400
if errorlevel 1 exit /b 1
del "%Q8%" "%Q8%.data"
%PY% -I "%WN%webnn_q8.py" "%OUTN%" "%OUTDIR%\%BASE%_q8a4fns_wnndrn.onnx" --like "%OUTR%"
if errorlevel 1 exit /b 1
%PY% -I "%WN%webnn_q8.py" "%OUTL%" "%OUTDIR%\%BASE%_q8a4fnLs_wnndr.onnx" --like "%OUTR%"
if errorlevel 1 exit /b 1
%PY% -I "%WN%webnn_q8.py" "%OUTLN%" "%OUTDIR%\%BASE%_q8a4fnLs_wnndrn.onnx" --like "%OUTR%"
if errorlevel 1 exit /b 1
copy /y "%OUTL%.lora_manifest.json" "%OUTDIR%\%BASE%_q8a4fnLs_wnndr.onnx.lora_manifest.json" >nul
copy /y "%OUTLN%.lora_manifest.json" "%OUTDIR%\%BASE%_q8a4fnLs_wnndrn.onnx.lora_manifest.json" >nul

echo.
echo --- [7/7] spectrum split (main graphs with the feature output + the fp16 head) ---
%PY% -I "%WN%spectrum_split.py" "%OUT%" --head "%HEAD%"
if errorlevel 1 exit /b 1
for %%G in (%BASE%_q8a4fns_wnndn %BASE%_q8a4fnLs_wnnd %BASE%_q8a4fnLs_wnndn %BASE%_q8a4fns_wnndr %BASE%_q8a4fns_wnndrn %BASE%_q8a4fnLs_wnndr %BASE%_q8a4fnLs_wnndrn) do (
  %PY% -I "%WN%spectrum_split.py" "%OUTDIR%\%%G.onnx"
  if errorlevel 1 exit /b 1
)

echo --- %BASE% done ---
exit /b 0
