#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_models.py — model-list.yml → web/js/config.js 정적 주입기.

대상 파일의 마커 사이를 yml 내용으로 교체한다. 결과는 정적 파일
(런타임 fetch 없음). 마커 밖 코드(custom 항목 등)는 건드리지 않는다.

사용법:
  python gen_models.py model-list.yml web/js/config.js
  python gen_models.py model-list.yml web/js/config.js --check   # 변경 미적용, 보고만

config.js 모드 (대상이 .js) — 배포별 차이는 이 파일 하나로 모인다:
  PATH_DEFAULTS:START..END     — PATHS 객체
  MODELS:START..END            — MODELS 맵 항목 (custom은 마커 밖이라 보존)
  SITE_DEFAULTS:START..END     — SELECTED_MODEL, PROMPT_DEFAULTS, MODEL_SOURCES, CONVERT_URL
                                 (드롭다운·프롬프트 기본값·출처 링크·변환 페이지 링크는 페이지가 이 데이터로 그린다.
                                  convert_url이 없으면 "" → 페이지 기본값 convert/)

레거시 html 모드 (대상이 .html, 모듈 분리 이전의 단일 index.html) 교체 대상:
  MODELS:START..END            — MODELS 맵 항목 (custom은 마커 밖이라 보존)
  MODEL_OPTIONS:START..END     — <select> option (custom은 마커 밖이라 보존)
  PATH_DEFAULTS:START..END     — PATHS 객체
  <textarea id="prompt">…</textarea>   — 긍정 프롬프트 초기값
  <textarea id="negative">…</textarea> — 부정 프롬프트 초기값
  FOOTER_SOURCES:START..END    — (있으면) 모델 출처 링크. 없으면 스킵.

파일 경로 규칙 (key + variant):
  base : out/dit/{key}_dit_dyn32_q8a4fns.onnx  / _q8a4fnLs.onnx (슬롯)
  turbo: out/dit/{key}_dit_turbo32_q8a4fns.onnx / _q8a4fnLs.onnx
"""
import argparse
import html
import re
import sys

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML이 필요합니다.  pip install pyyaml")

VARIANT_INFIX = {"base": "dit_dyn32", "turbo": "dit_turbo32"}


def dit_paths(key, variant):
    """key+variant → (비슬롯 dit, 슬롯 ditLora) 경로."""
    infix = VARIANT_INFIX[variant]
    base = f"out/dit/{key}_{infix}_q8a4f"
    return base + "ns.onnx", base + "nLs.onnx"


def js_str(s):
    """JS 문자열 리터럴로 안전 인코딩 (큰따옴표 이스케이프)."""
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_model_entries(cfg):
    """yml → MODELS 항목 리스트 [(model_key, dict)], option 리스트 [(key,label)]."""
    defaults = cfg.get("defaults", {})
    entries = []  # (model_key, fields_dict)
    options = []  # (model_key, label)

    for m in cfg["models"]:
        key = m["key"]
        base_label = m.get("label", key)
        variants = m.get("variants", ["base", "turbo"])
        # variants가 dict 형태({base: {...}})도, list 형태([base, turbo])도 허용
        if isinstance(variants, dict):
            variant_names = list(variants.keys())
        else:
            variant_names = list(variants)
        overrides = m.get("overrides", {}) or {}

        for variant in variant_names:
            if variant not in VARIANT_INFIX:
                sys.exit(f"ERROR: 알 수 없는 variant '{variant}' (key={key})")
            d = dict(defaults.get(variant, {}))  # variant 기본값 복사
            ov = dict(overrides.get(variant, {}) or {})
            d.update(ov)  # override 적용

            model_key = f"{key}_{variant}"
            suffix = d.get("label_suffix", "")
            # turbo는 모델명에 '+Turbo'를 붙여 base와 구분 (override label이 있으면 존중)
            if "label" in ov:
                label_core = ov["label"]
            elif variant == "turbo":
                label_core = base_label + "+Turbo"
            else:
                label_core = base_label
            label = (label_core + " " + suffix).strip()

            dit, dit_lora = dit_paths(key, variant)
            # 경로 직접 지정(override) 허용
            dit = d.get("dit", dit)
            dit_lora = d.get("ditLora", dit_lora)

            fields = {
                "label": label,
                "dit": dit,
                "ditLora": dit_lora,
                "steps": d.get("steps", 8),
                "cfg": d.get("cfg", 1.0),
                "sampler": d.get("sampler", "euler"),
                "scheduler": d.get("scheduler", "simple"),
            }
            entries.append((model_key, fields))
            options.append((model_key, label))

    return entries, options


HTML_INDENT = "      "  # index.html <script> 안 최상위 들여쓰기 (레거시 html 모드)


def render_models_js(entries, base=HTML_INDENT):
    lines = []
    for mk, f in entries:
        lines.append(f"{base}  {mk}: {{")
        lines.append(f"{base}    label: {js_str(f['label'])},")
        lines.append(f"{base}    dit: {js_str(f['dit'])},")
        lines.append(f"{base}    ditLora: {js_str(f['ditLora'])},")
        lines.append(f"{base}    steps: {f['steps']},")
        lines.append(f"{base}    cfg: {float(f['cfg'])},")
        lines.append(f"{base}    sampler: {js_str(f['sampler'])},")
        lines.append(f"{base}    scheduler: {js_str(f['scheduler'])},")
        lines.append(f"{base}  }},")
    return "\n".join(lines)


def render_options_html(options, selected):
    lines = []
    for mk, label in options:
        sel = " selected" if mk == selected else ""
        lines.append(f'                <option value="{mk}"{sel}>{html.escape(label)}</option>')
    return "\n".join(lines)


def render_paths_js(paths, base=HTML_INDENT):
    lines = [f"{base}const PATHS = {{"]
    for k in ("te", "adapter", "vae"):
        if k in paths:
            lines.append(f"{base}  {k}: {js_str(paths[k])},")
    lines.append(f"{base}}};")
    return "\n".join(lines)


def render_site_js(cfg, selected):
    """config.js 모드: 드롭다운 기본 선택·프롬프트 기본값·출처 링크를 데이터로 출력.
    (html 모드에선 이 값들이 option/textarea/footer HTML로 직접 들어갔다.)"""
    pd = cfg.get("prompt_defaults", {})
    lines = [f"const SELECTED_MODEL = {js_str(selected or '')};", "const PROMPT_DEFAULTS = {"]
    for k in ("positive", "negative"):
        if k in pd:
            lines.append(f"  {k}: {js_str(pd[k])},")
    lines += ["};", "const MODEL_SOURCES = ["]
    for m in cfg["models"]:
        if m.get("source"):
            lines.append(f"  {{ label: {js_str(m.get('label', m['key']))}, url: {js_str(m['source'])} }},")
    lines.append("];")
    lines.append(f"const CONVERT_URL = {js_str(cfg.get('convert_url') or '')};")
    # 변형별 생성 기본값 — 사용자 설정 모델(변환 결과 묶음)도 같은 값을 쓴다 (js/custom-ui.js)
    lines.append("const VARIANT_DEFAULTS = {")
    for v, d in cfg.get("defaults", {}).items():
        vals = ", ".join(
            f"{k}: {js_str(d[k]) if isinstance(d[k], str) else float(d[k]) if k == 'cfg' else d[k]}"
            for k in ("steps", "cfg", "sampler", "scheduler")
            if k in d
        )
        lines.append(f"  {v}: {{ {vals} }},")
    lines.append("};")
    return "\n".join(lines)


def render_footer_html(cfg):
    """source가 있는 모델만 출처 링크. 하나도 없으면 None(스킵)."""
    links = []
    for m in cfg["models"]:
        src = m.get("source")
        if src:
            label = m.get("label", m["key"])
            links.append(
                f'        <a href="{html.escape(src)}" target="_blank" '
                f'rel="noopener">{html.escape(label)}</a>'
            )
    # source가 하나도 없으면 빈 문자열을 반환해 마커 사이를 비운다
    # (None을 반환하면 마커를 안 건드려 이전 링크가 남는 버그가 됨).
    return "\n".join(links)


def replace_marker(text, name, new_inner):
    """<!-- NAME:START ... --> ... <!-- NAME:END --> 사이 또는
    // NAME:START ... // NAME:END 사이를 교체. 둘 다 지원. 멱등."""
    # HTML 주석형: START 주석의 들여쓰기를 재사용해 END 줄 정렬
    pat_html = re.compile(
        r"([ \t]*)(<!--\s*" + re.escape(name) + r":START.*?-->)"
        r".*?"
        r"([ \t]*)(<!--\s*" + re.escape(name) + r":END\s*-->)",
        re.DOTALL,
    )
    # JS 주석형
    pat_js = re.compile(
        r"([ \t]*)(//\s*" + re.escape(name) + r":START.*?\n)"
        r".*?"
        r"([ \t]*)(//\s*" + re.escape(name) + r":END)",
        re.DOTALL,
    )

    def sub_html(mo):
        indent, start_tag, end_indent, end_tag = mo.group(1), mo.group(2), mo.group(3), mo.group(4)
        return f"{indent}{start_tag}\n{new_inner}\n{end_indent}{end_tag}"

    def sub_js(mo):
        indent, start_tag, end_indent, end_tag = mo.group(1), mo.group(2), mo.group(3), mo.group(4)
        return f"{indent}{start_tag}{new_inner}\n{end_indent}{end_tag}"

    if pat_html.search(text):
        return pat_html.sub(sub_html, text), True
    if pat_js.search(text):
        return pat_js.sub(sub_js, text), True
    return text, False


def replace_textarea(text, tid, new_inner):
    pat = re.compile(
        r'(<textarea id="' + re.escape(tid) + r'">)(.*?)(</textarea)',
        re.DOTALL,
    )
    if pat.search(text):
        return pat.sub(lambda mo: mo.group(1) + "\n" + new_inner + mo.group(3), text), True
    return text, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yml")
    ap.add_argument("html")
    ap.add_argument("--check", action="store_true", help="적용하지 않고 변경 여부만 보고")
    args = ap.parse_args()

    with open(args.yml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.html, encoding="utf-8") as f:
        text = f.read()
    orig = text

    selected = cfg.get("selected")
    entries, options = build_model_entries(cfg)

    # selected 유효성
    keys = {mk for mk, _ in entries}
    if selected and selected not in keys:
        sys.exit(f"ERROR: selected '{selected}'가 생성된 모델 key에 없음. 가능: {sorted(keys)}")

    report = []

    if args.html.endswith(".js"):
        # config.js 모드 (모듈 분리 이후): 모든 배포 구성을 JS 데이터로. HTML은 건드리지 않는다.
        text, ok = replace_marker(text, "PATH_DEFAULTS", render_paths_js(cfg.get("paths", {}), base=""))
        report.append(("PATH_DEFAULTS", ok))
        text, ok = replace_marker(text, "MODELS", render_models_js(entries, base=""))
        report.append(("MODELS", ok))
        text, ok = replace_marker(text, "SITE_DEFAULTS", render_site_js(cfg, selected))
        report.append(("SITE_DEFAULTS", ok))
        return finish(args, text, orig, report, entries, selected)

    # ── 레거시 html 모드: 단일 index.html에 직접 주입 ──
    # MODELS
    text, ok = replace_marker(text, "MODELS", render_models_js(entries))
    report.append(("MODELS", ok))
    # options
    text, ok = replace_marker(text, "MODEL_OPTIONS", render_options_html(options, selected))
    report.append(("MODEL_OPTIONS", ok))
    # PATHS
    text, ok = replace_marker(text, "PATH_DEFAULTS", render_paths_js(cfg.get("paths", {})))
    report.append(("PATH_DEFAULTS", ok))
    # textarea
    pd = cfg.get("prompt_defaults", {})
    if "positive" in pd:
        text, ok = replace_textarea(text, "prompt", pd["positive"])
        report.append(("textarea#prompt", ok))
    if "negative" in pd:
        text, ok = replace_textarea(text, "negative", pd["negative"])
        report.append(("textarea#negative", ok))
    # footer (옵션)
    footer = render_footer_html(cfg)
    if footer is not None:
        text, ok = replace_marker(text, "FOOTER_SOURCES", footer)
        report.append(("FOOTER_SOURCES", ok))
    return finish(args, text, orig, report, entries, selected)


def finish(args, text, orig, report, entries, selected):
    print("주입 결과:")
    for name, ok in report:
        print(f"  {'OK ' if ok else 'SKIP(마커없음)'}  {name}")
    print(f"\n생성된 모델 ({len(entries)}): {', '.join(mk for mk, _ in entries)}")
    print(f"기본 선택: {selected}")

    if args.check:
        changed = text != orig
        print(f"\n--check: 변경 {'있음' if changed else '없음'} (파일 미수정)")
        return

    if text != orig:
        # newline="\n": Windows 텍스트 모드의 CRLF 변환 방지 (운영본이 CRLF였던 원인)
        with open(args.html, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"\n{args.html} 갱신 완료.")
    else:
        print("\n변경 없음.")


if __name__ == "__main__":
    main()
