#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_models.py — model-list.yml → index.html 정적 주입기.

index.html에 박힌 마커 사이를 yml 내용으로 교체한다. 결과는 정적 HTML
(런타임 fetch 없음). 마커 밖 코드(custom 항목 등)는 건드리지 않는다.

사용법:
  python gen_models.py model-list.yml web/index.html
  python gen_models.py model-list.yml web/index.html --check   # 변경 미적용, diff만

교체 대상:
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


def render_models_js(entries):
    lines = []
    for mk, f in entries:
        lines.append(f"        {mk}: {{")
        lines.append(f"          label: {js_str(f['label'])},")
        lines.append(f"          dit: {js_str(f['dit'])},")
        lines.append(f"          ditLora: {js_str(f['ditLora'])},")
        lines.append(f"          steps: {f['steps']},")
        lines.append(f"          cfg: {float(f['cfg'])},")
        lines.append(f"          sampler: {js_str(f['sampler'])},")
        lines.append(f"          scheduler: {js_str(f['scheduler'])},")
        lines.append("        },")
    return "\n".join(lines)


def render_options_html(options, selected):
    lines = []
    for mk, label in options:
        sel = " selected" if mk == selected else ""
        lines.append(f'                <option value="{mk}"{sel}>{html.escape(label)}</option>')
    return "\n".join(lines)


def render_paths_js(paths):
    lines = ["      const PATHS = {"]
    for k in ("te", "adapter", "vae"):
        if k in paths:
            lines.append(f"        {k}: {js_str(paths[k])},")
    lines.append("      };")
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
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n{args.html} 갱신 완료.")
    else:
        print("\n변경 없음.")


if __name__ == "__main__":
    main()
