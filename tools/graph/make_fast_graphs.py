# config.js의 모든 DiT 그래프(8모델 × base/turbo × 일반/LoRA 슬롯 = 32개)에 그래프 후처리를 적용해 배포용 파일을 만든다.
# 운영 서버에서 원본 그래프 .onnx만 HTTP GET으로 받는다(가중치 샤드는 받지 않음 — 결과물이 원본 샤드를 그대로 참조).
#   <그래프>_x.onnx    : 표준 + cross-attention 패딩 압축 입력(xattn_compact.py). 운영 nginx가 .onnx를 1년 immutable로
#                        캐시시키므로 같은 이름 교체는 기존 사용자에게 반영되지 않는다 → 반드시 새 이름.
#                        패딩 압축을 끄면(512토큰 + bias 0) 종전 그래프와 같은 결과 (골든 일치로 검증).
#   <그래프>_f16.onnx  : 위 + 블록 내부 fp16(fp16_zones.py --fast) — 페이지 '빠르게' 모드
# 결과는 out-local/에 같은 경로로 저장 → devserver --local-out으로 검증, 운영 out/dit/에 그대로 업로드.
#
#   python -I tools/graph/make_fast_graphs.py --upstream <가중치 서버 http://host:port> [--only anima_turbo]
#   (--upstream이 없으면 환경변수 ANIMA_UPSTREAM)
import os, re, subprocess, sys, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
upstream = args[args.index("--upstream") + 1] if "--upstream" in args else os.environ.get("ANIMA_UPSTREAM")
if not upstream:
    sys.exit("--upstream <가중치 서버> 또는 환경변수 ANIMA_UPSTREAM 필요")
only = args[args.index("--only") + 1] if "--only" in args else None

config = open(os.path.join(ROOT, "web", "js", "config.js"), encoding="utf-8").read()
paths = []
for m in re.finditer(r"^\s{2}(\w+): \{(.*?)^\s{2}\},", config, re.S | re.M):
    if only and m.group(1) != only:
        continue
    for p in re.findall(r'(?:dit|ditLora): "([^"]+\.onnx)"', m.group(2)):
        paths.append(p)


def run(script, src, dst, *extra):
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, script), src, dst, *extra],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise SystemExit(f"{script} {src}: 실패\n{r.stdout}{r.stderr}")


cache = os.path.join(ROOT, "out-local", ".src")   # 원본(운영) 그래프 — 덮어쓰지 않는다
os.makedirs(cache, exist_ok=True)
for p in paths:
    src = os.path.join(cache, os.path.basename(p))
    if not os.path.exists(src):
        urllib.request.urlretrieve(f"{upstream}/{p}", src)
    std = os.path.join(ROOT, "out-local", p[len("out/"):]).replace(".onnx", "_x.onnx")
    fast = std.replace("_x.onnx", "_f16.onnx")
    os.makedirs(os.path.dirname(std), exist_ok=True)
    run("xattn_compact.py", src, std)
    run("fp16_zones.py", std, fast, "--fast")
    print(f"{p} → {os.path.relpath(std, ROOT)} ({os.path.getsize(std) / 2**20:.1f} MB), _f16 ({os.path.getsize(fast) / 2**20:.1f} MB)")
print(f"{len(paths)}개 × 2 완료")
