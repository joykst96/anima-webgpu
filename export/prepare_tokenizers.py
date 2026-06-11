"""
토크나이저 자가호스팅 준비: HF에서 받아 transformers.js가 읽는 형식
(tokenizer.json fast 포맷)으로 로컬 디렉토리에 저장.

웹페이지는 tokenizers/qwen3, tokenizers/t5 경로를 기본으로 찾고,
없으면 HF Hub로 폴백한다. 이 스크립트 산출물을 페이지와 같은 서버에
올리면 HF 의존이 사라져 완전 자급자족 배포가 됨.

보너스: 파이썬 검증에서 쓰던 진짜 t5-v1_1 토크나이저를 fast로 변환해
저장하므로, 웹에서도 flan-t5 대체품 대신 원본 토크나이저를 쓰게 됨.

사용 예 (portable 루트에서):
  python export/prepare_tokenizers.py --out tokenizers
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tokenizers", help="출력 루트 디렉토리")
    args = p.parse_args()

    from transformers import AutoTokenizer

    targets = [
        ("qwen3", "Qwen/Qwen3-0.6B-Base", "qwen3"),
        ("t5", "google/t5-v1_1-xxl", "t5"),
    ]
    for local_name, hf_id, model_type in targets:
        d = os.path.join(args.out, local_name)
        os.makedirs(d, exist_ok=True)
        print(f"[tok] {hf_id} → {d}")
        tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
        if not tok.is_fast:
            raise RuntimeError(
                f"{hf_id}의 fast 변환 실패 — transformers.js는 tokenizer.json"
                "(fast 포맷)이 필요. tokenizers/sentencepiece 패키지 확인.")
        tok.save_pretrained(d)
        # transformers.js가 config.json을 조회하는 경우 대비한 스텁
        cfg_path = os.path.join(d, "config.json")
        if not os.path.exists(cfg_path):
            with open(cfg_path, "w") as f:
                json.dump({"model_type": model_type}, f)
        files = os.listdir(d)
        assert "tokenizer.json" in files, f"tokenizer.json 누락: {files}"
        print(f"  저장됨: {', '.join(sorted(files))}")

    print(f"[done] '{args.out}/' 디렉토리를 index.html과 같은 위치에 둘 것.")


if __name__ == "__main__":
    main()
