# /// script
# requires-python = ">=3.10"
# ///
"""태그 자동완성 데이터 변환: mengod-hf/danbooru-tags-kr → out/danbooru_ko.csv 형식.

원본: https://huggingface.co/datasets/mengod-hf/danbooru-tags-kr
      (danbooru-tags-backslash-kr-<날짜>.csv — 소괄호 앞에 역슬래시가 붙은 변형. 지금 페이지 태그 표기와 같다)
라이선스: 한글화 기여분은 CC BY-SA 4.0 (데이터셋의 LICENSE 파일). 변환본을 배포할 때도 같은 라이선스·출처를 따른다.

원본 한 줄:  tag,category,post_count,"[분류 > 하위] 설명. <키워드> 별칭1, 별칭2."
변환 결과:   tag,category,post_count,"[분류 > 하위] 설명. 키워드: 별칭1, 별칭2"
  - web/workers/tag-worker.js는 "키워드:" 뒤를 콤마로 나눠 한글 별칭으로 표시한다(그래서 표지만 바꾼다).
  - 끝의 마침표 하나는 마지막 별칭에 붙지 않게 뗀다.
  - 워커가 줄 단위로 나누므로 설명 안의 줄바꿈은 공백으로 바꾼다.
  - 나머지(태그·카테고리 번호·post_count·설명 본문)는 그대로 둔다. 정렬은 워커가 post_count로 한다.

사용:
  uv run tools/convert-tags-kr.py <원본.csv> <출력.csv>
"""
import csv
import sys

SRC_MARK = "<키워드>"
DST_MARK = "키워드:"


def convert_desc(desc: str) -> str:
    desc = " ".join(desc.split("\n")).strip()
    i = desc.rfind(SRC_MARK)
    if i < 0:
        return desc
    head = desc[:i].rstrip()
    kw = desc[i + len(SRC_MARK):].strip()
    if kw.endswith("."):
        kw = kw[:-1].rstrip()
    return f"{head} {DST_MARK} {kw}" if head else f"{DST_MARK} {kw}"


def main(src: str, dst: str) -> None:
    n = no_kw = bad = 0
    seen = set()
    with open(src, encoding="utf-8-sig", newline="") as fi, \
         open(dst, "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, lineterminator="\n")
        for row in csv.reader(fi):
            if len(row) != 4 or not row[2].strip().lstrip("-").isdigit() or not row[0]:
                bad += 1
                continue
            tag, cat, pc, desc = row
            if tag in seen:
                bad += 1
                continue
            seen.add(tag)
            if SRC_MARK not in desc:
                no_kw += 1
            w.writerow([tag, cat, pc.strip(), convert_desc(desc)])
            n += 1
    print(f"written {n} rows -> {dst} (no keyword marker: {no_kw}, skipped: {bad})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
