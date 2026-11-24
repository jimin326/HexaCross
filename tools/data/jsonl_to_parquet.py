"""collect_pomdp_data.py가 만든 .jsonl.gz를 Parquet(zstd)로 바꾼다 (DATA_FORMAT.md 5절 "HF 업로드 때 Parquet로 변환").

값은 JSONL과 같고 모양만 바꾼다.
- 컬럼 이름 = JSONL 키 경로 (점으로 이음): `state.ego_x`, `state.tl_color_10`, `event.collided`, `action.accel_raw` …
- `observation`, `next_observation`, `action_raw`: 숫자 리스트 (float32)
- `state.vehicles`, `detected_vehicles`: 구조체 리스트 (JSONL 항목 그대로)
- `next_observation`, `next_state.*`: JSONL처럼 에피소드 마지막 스텝에만 값이 있고 나머지 행은 null
- 숫자형: 실수 float32 (좌표·보상 포함), 정수 int32, 참거짓 bool. 문자열 컬럼은 없다.
- 컬럼 목록은 첫 행에서 정하고, 그 밖의 키가 나오면 멈춘다 (조용히 버리지 않음).

usage: python tools/data/jsonl_to_parquet.py IN.jsonl.gz OUT.parquet
"""
import gzip, json, os, sys
import pyarrow as pa
import pyarrow.parquet as pq

ROWGROUP = int(os.environ.get("ROWGROUP_ROWS", "50000"))
NUM_LISTS = {"observation", "next_observation", "action_raw"}


def scalar_type(v):
    if isinstance(v, bool):
        return pa.bool_()
    if isinstance(v, int):
        return pa.int32()
    return pa.float32()


def struct_type(items):
    fields = {}
    for it in items:
        for k, v in it.items():
            t = scalar_type(v)
            if k not in fields or (fields[k] == pa.int32() and t == pa.float32()):
                fields[k] = t
    return pa.list_(pa.struct(list(fields.items())))


def flatten(rec, prefix="", out=None):
    out = {} if out is None else out
    for k, v in rec.items():
        name = prefix + k
        if isinstance(v, dict):
            flatten(v, name + ".", out)
        else:
            out[name] = v
    return out


def build_schema(flat, sample_lists):
    fields = []
    for k, v in flat.items():
        if k in NUM_LISTS:
            fields.append((k, pa.list_(pa.float32())))
        elif isinstance(v, list):
            fields.append((k, sample_lists[k]))
        elif v is None:
            fields.append((k, pa.float32()))
        else:
            fields.append((k, scalar_type(v)))
    # 에피소드 마지막 스텝에만 있는 다음 관측·다음 상태 (state.* → next_state.*)
    fields.append(("next_observation", pa.list_(pa.float32())))
    for k, t in list(fields):
        if k.startswith("state."):
            fields.append(("next_" + k, t))
    return pa.schema(fields)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    # 구조체 리스트의 필드 타입은 앞쪽 몇 천 행에서 모은다 (정수/실수 섞이면 실수)
    sample = {}
    with gzip.open(src, "rt", encoding="utf-8") as f:
        first = None
        for i, line in enumerate(f):
            r = flatten(json.loads(line))
            first = first or r
            for k, v in r.items():
                if isinstance(v, list) and k not in NUM_LISTS and v and isinstance(v[0], dict):
                    sample.setdefault(k, []).extend(v[:5])
            if i >= 3000:
                break
    for k, v in first.items():
        if isinstance(v, list) and k not in NUM_LISTS:
            sample.setdefault(k, [])
    lists = {k: (struct_type(v) if v else pa.list_(pa.struct([]))) for k, v in sample.items()}
    schema = build_schema(first, lists)
    names = schema.names
    tmp = dst + ".tmp"
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    cols = {n: [] for n in names}
    rows = 0

    def flush():
        nonlocal cols
        if not cols[names[0]]:
            return
        writer.write_table(pa.table({n: pa.array(cols[n], type=schema.field(n).type) for n in names}, schema=schema))
        cols = {n: [] for n in names}

    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            r = flatten(json.loads(line))
            extra = set(r) - set(names)
            if extra:
                raise SystemExit(f"{src}: 컬럼 목록에 없는 키 {sorted(extra)[:5]} (행 {rows})")
            for n in names:
                cols[n].append(r.get(n))
            rows += 1
            if len(cols[names[0]]) >= ROWGROUP:
                flush()
    flush()
    writer.close()
    os.replace(tmp, dst)
    print(f"{src} → {dst}: {rows} 행, 컬럼 {len(names)}개, {os.path.getsize(dst) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
