"""다 쓴 Parquet 샤드를 Hugging Face 데이터셋 저장소에 올린다 (수집과 동시에 돌려도 됨).

usage: python tools/data/upload_hf.py DATA_DIR REPO_ID PATH_IN_REPO [--watch]
  예: python tools/data/upload_hf.py data/v2/loopB_RL hoseob0902/HexaCross-dataset v2/loopB_RL --watch
- 이름이 shard_*.parquet인 파일만 올린다 (.tmp는 아직 쓰는 중이라 건너뜀).
- 올린 파일은 DATA_DIR/uploaded.json에 적어 두고 다시 올리지 않는다.
- --watch: 수집이 끝날 때까지(DATA_DIR/progress_w*.json의 행 수 합이 목표에 닿고 .tmp가 없을 때) 1분마다 새 샤드를 찾는다.
- 토큰은 HF_TOKEN 환경 변수 또는 huggingface-cli login.
"""
import glob, json, os, sys, time
from huggingface_hub import HfApi

d, repo, prefix = sys.argv[1], sys.argv[2], sys.argv[3].strip("/")
watch = "--watch" in sys.argv
target = int(os.environ.get("TARGET_ROWS", "10000000"))
api = HfApi(token=os.environ.get("HF_TOKEN"))
log_path = os.path.join(d, "uploaded.json")
done = json.load(open(log_path)) if os.path.exists(log_path) else {}


def collected():
    return sum(json.load(open(p))["rows"] for p in glob.glob(os.path.join(d, "progress_w*.json")))


while True:
    for f in sorted(glob.glob(os.path.join(d, "shard_*.parquet"))):
        name = os.path.basename(f)
        if name in done:
            continue
        for k in range(5):
            try:
                api.upload_file(path_or_fileobj=f, path_in_repo=f"{prefix}/{name}", repo_id=repo, repo_type="dataset",
                                commit_message=f"add {prefix}/{name}")
                break
            except Exception as e:
                print("retry", name, k, repr(e)[:200], flush=True)
                time.sleep(30 * (k + 1))
        else:
            print("FAILED", name, flush=True)
            continue
        done[name] = os.path.getsize(f)
        json.dump(done, open(log_path, "w"), indent=1)
        print(time.strftime("%H:%M:%S"), "uploaded", name, f"{len(done)} files, {sum(done.values()) / 1e9:.2f} GB", flush=True)
    if not watch:
        break
    if collected() >= target and not glob.glob(os.path.join(d, "*.tmp")) and \
            all(os.path.basename(f) in done for f in glob.glob(os.path.join(d, "shard_*.parquet"))):
        break
    time.sleep(60)
print("UPLOAD DONE", len(done), "files", flush=True)
