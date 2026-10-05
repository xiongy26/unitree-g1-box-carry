#!/usr/bin/env python3
"""按需下载 motions/ 运动轨迹数据（该目录不随仓库分发，首次使用需联网）。

数据以 GitHub Release 资产形式发布；本脚本仅使用 Python 标准库。

用法:
    python3 tools/fetch_motions.py [--base-url URL] [--dest DIR] [--force] [--list]

URL 解析优先级: --base-url > 环境变量 G1_MOTIONS_BASE_URL > DEFAULT_BASE_URL。
未配置任何来源时退出码 2 并打印配置提示，不发起任何请求。

行为:
    逐文件 GET <base-url>/<文件名>，先写 <文件名>.part 再原子改名到 motions/；
    目标已存在且未给 --force 时跳过（本地已有数据的用户零影响）。
    MOTION_SHA256 非空时逐文件校验，失败删除半成品并以退出码 1 结束。
"""
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# Release 发布后填入，例如:
#   https://github.com/<OWNER>/unitree-g1-box/releases/download/<TAG>/
DEFAULT_BASE_URL = ""

# 8 个轨迹文件，与 motions/ 外置前的 git 跟踪全集一一对应（契约 K12）。
MOTION_FILES = [
    "walk_meta.json",
    "manipulation.json",
    "carry_low_largebox.json",
    "carry_low_plasticbox.json",
    "carry_mid_largebox.json",
    "carry_high_largebox.json",
    "carrywalk_low_largebox.json",
    "carrywalk_low_plasticbox.json",
]

# 发布资产的 SHA256；发布后回填启用校验（为空 dict 时跳过校验）。
MOTION_SHA256 = {}

CHUNK_SIZE = 1 << 16


def config_hint(dest: Path) -> str:
    return (
        "未配置下载来源。三种配置方式（优先级从高到低）:\n"
        "  1. 命令行参数:  python3 tools/fetch_motions.py --base-url <URL>\n"
        "  2. 环境变量:    export G1_MOTIONS_BASE_URL=<URL>\n"
        "  3. 脚本内常量:  编辑 tools/fetch_motions.py 的 DEFAULT_BASE_URL\n"
        "URL 应指向包含以下文件的发布目录: " + " ".join(MOTION_FILES) + "\n"
        f"下载目标目录: {dest}"
    )


def fetch_one(url: str, dest_file: Path, expect_sha: str) -> None:
    """下载单个文件到 dest_file（经 .part 原子改名），非空 expect_sha 时校验。"""
    part_file = dest_file.with_suffix(dest_file.suffix + ".part")
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "unitree-g1-box-fetch/1.0"})
        with urllib.request.urlopen(request) as response, open(part_file, "wb") as out:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                out.write(chunk)
        actual_sha = digest.hexdigest()
        if expect_sha and actual_sha != expect_sha:
            raise IOError(f"SHA256 校验失败: 期望 {expect_sha}, 实际 {actual_sha}")
        part_file.replace(dest_file)
    except BaseException:
        part_file.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="按需下载 motions/ 运动轨迹数据")
    parser.add_argument("--base-url", default="", help="发布资产根 URL（优先级最高）")
    parser.add_argument("--dest", default="", help="下载目标目录（默认 <仓库根>/motions）")
    parser.add_argument("--force", action="store_true", help="目标已存在时也重新下载")
    parser.add_argument("--list", action="store_true", help="仅列出轨迹文件清单后退出")
    args = parser.parse_args()

    if args.list:
        for name in MOTION_FILES:
            print(name)
        return 0

    dest = Path(args.dest) if args.dest else Path(__file__).resolve().parents[1] / "motions"

    base_url = args.base_url or os.environ.get("G1_MOTIONS_BASE_URL", "") or DEFAULT_BASE_URL
    if not base_url:
        print(config_hint(dest), file=sys.stderr)
        return 2
    if not base_url.endswith("/"):
        base_url += "/"

    dest.mkdir(parents=True, exist_ok=True)
    succeeded = []
    for index, name in enumerate(MOTION_FILES, start=1):
        dest_file = dest / name
        if dest_file.exists() and not args.force:
            print(f"[{index}/{len(MOTION_FILES)}] 跳过（已存在）: {name}")
            succeeded.append(name)
            continue
        url = base_url + name
        expect_sha = MOTION_SHA256.get(name, "")
        try:
            fetch_one(url, dest_file, expect_sha)
        except Exception as exc:  # 网络、校验、写盘失败统一走失败报告
            print(f"[{index}/{len(MOTION_FILES)}] 失败: {name} ({url})\n  原因: {exc}", file=sys.stderr)
            print("已成功文件: " + (" ".join(succeeded) if succeeded else "（无）"), file=sys.stderr)
            print(config_hint(dest), file=sys.stderr)
            return 1
        print(f"[{index}/{len(MOTION_FILES)}] 完成: {name}")
        succeeded.append(name)

    print(f"全部 {len(succeeded)}/{len(MOTION_FILES)} 个文件就绪: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
