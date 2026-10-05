#!/usr/bin/env python3
# 一次性分析：从 motions/*.json 标定 lift_end / lower_start → motions/walk_meta.json
# （方案 docs/rl-walking-plan.md 5.2 / T2 / H6）。只读 motions/*.json，覆写 walk_meta.json。
#
# 标定口径（方案 5.2 校验断言）：
#   lift_end    = 搬运窗内 obj_z 首次达 0.85·峰值且其后 HOLD 帧保持、且 root_z ≥ 搬运窗均值 90%
#                 的最早帧（提起完成：箱已抬到位、躯干已直起）
#   lower_start = 对称：obj_z ≥0.85·峰值且其后 HOLD 帧全部低于的最晚帧（下放开始）
#
# 用法：python3 tools/pipeline/analyze_clips.py [--print-profile]
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CLIPS = {
    'largebox': {'low': 'carry_low_largebox', 'mid': 'carry_mid_largebox', 'high': 'carry_high_largebox'},
    'plasticbox': {'low': 'carry_low_plasticbox'},
}
PEAK_FRAC = 0.85
HOLD = 5  # 保持帧数


def calibrate(c):
    g, r = c['grasp_frame'], c['release_frame']
    window = list(range(g, min(r, c['T'] - 1) + 1))
    oz = lambda f: c['obj_pos'][f][2]
    rz = lambda f: c['root_pos'][f][2]
    peak = max(oz(f) for f in window)
    thr = PEAK_FRAC * peak
    root_mean = sum(rz(f) for f in window) / len(window)

    lift_end = None
    for i, f in enumerate(window):
        if oz(f) >= thr and all(oz(ff) >= thr for ff in window[i:i + HOLD]) and rz(f) >= root_mean * 0.9:
            lift_end = f
            break
    lower_start = None
    for i in range(len(window) - HOLD, -1, -1):
        f = window[i]
        if oz(f) >= thr and all(oz(ff) < thr for ff in window[i + 1:i + 1 + HOLD]):
            lower_start = f
            break
    return {'lift_end': lift_end, 'lower_start': lower_start, 'peak': peak, 'thr': thr,
            'root_mean': root_mean, 'grasp': g, 'release': r, 'T': c['T']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--print-profile', action='store_true', help='打印剖面与标定标记（人工核对表）')
    args = ap.parse_args()

    meta = {}
    for box_type, tiers in CLIPS.items():
        meta[box_type] = {}
        for tier, stem in tiers.items():
            path = ROOT / 'motions' / f'{stem}.json'
            if not path.exists():
                print(f'[skip] {stem}: 文件缺失')
                continue
            c = json.loads(path.read_text())
            r = calibrate(c)
            meta[box_type][tier] = {'lift_end': r['lift_end'], 'lower_start': r['lower_start']}
            status = 'ok' if (r['lift_end'] is not None and r['lower_start'] is not None) else '未标定'
            print(f'[{stem}] grasp={r["grasp"]} release={r["release"]} T={r["T"]} '
                  f'obj_peak={r["peak"]:.3f} thr={r["thr"]:.3f} root_mean={r["root_mean"]:.3f} '
                  f'lift_end={r["lift_end"]} lower_start={r["lower_start"]} → {status}')

            if args.print_profile:
                marks = {r['lift_end']: ' ←lift_end', r['lower_start']: ' ←lower_start',
                         r['grasp']: ' ←grasp', r['release']: ' ←release'}
                for f in range(r['grasp'], min(r['release'] + 20, c['T']), 4):
                    tag = marks.get(f, '')
                    print(f'    f={f:3d} root_z={c["root_pos"][f][2]:.3f} obj_z={c["obj_pos"][f][2]:.3f}{tag}')

    out = ROOT / 'motions' / 'walk_meta.json'
    # MERO-10：walk_meta.json 已升 v3（version + carry 节为手工维护的选片声明）。
    # 本工具重写 v2 帧域条目时必须保留这些键，否则静默降级链条失去 mocap 声明。
    prev = {}
    if out.exists():
        try:
            old = json.loads(out.read_text())
            prev = {k: old[k] for k in ('version', 'carry') if k in old}
        except (json.JSONDecodeError, OSError):
            pass
    merged = {'version': prev.get('version', 3), 'carry': prev.get('carry', {}), **meta}
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + '\n')
    print(f'写出 {out}（保留 v3 键: version/carry）')


if __name__ == '__main__':
    main()
