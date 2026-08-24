#!/usr/bin/env python
"""Step 5: grade one slide end to end - the path a deployment actually takes.

Every earlier step processes a cohort. This processes one slide, the way a
commercial user actually calls it: upload a WSI, get back a grade and an
attention overlay. No separate feature-extraction step, no thumbnail file -
both are derived from the slide itself, inside pathgrade.inference.grade_slide.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathgrade.inference import grade_slide


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("slide", help="path to a whole-slide image (.svs, .ndpi, ...)")
    p.add_argument("--run-dir", required=True,
                   help="a trained run directory, e.g. runs/asmil-ord-hoptimus0")
    p.add_argument("--out", default="explained.png", help="where to save the attention overlay")
    p.add_argument("--device", default=None, help="cuda | xla | cpu | auto")
    p.add_argument("--max-patches", type=int, default=8000)
    args = p.parse_args()

    prediction, overlay = grade_slide(
        args.slide, args.run_dir, device=args.device, max_patches=args.max_patches,
    )

    print(prediction.summary())
    print("\ntop attended regions:")
    for r in prediction.top_regions(5):
        print(f"  #{r['rank']}  ({r['x']:>7}, {r['y']:>7})  "
              f"attention {r['attention']:.4f}  ({r['share_of_slide']:.1f}x average)")

    overlay.save(args.out)
    print(f"\noverlay written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
