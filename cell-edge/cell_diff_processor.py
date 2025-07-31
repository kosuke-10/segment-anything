import os
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from cell_diff_core import get_cell_diff_image

def process_cell_image_folder(input_dir, output_dir, scale_min, scale_max, scale_step,
                              rank_num, first_sensitive, second_sensitive, debug):
    os.makedirs(output_dir, exist_ok=True)
    result_list = []

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"読み込み失敗: {img_path}")
            continue

        try:
            diff_img, best_roll = get_cell_diff_image(
                img,
                scale_min=scale_min,
                scale_max=scale_max,
                scale_step=scale_step,
                rank_num=rank_num,
                first_sensitive=first_sensitive,
                second_sensitive=second_sensitive,
                debug=debug
            )

            out_path = os.path.join(output_dir, f"diff_{img_file}")
            cv2.imwrite(out_path, diff_img)

            result_list.append({
                "filename": img_file,
                "G_image_diff": best_roll
            })

        except Exception as e:
            print(f"エラー発生: {img_file}: {e}")

    df = pd.DataFrame(result_list)
    df.to_csv(os.path.join(output_dir, "G_image_diff_results.csv"), index=False)
    print("処理完了 ✔ 差分値を CSV に保存しました。")

def main():
    parser = argparse.ArgumentParser(description="細胞画像に対する差分処理")
    parser.add_argument('--input_dir', type=str, required=True, help="入力画像ディレクトリ")
    parser.add_argument('--output_dir', type=str, required=True, help="出力先ディレクトリ")
    parser.add_argument('--scale_min', type=int, default=0, help="roll の最小値")
    parser.add_argument('--scale_max', type=int, default=20, help="roll の最大値")
    parser.add_argument('--scale_step', type=int, default=100, help="近似関数用の補間ステップ数")
    parser.add_argument('--rank_num', type=int, default=6, help="多項式近似の次数")
    parser.add_argument('--first_sensitive', type=float, default=10, help="first curvature のしきい値")
    parser.add_argument('--second_sensitive', type=float, default=50, help="second curvature のしきい値")
    parser.add_argument('--debug', action='store_true', help="デバッグ画像やグラフを出力する")

    args = parser.parse_args()

    process_cell_image_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        scale_step=args.scale_step,
        rank_num=args.rank_num,
        first_sensitive=args.first_sensitive,
        second_sensitive=args.second_sensitive,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
