import os
import argparse
import cv2
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from cell_diff_core import get_cell_diff_image, extract_points_from_diff, extract_all_points_from_diff, extract_contour_points

# segment_anythingのパスを追加（親ディレクトリにある場合）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segment_anything import sam_model_registry, SamPredictor

def overlay_mask_on_image(image, mask, color=(0, 0, 255), alpha=0.4):
    """元画像とマスクを重ねて半透明で合成"""
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = color
    cv2.addWeighted(mask_colored, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

def process_cell_image_folder(input_dir, diff_dir, mask_bin_dir, mask_sam_dir, overlay_dir, sam_checkpoint, model_type,
                             scale_min, scale_max, scale_step, rank_num,
                             first_sensitive, second_sensitive, threshold, min_area, debug):
    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(mask_bin_dir, exist_ok=True)  # 2値化マスク保存用
    os.makedirs(mask_sam_dir, exist_ok=True)  # SAMマスク保存用
    os.makedirs(overlay_dir, exist_ok=True)   # オーバーレイ画像保存用
    result_list = []

    # SAMセットアップ
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to('cuda')
    predictor = SamPredictor(sam)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"読み込み失敗: {img_path}")
            continue

        try:
            basename = os.path.splitext(img_file)[0]
            diff_img, best_roll = get_cell_diff_image(
                img,
                scale_min=scale_min,
                scale_max=scale_max,
                scale_step=scale_step,
                rank_num=rank_num,
                first_sensitive=first_sensitive,
                second_sensitive=second_sensitive,
                debug=debug,
                diff_save_dir=diff_dir,
                img_basename=basename
            )

            # --- ここから大津の2値化 ---
            otsu_thresh, mask_bin = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_bin_path = os.path.join(mask_bin_dir, f"{basename}_mask_bin.png")
            cv2.imwrite(mask_bin_path, mask_bin)

            # ポイント抽出（全ピクセルから最大100点）
            points = extract_all_points_from_diff(diff_img, threshold=int(otsu_thresh))
            max_points = 50
            if len(points) > max_points:
                idx = np.random.choice(len(points), max_points, replace=False)
                points = points[idx]
            if len(points) == 0:
                print(f"ポイント抽出失敗: {img_file}")
                continue

            # SAM推論
            predictor.set_image(img, image_format="BGR")
            point_labels = np.ones(len(points))
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                multimask_output=True
            )
            best_mask = masks[np.argmax(scores)]
            mask_sam_path = os.path.join(mask_sam_dir, f"{basename}_sam_mask.png")
            cv2.imwrite(mask_sam_path, (best_mask * 255).astype(np.uint8))

            # オーバーレイ画像の保存
            overlay = overlay_mask_on_image(img, (best_mask * 255).astype(np.uint8), color=(0, 0, 255), alpha=0.4)
            overlay_path = os.path.join(overlay_dir, f"{basename}_overlay.png")
            cv2.imwrite(overlay_path, overlay)

            result_list.append({
                "filename": img_file,
                "G_image_diff": best_roll,
                "otsu_thresh": otsu_thresh,
                "num_points": len(points)
            })

        except Exception as e:
            print(f"エラー発生: {img_file}: {e}")

    df = pd.DataFrame(result_list)
    df.to_csv(os.path.join(mask_sam_dir, "sam_mask_results.csv"), index=False)
    print("処理完了 ✔ 2値化マスク・SAMマスク・オーバーレイ画像・CSVを保存しました。")

def main():
    parser = argparse.ArgumentParser(description="細胞画像→差分→2値化マスク→SAMマスク→オーバーレイ画像一貫処理")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--diff_dir', type=str, required=True)
    parser.add_argument('--mask_bin_dir', type=str, required=True, help="2値化マスク保存先")
    parser.add_argument('--mask_sam_dir', type=str, required=True, help="SAMマスク保存先")
    parser.add_argument('--overlay_dir', type=str, required=True, help="オーバーレイ画像保存先")
    parser.add_argument('--sam_checkpoint', type=str, required=True)
    parser.add_argument('--model_type', type=str, default="vit_h")
    parser.add_argument('--scale_min', type=int, default=0)
    parser.add_argument('--scale_max', type=int, default=20)
    parser.add_argument('--scale_step', type=int, default=100)
    parser.add_argument('--rank_num', type=int, default=6)
    parser.add_argument('--first_sensitive', type=float, default=10)
    parser.add_argument('--second_sensitive', type=float, default=50)
    parser.add_argument('--threshold', type=int, default=30)
    parser.add_argument('--min_area', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    process_cell_image_folder(
        input_dir=args.input_dir,
        diff_dir=args.diff_dir,
        mask_bin_dir=args.mask_bin_dir,
        mask_sam_dir=args.mask_sam_dir,
        overlay_dir=args.overlay_dir,
        sam_checkpoint=args.sam_checkpoint,
        model_type=args.model_type,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        scale_step=args.scale_step,
        rank_num=args.rank_num,
        first_sensitive=args.first_sensitive,
        second_sensitive=args.second_sensitive,
        threshold=args.threshold,
        min_area=args.min_area,
        debug=args.debug
    )

if __name__ == "__main__":
    main()