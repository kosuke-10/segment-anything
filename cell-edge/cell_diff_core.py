import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_cell_diff_image(
    color_image,
    scale_min=0,
    scale_max=20,
    scale_step=100,
    rank_num=6,
    first_sensitive=10,
    second_sensitive=50,
    debug=True,
    diff_save_dir=None,
    img_basename="image"
):
    """
    入力画像からB-G差分画像を生成し、最適なroll値での差分画像とその値を返す。
    debug=Trueなら経過画像やグラフも保存。
    save_dir: 画像保存先ディレクトリ（Noneならカレント）
    prefix: 保存ファイル名の先頭につける文字列
    """
    H, W, _ = color_image.shape
    B = color_image[:, :, 0].astype(np.float32)
    G = color_image[:, :, 1].astype(np.float32)

    sories = np.zeros(scale_max - scale_min)

    for roll in range(scale_min, scale_max):
        G_roll = G - roll
        diff_image = B - G_roll
        sories[roll - scale_min] = np.count_nonzero(B > G_roll)

        # roll毎の保存は不要ならコメントアウト
        # if debug:
        #     out = np.clip(diff_image, 0, 255).astype(np.uint8)
        #     if diff_save_dir:
        #         os.makedirs(diff_save_dir, exist_ok=True)
        #         cv2.imwrite(os.path.join(diff_save_dir, f"{img_basename}_diff_roll_{roll}.png"), out)
        #     else:
        #         cv2.imwrite(f"{img_basename}_diff_roll_{roll}.png", out)

    x = np.arange(scale_min, scale_max)
    y = sories

    if debug:
        plt.plot(x, y, marker='o')
        plt.xlabel("G_image diff (roll)")
        plt.ylabel("Number of positive pixels (B > G_roll)")
        plt.title("Pixel count vs. roll")
        plt.show()

    z = np.polyfit(x, y, rank_num)
    p = np.poly1d(z)
    xp = np.linspace(scale_min, scale_max, scale_step)
    yp = p(xp)
    dy = np.gradient(yp, xp)
    d2y = np.gradient(dy, xp)
    first_curvature = ((1 + dy**2)**1.5 / (np.abs(d2y) + 1e-20)) / 100000
    dy2 = np.gradient(first_curvature, xp)
    d2y2 = np.gradient(dy2, xp)
    second_curvature = ((1 + dy2**2)**1.5 / (np.abs(d2y2) + 1e-20))
    curvature_diff = np.diff(second_curvature)
    sdiff_sign = ((curvature_diff[:-1] * curvature_diff[1:]) < 0) & (curvature_diff[:-1] < 0)

    index = 0
    find_flag = False
    for i in range(len(sdiff_sign)):
        if sdiff_sign[i] and first_curvature[i] < first_sensitive and second_curvature[i] < second_sensitive:
            index = i + 1
            find_flag = True
            break

    if find_flag:
        G_image_diff = round(index * (scale_max - scale_min)/scale_step + scale_min)
    else:
        G_image_diff = scale_max

    G_final = G - G_image_diff
    final_diff = B - G_final
    final_diff_clipped = np.clip(final_diff, 0, 255).astype(np.uint8)

    # 最終差分画像だけ保存
    if diff_save_dir:
        os.makedirs(diff_save_dir, exist_ok=True)
        fname = f"{img_basename}_diff_final_{G_image_diff}.png"
        cv2.imwrite(os.path.join(diff_save_dir, fname), final_diff_clipped)
    else:
        cv2.imwrite(f"{img_basename}_diff_final_{G_image_diff}.png", final_diff_clipped)

    return final_diff_clipped, G_image_diff


def extract_points_from_diff(diff_img, threshold=90, min_area=10):
    """
    差分画像からSAM用ポイントを抽出
    """
    # 2値化
    _, mask = cv2.threshold(diff_img, threshold, 255, cv2.THRESH_BINARY)
    # 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append([cx, cy])
    return np.array(points)

def extract_all_points_from_diff(diff_img, threshold=30):
    """
    差分画像から2値化し、白領域の全ピクセル座標をSAM用ポイントとして抽出
    """
    _, mask = cv2.threshold(diff_img, threshold, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(mask == 255)
    points = np.stack([xs, ys], axis=1)
    return points

def extract_contour_points(diff_img, threshold=90, min_area=50, points_per_contour=10):
    _, mask = cv2.threshold(diff_img, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        cnt = cnt.squeeze()
        if len(cnt.shape) != 2 or cnt.shape[0] < points_per_contour:
            continue
        idxs = np.linspace(0, len(cnt)-1, points_per_contour, dtype=int)
        for idx in idxs:
            points.append(cnt[idx])
    return np.array(points)