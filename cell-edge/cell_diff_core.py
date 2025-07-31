import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_cell_diff_image(color_image, scale_min=0, scale_max=20, scale_step=100, rank_num=6, first_sensitive=10, second_sensitive=50, debug=True):
    H, W, _ = color_image.shape
    B = color_image[:, :, 0].astype(np.float32)
    G = color_image[:, :, 1].astype(np.float32)

    sories = np.zeros(scale_max - scale_min)

    # Gを暗くしながら B-G をとる
    for roll in range(scale_min, scale_max):
        G_roll = G - roll
        diff_image = B - G_roll

        # 差分が正（B > G_roll）なピクセルをカウント（特徴あり）
        sories[roll - scale_min] = np.count_nonzero(B > G_roll)

        if debug:
            out = np.clip(diff_image, 0, 255).astype(np.uint8)
            cv2.imwrite(f'diff_image_roll_{roll}.png', out)

    x = np.arange(scale_min, scale_max)
    y = sories

    if debug:
        plt.plot(x, y, marker='o')
        plt.xlabel("G_image diff (roll)")
        plt.ylabel("Number of positive pixels (B > G_roll)")
        plt.title("Pixel count vs. roll")
        plt.show()

    # 多項式近似
    z = np.polyfit(x, y, rank_num)
    p = np.poly1d(z)

    xp = np.linspace(scale_min, scale_max, scale_step)
    yp = p(xp)

    # 曲率1
    dy = np.gradient(yp, xp)
    d2y = np.gradient(dy, xp)
    first_curvature = ((1 + dy**2)**1.5 / (np.abs(d2y) + 1e-20)) / 100000

    # 曲率2
    dy2 = np.gradient(first_curvature, xp)
    d2y2 = np.gradient(dy2, xp)
    second_curvature = ((1 + dy2**2)**1.5 / (np.abs(d2y2) + 1e-20))

    # second_curvatureの極小点を探す
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

    # 最終差分画像を出力
    G_final = G - G_image_diff
    final_diff = B - G_final
    final_diff_clipped = np.clip(final_diff, 0, 255).astype(np.uint8)

    if debug:
        cv2.imwrite(f'final_diff_image_roll_{G_image_diff}.png', final_diff_clipped)

    return final_diff_clipped, G_image_diff
