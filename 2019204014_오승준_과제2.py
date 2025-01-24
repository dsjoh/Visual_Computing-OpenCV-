import numpy as np
import cv2
from matplotlib import pyplot as plt


# Plot 시각화 함수
def plot_img(rows_cols, index, img, title):
    plt.subplot(rows_cols[0], rows_cols[1], index)
    plt.imshow(img[...,::-1])
    plt.axis('off'), plt.title(title)


# Gaussian Pyramids 생성 함수
def generate_gaussian_pyramid(img, levels):
    GP = [img]
    for _ in range(1, levels):
        img = cv2.pyrDown(img)
        GP.append(img)
    return GP


# Laplacian Pyramids 생성 함수
def generate_laplacian_pyramid(GP):
    levels = len(GP)
    LP = []
    for i in range(levels - 1, 0, -1):
        img_higher = cv2.pyrUp(GP[i], None, GP[i - 1].shape[:2][::-1])
        img_lap = cv2.subtract(GP[i - 1], img_higher)
        LP.append(img_lap)
    LP.reverse()
    return LP


# Pyramids Compsition 시각화 함수
def generate_pyramid_composition_image(Pimgs):
    levels = len(Pimgs)
    rows, cols = Pimgs[0].shape[:2]
    composite_image = np.zeros((rows, cols + int(cols / 2 + 0.5), 3), dtype=Pimgs[0].dtype)
    composite_image[:rows, :cols, :] = Pimgs[0]
    i_row = 0
    for p in Pimgs[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows
    return composite_image


# 피라미드 시각화 함수
def visualize_combined_pyramid(n, pyramid, title):
    max_width = max([img.shape[1] for img in pyramid])
    max_height = max([img.shape[0] for img in pyramid])
    n_levels = len(pyramid)

    plt.figure(n, figsize=(8, 8))

    # 독립적으로 중앙에 배치된 시각화
    for i, img in enumerate(pyramid):
        canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        y_offset = (max_height - img.shape[0]) // 2
        x_offset = (max_width - img.shape[1]) // 2
        canvas[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img

        plt.subplot(2, n_levels, i + 1)
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title(f'{title} Level {i+1}')
        plt.axis('off')

    # 연속적으로 배치된 시각화
    for i in range(n_levels):
        plt.subplot(2, n_levels, n_levels + i + 1)
        plt.imshow(cv2.cvtColor(pyramid[i], cv2.COLOR_BGR2RGB))
        plt.title(f'{title} Level {i+1}')
        plt.axis('off')


# 피라미드 Stitch 함수
def stitch_pyramid_levels(LP_eye, LP_hand, x_offset, y_offset):
    stitched_pyramid = []
    divider = 1
    blend_ratio=0.9
    
    for i in range(len(LP_hand)):
        # 현재 레벨에 맞게 위치 및 크기 조정
        current_x_offset = x_offset // divider
        current_y_offset = y_offset // divider
        current_eye_height = 57 // divider
        current_eye_width = 130 // divider
        eye_x = 23 // divider
        eye_y = 86 // divider

        # 잘라낼 눈 이미지의 크기는 이전에 정의된 크기를 그대로 사용
        cropped_eye = LP_eye[i][eye_y:eye_y + current_eye_height, eye_x:eye_x + current_eye_width]

        # 손 이미지의 해당 위치에 눈 이미지 합성
        stitched = LP_hand[i].copy()
        blended_region = cropped_eye * blend_ratio + stitched[current_y_offset:current_y_offset + current_eye_height, current_x_offset:current_x_offset + current_eye_width] * (1 - blend_ratio)
        stitched[current_y_offset:current_y_offset + current_eye_height, current_x_offset:current_x_offset + current_eye_width] = blended_region

        stitched_pyramid.append(stitched)

        # 다음 레벨을 위해 divider 증가
        divider *= 2

    return stitched_pyramid



# 눈 이미지 불러오기
eye_img = cv2.imread('2019204014_eye.jpg')
eye_img_rgb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)  # RGB 변환


# 손 이미지 불러오기
hand_img = cv2.imread('2019204014_hand.jpg')
hand_img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)  # RGB 변환


# 눈 이미지 Crop
x = 600       # 시작점 x 좌표
y = 1000      # 시작점 y 좌표 
width = 1000   # 너비
height = 1000  # 높이
eye = eye_img[y:y+height, x:x+width]  # ROI 
eye = cv2.resize(eye, (100*2, 100*2)) # Resize
eye_rgb = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)  # RGB 변환


# 손 이미지 resize (640, 480)
hand = cv2.resize(hand_img, (640, 480))
hand_rgb = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)  # RGB 변환


# Gaussian Pyramids

# 설정할 피라미드 층수
n_levels = 5

# 눈 이미지의 가우시안 피라미드 생성
eye_gaussian_pyramid = generate_gaussian_pyramid(eye, n_levels)

# 손 이미지의 가우시안 피라미드 생성
hand_gaussian_pyramid = generate_gaussian_pyramid(hand, n_levels)


# laplacian Pyramids

# 눈 이미지의 라플라시안 피라미드 생성
eye_laplacian_pyramid = generate_laplacian_pyramid(eye_gaussian_pyramid)

# 손 이미지의 라플라시안 피라미드 생성
hand_laplacian_pyramid = generate_laplacian_pyramid(hand_gaussian_pyramid)


# Combined Pyramids Image 

# 손의 가우시안 & 라플라시안 피라미드 합성 이미지
hand_combine_g = generate_pyramid_composition_image(hand_gaussian_pyramid)
hand_combine_l = generate_pyramid_composition_image(hand_laplacian_pyramid)

# 눈의 가우시안 & 라플라시안 피라미드 합성 이미지
eye_combine_g = generate_pyramid_composition_image(eye_gaussian_pyramid)
eye_combine_l = generate_pyramid_composition_image(eye_laplacian_pyramid)

# combined pyramid 이미지들에 대해 RGB 변환
hand_combine_g = cv2.cvtColor(hand_combine_g, cv2.COLOR_BGR2RGB)  # RGB 변환
hand_combine_l = cv2.cvtColor(hand_combine_l, cv2.COLOR_BGR2RGB)
eye_combine_g = cv2.cvtColor(eye_combine_g, cv2.COLOR_BGR2RGB)
eye_combine_l = cv2.cvtColor(eye_combine_l, cv2.COLOR_BGR2RGB)


# 눈 이미지를 붙일 위치 지정 (x,y)
x_offset = 270
y_offset = 300

# 각각의 피라미드 이미지를 합성 (Stitched)
blended_G = stitch_pyramid_levels(eye_gaussian_pyramid, hand_gaussian_pyramid, x_offset, y_offset)
blended_L = stitch_pyramid_levels(eye_laplacian_pyramid, hand_laplacian_pyramid, x_offset, y_offset)


# 가우시안 피라미드의 가장 작은 레벨 이미지를 시작점으로 설정
blended_img = blended_G[-1]  # 가장 높은 level
cv2.imshow("Initial Level", blended_img)

for i in range(len(blended_L) - 1, -1, -1):
    # pyrup 적용한 gaussian pyramid 이미지와 다음 level의 Laplacian pyramid 이미지를 add
    blended_img = cv2.pyrUp(blended_img, dstsize=blended_L[i].shape[1::-1])
    blended_img = cv2.add(blended_img, blended_L[i])

    # 각 레벨의 합성 이미지를 별도의 창으로 표시 
    if (i == 0) : 
        window_name = "Blended Image (level 0)"
    else :
        window_name = f"Level {i}"        # level 0 은 최종 합성 이미지
    cv2.imshow(window_name, blended_img)


# 채널 분할 후 CLAHE
lab = cv2.cvtColor(blended_img, cv2.COLOR_BGR2LAB)

# LAB 색상 공간을 채널별로 분리
l, a, b = cv2.split(lab)

# CLAHE 객체 생성
clahe = cv2.createCLAHE(clipLimit=0.6, tileGridSize=(2, 2))

# L 채널(밝기)에 CLAHE 적용
l_clahe = clahe.apply(l)

# 채널을 다시 결합
lab_clahe = cv2.merge((l_clahe, a, b))

# LAB 색상 공간에서 BGR 색상 공간으로 변환
img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
cv2.imshow('Blended Image (CLAHE)', img_clahe)


# 감마 필터 적용
gamma = 0.8  # gamma = 0.8
inv_gamma = 1.0 / gamma
table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

# Apply gamma correction using the lookup table
gamma_corrected = cv2.LUT(blended_img, table)
cv2.imshow('Blended Image (Gamma)', gamma_corrected)


##############

# RGB 채널별 가우시안 및 라플라시안 피라미드 생성 및 합성 함수
def blend_rgb_pyramids(eye_rgb, hand_rgb, n_levels, x_offset, y_offset):
    # 채널별로 분리
    eye_R, eye_G, eye_B = cv2.split(eye_rgb)
    hand_R, hand_G, hand_B = cv2.split(hand_rgb)

    # 각 채널별로 가우시안 및 라플라시안 피라미드 생성
    GP_eye_R = generate_gaussian_pyramid(eye_R, n_levels)
    GP_eye_G = generate_gaussian_pyramid(eye_G, n_levels)
    GP_eye_B = generate_gaussian_pyramid(eye_B, n_levels)
    GP_hand_R = generate_gaussian_pyramid(hand_R, n_levels)
    GP_hand_G = generate_gaussian_pyramid(hand_G, n_levels)
    GP_hand_B = generate_gaussian_pyramid(hand_B, n_levels)

    LP_eye_R = generate_laplacian_pyramid(GP_eye_R)
    LP_eye_G = generate_laplacian_pyramid(GP_eye_G)
    LP_eye_B = generate_laplacian_pyramid(GP_eye_B)
    LP_hand_R = generate_laplacian_pyramid(GP_hand_R)
    LP_hand_G = generate_laplacian_pyramid(GP_hand_G)
    LP_hand_B = generate_laplacian_pyramid(GP_hand_B)

    # 각 채널별 피라미드 합성
    stitched_R_L = stitch_pyramid_levels(LP_eye_R, LP_hand_R, x_offset, y_offset)
    stitched_G_L = stitch_pyramid_levels(LP_eye_G, LP_hand_G, x_offset, y_offset)
    stitched_B_L = stitch_pyramid_levels(LP_eye_B, LP_hand_B, x_offset, y_offset)
    stitched_R_G = stitch_pyramid_levels(GP_eye_R, GP_hand_R, x_offset, y_offset)
    stitched_G_G = stitch_pyramid_levels(GP_eye_G, GP_hand_G, x_offset, y_offset)
    stitched_B_G = stitch_pyramid_levels(GP_eye_B, GP_hand_B, x_offset, y_offset)

    return stitched_R_L, stitched_G_L, stitched_B_L, stitched_R_G, stitched_G_G, stitched_B_G

# 각 채널별로 가우시안 피라미드의 최소 레벨 이미지와 라플라시안 피라미드를 합성하여 최종 이미지 재구성
def reconstruct_rgb_image(stitched_G_R, stitched_G_G, stitched_G_B, stitched_L_R, stitched_L_G, stitched_L_B):
    final_R = stitched_G_R[-1]
    final_G = stitched_G_G[-1]
    final_B = stitched_G_B[-1]

    for i in range(len(stitched_L_R) - 1, -1, -1):
        final_R = cv2.pyrUp(final_R, dstsize=stitched_L_R[i].shape[1::-1])
        final_R = cv2.add(final_R, stitched_L_R[i])

        final_G = cv2.pyrUp(final_G, dstsize=stitched_L_G[i].shape[1::-1])
        final_G = cv2.add(final_G, stitched_L_G[i])

        final_B = cv2.pyrUp(final_B, dstsize=stitched_L_B[i].shape[1::-1])
        final_B = cv2.add(final_B, stitched_L_B[i])

    final_rgb_image = cv2.merge([final_R, final_G, final_B])
    return final_rgb_image, final_R, final_G, final_B

# RGB 채널별 피라미드 생성 및 합성
stitched_L_R, stitched_L_G, stitched_L_B, stitched_G_R, stitched_G_G, stitched_G_B = blend_rgb_pyramids(eye, hand, n_levels, x_offset, y_offset)

# 최종 이미지 재구성 및 표시
final_rgb_image, final_R, final_G, final_B = reconstruct_rgb_image(stitched_G_R, stitched_G_G, stitched_G_B, stitched_L_R, stitched_L_G, stitched_L_B)
cv2.imshow("RGB Splited Blended Image", final_rgb_image)
cv2.imshow("R Channel Blended Image", final_R)
cv2.imshow("G Channel Blended Image", final_G)
cv2.imshow("B Channel Blended Image", final_B)


##############


# Level에 따른 변화를 한 화면에서 시각화
blend_ing_img = blended_G[-1]
rc = (3,4)
L_MAX = len(blended_L)
plt.figure(8, figsize=(7, 7))
for i in range(L_MAX - 1, -1, -1):
    # pyrup 적용한 gaussian pyramid 이미지와 다음 level의 Laplacian pyramid 이미지를 add
    blend_ing_img = cv2.pyrUp(blend_ing_img, None, blended_G[i].shape[:2][::-1])
    plot_img(rc, 8 + i + 1, blend_ing_img, "Level " + str(i)) # gaussian의 영향 (셋째줄)
    blend_ing_img = cv2.add(blend_ing_img, blended_L[i])
    plot_img(rc, 4 + i + 1, blended_L[i]*7, "Level " + str(i)) # laplacian의 영향 (둘째줄) + 선을 잘 관찰하기 위해 weight 7
    plot_img(rc, i + 1, blend_ing_img, "Level  " + str(i) ) # 합성 결과 (첫줄)


# 라플라시안 피라미드 시각화
visualize_combined_pyramid(7, [img * 6 for img in eye_laplacian_pyramid], 'L')   # 눈 weight 6
visualize_combined_pyramid(6, [img * 6 for img in hand_laplacian_pyramid], 'L')  # 손 weight 6


# 가우시안 피라미드 시각화
visualize_combined_pyramid(5, eye_gaussian_pyramid, 'G')   # 눈
visualize_combined_pyramid(4, hand_gaussian_pyramid, 'G')  # 손


# Combined Pyramid Image 시각화
plt.figure(3, figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.imshow(hand_combine_g)
plt.title("Hand Gaussian Combine Pyramid")
plt.subplot(2, 2, 2)
plt.imshow(hand_combine_l*6) # weight 6
plt.title("Hand Laplacian Combine Pyramid")
plt.subplot(2, 2, 3)
plt.imshow(eye_combine_g)
plt.title("Eye Gaussian Combine Pyramid")
plt.subplot(2, 2, 4)
plt.imshow(eye_combine_l*6) # weight 6
plt.title("Eye Laplacian Combine Pyramid")


# Preprocessed Image 시각화
plt.figure(2, figsize=(7, 7))
plt.subplot(1, 2, 1)
plt.imshow(eye_rgb)
plt.title("Cropped Eye Image")
plt.subplot(1, 2, 2)
plt.imshow(hand_rgb)
plt.title("Resized Hand Image")


# Original Image 시각화
plt.figure(1, figsize=(7, 7))
plt.subplot(1, 2, 1)
plt.imshow(eye_img_rgb)
plt.title("Eye Image")
plt.subplot(1, 2, 2)
plt.imshow(hand_img_rgb)
plt.title("Hand Image")


plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
