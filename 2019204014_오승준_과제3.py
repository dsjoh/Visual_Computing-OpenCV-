import cv2
import numpy as np

def load_resize_image(path):  # 이미지를 불러오되, 이미지 사이즈가 너무 커서 resize를 진행
    img = cv2.imread(path)
    h, w, c = img.shape
    img = cv2.resize(img, (w // 6, h // 6)) # 1/6 Scale
    return img

def make_panorama(img_left, img_right, reverse=False):  # 파노라마 이미지를 만드는 함수
    if reverse:                                         # 첫번째 이미지와 두번째 이미지의 Stitching 과정은 이미지를 좌우로 Flip하여 적용
        img_left = cv2.flip(img_left, 1)                # 첫 번째 이미지 좌우 반전
        img_right = cv2.flip(img_right, 1)              # 두 번째 이미지 좌우 반전

    # 이미지를 그레이스케일로 변환
    img_gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # SIFT로 특징점 추출
    sift = cv2.SIFT_create()
    keypoints_left, descriptors_left = sift.detectAndCompute(img_gray_left, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(img_gray_right, None)

    # FLANN 기반 매처
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)  # KNN의 K = 2

    # Lowe's ratio test로 좋은 매치 찾기
    good_correspondences = []
    for m, n in matches:
        if m.distance/n.distance < 0.7:
            good_correspondences.append(m)

    # 호모그래피 계산
    src_pts = np.float32([keypoints_left[m.queryIdx].pt for m in good_correspondences]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_right[m.trainIdx].pt for m in good_correspondences]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)  # 호모그래피 계산에는 RANSAC을 사용 


    if reverse:
        height, width = img_left.shape[:2]      # 첫 번째 이미지와 두 번째 이미지 (좌우 반전된 상태)
        img_result = cv2.warpPerspective(img_right, H, (width + width, height+50))  # 호모그래피를 활용한 변환 적용 + Warp된 이미지가 잘리지 않도록 검은 여백 제공

        # 원본 이미지와 겹치는 영역 마스크 생성
        mask = cv2.warpPerspective(np.ones_like(img_right[:, :, 0], dtype=np.uint8), H, (width + width, height+50))
        overlap_mask = mask[0:height, 0:width] & np.ones_like(img_left[:, :, 0], dtype=np.uint8)
        overlap_mask = cv2.flip(overlap_mask, 1)
        cv2.imshow("left_mid_Overlap Mask", overlap_mask * 255)  # 이진 마스크 시각화

        img_result[0:height, 0:width] = img_left
        img_result = cv2.flip(img_result, 1)  # 좌우 반전 상태에서 변환 적용 후 다시 반전하여 원상태의 이미지로 변환

    else:                                     # 두 번째 이미지와 세 번째 이미지 (그대로 진행)
        height, width = img_left.shape[:2]
        img_result = cv2.warpPerspective(img_right, H, (width + width//2, height+200))

        # 원본 이미지와 겹치는 영역 마스크 생성
        mask = cv2.warpPerspective(np.ones_like(img_right[:, :, 0], dtype=np.uint8), H, (width + width//2, height+200))  # 호모그래피를 활용한 변환 적용 + Warp된 이미지가 잘리지 않도록 검은 여백 제공
        overlap_mask = mask[0:height, 0:width] & np.ones_like(img_left[:, :, 0], dtype=np.uint8)
        cv2.imshow("mid_right_Overlap Mask", overlap_mask * 255)  # 이진 마스크 시각화

        img_result[0:height, 0:width] = img_left   

    return img_result


# 이미지 로드
img1 = load_resize_image("2019204014_left.jpg")  # 첫 번째 이미지
img2 = load_resize_image("2019204014_mid.jpg")   # 두 번째 이미지
img3 = load_resize_image("2019204014_right.jpg")  # 세 번째 이미지

# 이미지 Stitcing
result1 = make_panorama(img2, img1, True)
result2 = make_panorama(img2, img3,  False)
result = make_panorama(result1, img3, False)


# 해상도 문제로 인한 resize 및 ROI
h, w, c = result.shape
result = cv2.resize(result, (int(w * 3 / 4), int(h * 3 / 4)))
x=210; y=0; w=1550; h=365           # roi 좌표
Roi_result = result[y:y+h, x:x+w]  # roi 지정 


# 결과 표시
cv2.imshow("Left & Mid", result1)
cv2.imshow("Mid & Right", result2)
cv2.imshow("Panoramoa Image(Unprocessed)", result)
cv2.imshow("Panoramoa Image(2019204014)", Roi_result)

############## 추가 시각화 ##############

# 이미지 로드
img1 = load_resize_image("2019204014_left.jpg")  # 첫 번째 이미지
img2 = load_resize_image("2019204014_mid.jpg")   # 두 번째 이미지
img3 = load_resize_image("2019204014_right.jpg")  # 세 번째 이미지


# 가운데 이미지의 높이, 너비 및 채널 가져오기
height, width, channels = img2.shape

# 검은색 여백의 크기 설정
black_space = 300

# 검은색 여백을 추가하기 위해 새로운 이미지 크기 생성
new_height = height + black_space
new_image = np.zeros((new_height, width, channels), dtype=np.uint8)

# 원본 이미지를 새 이미지의 아래쪽에 복사 (여백은 상단에 위치)
new_image[black_space:height + black_space, 0:width] = img2
img2 = new_image


# 이미지 Stitcing
result1 = make_panorama(img2, img1, True)
result2 = make_panorama(img2, img3,  False)
result = make_panorama(result1, img3, False)


# 결과 표시
cv2.imshow("Panoramoa Image(full)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

