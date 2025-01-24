import cv2
import numpy as np
import matplotlib.pyplot as plt

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

    # 특징점 시각화
    img_keypoints_left = cv2.drawKeypoints(img_left, keypoints_left, None)
    img_keypoints_right = cv2.drawKeypoints(img_right, keypoints_right, None)

    # FLANN 기반 매처
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)  # KNN의 K = 2

    # Lowe's ratio test로 좋은 매치 찾기
    good_correspondences = []
    for m, n in matches:
        if m.distance/n.distance < 0.7:
            good_correspondences.append(m)

    # 매치 시각화
    img_matches = cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, good_correspondences, None)

    # 호모그래피 계산
    src_pts = np.float32([keypoints_left[m.queryIdx].pt for m in good_correspondences]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_right[m.trainIdx].pt for m in good_correspondences]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)  # 호모그래피 계산에는 RANSAC을 사용 
    if reverse: 
        print("Left + Mid Homography : \n" + str(H))
    else: 
        print("SEMI + Right Homography : \n" + str(H))   

    if reverse:                                  # 첫 번째 이미지와 두 번째 이미지 (좌우 반전된 상태)
        height, width = img_left.shape[:2]
        img_result = cv2.warpPerspective(img_right, H, (width + width, height+50)) # 호모그래피를 활용한 변환 적용 + Warp된 이미지가 잘리지 않도록 검은 여백 제공
        temp = cv2.flip(img_result, 1)  
        cv2.imshow("Transformed Image1", temp)
        img_result[0:height, 0:width] = img_left                                      

        img_result = cv2.flip(img_result, 1)     # 좌우 반전 상태에서 변환 적용 후 다시 반전하여 원상태의 이미지로 변환

    else :                                       # 두 번째 이미지와 세 번째 이미지 (그대로 진행)
        height, width = img_left.shape[:2]
        img_result = cv2.warpPerspective(img_right, H, (width + width//2, height+250)) # 호모그래피를 활용한 변환 적용 + Warp된 이미지가 잘리지 않도록 검은 여백 제공
        cv2.imshow("Transformed Image2", img_result)
        img_result[0:height, 0:width] = img_left

    return img_result, img_keypoints_left, img_keypoints_right, img_matches


def visualize_matches(img_left, img_right, flann, keypoints_left, descriptors_left, keypoints_right, descriptors_right, ratio_threshold):
    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)
    good_correspondences = [m for m, n in matches if m.distance/n.distance < ratio_threshold]
    img_matches = cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, good_correspondences, None)
    return img_matches  


def visualize_matches_with_different_parameters(img_left, img_right, trees_values, checks_values, ratio_values):
    sift = cv2.SIFT_create()
    keypoints_left, descriptors_left = sift.detectAndCompute(img_left, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(img_right, None)

    # Trees에 따른 시각화 (인덱스 구축에 사용, tree가 클수록 정확도 Up)
    plt.figure(figsize=(19,10))
    for i, trees in enumerate(trees_values, 1):
        flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": trees}, {"checks": 50})
        img_matches = visualize_matches(img_left, img_right, flann, keypoints_left, descriptors_left, keypoints_right, descriptors_right, 0.7)
        plt.subplot(1, len(trees_values), i)
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f"Trees: {trees}")
        plt.axis('off')

    # Checks에 따른 시각화 (인덱스 트리에서 얼마나 많은 경로를 확인할 것인가?, 클수록 정확도 Up)
    plt.figure(figsize=(19,10))
    for i, checks in enumerate(checks_values, 1):
        flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": checks})
        img_matches = visualize_matches(img_left, img_right, flann, keypoints_left, descriptors_left, keypoints_right, descriptors_right, 0.7)
        plt.subplot(1, len(checks_values), i)
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f"Checks: {checks}")
        plt.axis('off')

    # Ratio에 따른 시각화
    plt.figure(figsize=(19,10))
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    for i, ratio in enumerate(ratio_values, 1):
        img_matches = visualize_matches(img_left, img_right, flann, keypoints_left, descriptors_left, keypoints_right, descriptors_right, ratio)
        plt.subplot(1, len(ratio_values), i)
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f"Ratio: {ratio}")
        plt.axis('off')

# 이미지 로드
img1 = load_resize_image("2019204014_left.jpg")  # 첫 번째 이미지
img2 = load_resize_image("2019204014_mid.jpg")   # 두 번째 이미지
img3 = load_resize_image("2019204014_right.jpg")  # 세 번째 이미지


# 이미지 Stitching
result1, kpts1_left, kpts1_right, matches1 = make_panorama(img2, img1, True)
result2, kpts2_left, kpts2_right, matches2 = make_panorama(img2, img3, False)
result, kpts_left, kpts_right, matches = make_panorama(result1, img3, False)


# 파라미터 설정
trees_values = [2, 4, 6, 8, 10]
checks_values = [20, 40, 60, 80, 100]
ratio_values = [0.2, 0.4, 0.6, 0.8, 1.0]

# 시각화 실행
visualize_matches_with_different_parameters(img1, img2, trees_values, checks_values, ratio_values)


# 결과와 특징점, 매치 시각화 표시 (첫 번째 이미지와 두 번째 이미지)
plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(kpts1_left, cv2.COLOR_BGR2RGB))
plt.title("Keypoints Left")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(kpts1_right, cv2.COLOR_BGR2RGB))
plt.title("Keypoints Mid")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(matches1, cv2.COLOR_BGR2RGB))
plt.title("Matches Left & Mid")
plt.axis('off')


# 결과와 특징점, 매치 시각화 표시 (두 번째 이미지와 세 번째 이미지)
plt.figure(figsize=(15, 7))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(kpts2_left, cv2.COLOR_BGR2RGB))
plt.title("Keypoints Mid")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(kpts2_right, cv2.COLOR_BGR2RGB))
plt.title("Keypoints Right")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(matches2, cv2.COLOR_BGR2RGB))
plt.title("Matches Mid & Right")
plt.axis('off')


# 결과와 특징점, 매치 시각화 표시 (두 번째 이미지와 세 번째 이미지)
plt.figure(figsize=(15, 7))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(kpts_left, cv2.COLOR_BGR2RGB))
plt.title("Keypoints Left+Mid")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(kpts_right, cv2.COLOR_BGR2RGB))
plt.title("Keypoints Right")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(matches, cv2.COLOR_BGR2RGB))
plt.title("Matches Full")
plt.axis('off')

plt.show()
