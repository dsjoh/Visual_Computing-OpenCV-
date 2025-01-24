import cv2
import numpy as np

def load_resize_image(path):  # 이미지를 불러오되, 이미지 사이즈가 너무 커서 resize를 진행
    img = cv2.imread(path)
    h, w, c = img.shape
    img = cv2.resize(img, (w // 6, h // 6)) # 1/6 Scale
    return img

def preproceesed_image(img, start_x=0, start_y=0, minus_h=0, minus_w=0 ,angle=0):  
    # 원본 이미지의 정보
    h, w, c = img.shape
    h = h-minus_h
    w = w-minus_w

    # ROI를 기반으로 이미지 Crop
    cropped_img = img[start_y:start_y+h, start_x:start_x+w]

    # Crop된 이미지의 정보
    h, w, c = cropped_img .shape

    # 이미지 중심을 계산
    center = (w // 2, h // 2)

    # 회전을 위한 변환 행렬 계산
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 회전 적용
    rotated_img = cv2.warpAffine(cropped_img, rotation_matrix, (w, h))

    return rotated_img

def make_panorama(img_left, img_right, reverse=False, before=True):  # 파노라마 이미지를 만드는 함수
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

    if before :    # Advanced1 적용 전에 대한 코드 
        if reverse:                                  # 첫 번째 이미지와 두 번째 이미지 (좌우 반전된 상태)
            height, width = img_left.shape[:2]
            img_result = cv2.warpPerspective(img_right, H, (width + width, height+50)) # 호모그래피를 활용한 변환 적용 + Warp된 이미지가 잘리지 않도록 검은 여백 제공
            img_result[0:height, 0:width] = img_left                                      

            img_result = cv2.flip(img_result, 1)     # 좌우 반전 상태에서 변환 적용 후 다시 반전하여 원상태의 이미지로 변환

        else :                                       # 두 번째 이미지와 세 번째 이미지 (그대로 진행)
            height, width = img_left.shape[:2]
            img_result = cv2.warpPerspective(img_right, H, (width + width, height+200)) # 호모그래피를 활용한 변환 적용 + Warp된 이미지가 잘리지 않도록 검은 여백 제공
            img_result[0:height, 0:width] = img_left

        return img_result        


    else :         # Advanced1 적용 후에 대한 코드 
        if reverse:                                  # 첫 번째 이미지와 두 번째 이미지 (좌우 반전된 상태)
            height, width = img_left.shape[:2]
            img_result = cv2.warpPerspective(img_right, H, (width + int(width*4/2), height+50)) # 호모그래피를 활용한 변환 적용 + Warp된 이미지가 잘리지 않도록 검은 여백 제공
            img_result[0:height, 0:width] = img_left                                      

            img_result = cv2.flip(img_result, 1)     # 좌우 반전 상태에서 변환 적용 후 다시 반전하여 원상태의 이미지로 변환

        else :                                       # 두 번째 이미지와 세 번째 이미지 (그대로 진행)
            height, width = img_left.shape[:2]
            img_result = cv2.warpPerspective(img_right, H, (width + int(width*1/2), height+200)) # 호모그래피를 활용한 변환 적용 + Warp된 이미지가 잘리지 않도록 검은 여백 제공
            img_result[0:height, 0:width] = img_left

        return img_result


############### Advanced1 추가 전 ##################


# 이미지 로드
img1 = load_resize_image("2019204014_left.jpg")  # 첫 번째 이미지
img2 = load_resize_image("2019204014_mid.jpg")   # 두 번째 이미지
img3 = load_resize_image("2019204014_right.jpg")  # 세 번째 이미지

# 가운데 이미지의 높이, 너비 및 채널 가져오기
height, width, channels = img2.shape

# 검은색 여백의 크기 설정
black_space = 500

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

# 해상도 문제로 인한 resize 및 ROI
h, w, c = result.shape
result = cv2.resize(result, (int(w * 3 / 4), int(h * 3 / 4)))
x=220; y=int(500*3/4); w=1550; h=365           # roi 좌표
Roi_result = result[y:y+h, x:x+w]  # roi 지정 

# 결과 표시
cv2.imshow("Panoramoa Image(Before Unprocessed Full Image)", result)
cv2.imshow("Panoramoa Image(Before Unprocessed ROI Image)", Roi_result)


############### Advanced1 추가 후 ##################


# 이미지 로드
img1 = load_resize_image("2019204014_left.jpg")  # 첫 번째 이미지
img2 = load_resize_image("2019204014_mid.jpg")   # 두 번째 이미지
img3 = load_resize_image("2019204014_right.jpg")  # 세 번째 이미지


# 이미지에서 손실되는 정보를 최소화하기 위해 겹쳐지는 영역들을 Crop할 수 있는 적절한 ROI를 지정한다. 
# (left는 이미 적절히 변환된 상태라 전처리의 효과가 상대적으로 미미했기에 mid와 right에 대해서 수행한다.)

# mid
h, w, c = img2.shape
x=int(w*(16/36)); y=0; w=int(w*(6/7)); h=int(h*1)  # roi 좌표
img2 = img2[y:y+h, x:x+w]  # roi 지정 

# right
h, w, c = img3.shape
x=int(w*(17/80)); y=0; w=int(w*(7/7)); h=int(h*1)  # roi 좌표
img3 = img3[y:y+h, x:x+w]  # roi 지정 


# 가운데 이미지의 높이, 너비 및 채널 가져오기
height, width, channels = img2.shape

# 검은색 여백의 크기 설정
black_space = 400

# 검은색 여백을 추가하기 위해 새로운 이미지 크기 생성
new_height = height + black_space
new_image = np.zeros((new_height, width, channels), dtype=np.uint8)

# 원본 이미지를 새 이미지의 아래쪽에 복사 (여백은 상단에 위치)
new_image[black_space:height + black_space, 0:width] = img2
img2 = new_image


# 이미지 Stitcing
result1 = make_panorama(img2, img1, True, False)
result2 = make_panorama(img2, img3,  False, False)
result = make_panorama(result1, img3, False, False)


# 해상도 문제로 인한 resize 및 ROI
h, w, c = result.shape
result = cv2.resize(result, (int(w * 3 / 4), int(h * 3 / 4)))
x=140; y=int(400*3/4); w=1100; h=380           # roi 좌표
Roi_result = result[y:y+h, x:x+w]  # roi 지정 

# 결과 표시
cv2.imshow("Panoramoa Image(After processed Full Image)", result)
cv2.imshow("Panoramoa Image(After processed ROI Image)", Roi_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
