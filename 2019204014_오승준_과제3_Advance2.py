import cv2
import numpy as np

def load_resize_image(path):  # 이미지를 불러오되, 이미지 사이즈가 너무 커서 resize를 진행
    img = cv2.imread(path)
    h, w, c = img.shape
    img = cv2.resize(img, (w // 6, h // 6)) # 1/6 Scale
    return img

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
        if reverse:                                  
            height, width = img_left.shape[:2]
            img_result = cv2.warpPerspective(img_right, H, (width + width, height+50)) # 호모그래피를 활용한 변환 적용 + Warp된 이미지가 잘리지 않도록 검은 여백 제공
            
            # 포아송 블렌딩을 위한 마스크 생성
            mask = np.zeros((height+50, width + width), dtype=np.uint8)
            mask[0:height, 0:width] = 255  # 왼쪽 이미지 부분을 덮는 마스크 생성

            # 포아송 블렌딩 적용 (SeamlessClone)
            center = (width // 2, height // 2)
            img_result = cv2.seamlessClone(img_left, img_result, mask, center, cv2.NORMAL_CLONE)

            img_result = cv2.flip(img_result, 1)     

        else :                                       
            height, width = img_left.shape[:2]
            img_result = cv2.warpPerspective(img_right, H, (width + width, height+200)) # 호모그래피를 활용한 변환 적용 + Warp된 이미지가 잘리지 않도록 검은 여백 제공

            # 포아송 블렌딩을 위한 마스크 생성
            mask = np.zeros((height+200, width + width//2), dtype=np.uint8)
            mask[0:height, 0:width] = 255  # 왼쪽 이미지 부분을 덮는 마스크 생성

            # 포아송 블렌딩 적용 (SeamlessClone)
            center = (width // 2, height // 2)
            img_result = cv2.seamlessClone(img_left, img_result, mask, center, cv2.NORMAL_CLONE)

        return img_result
    

    else :         # Advanced1 적용 후에 대한 코드 
        if reverse:                                  
            height, width = img_left.shape[:2]
            img_result = cv2.warpPerspective(img_right, H, (width + int(width*4/2), height+50)) # 호모그래피를 활용한 변환 적용 + Warp된 이미지가 잘리지 않도록 검은 여백 제공
            
            # 포아송 블렌딩을 위한 마스크 생성
            mask = np.zeros((height+50, width + width), dtype=np.uint8)
            mask[0:height, 0:width] = 255  # 왼쪽 이미지 부분을 덮는 마스크 생성

            # 포아송 블렌딩 적용 (SeamlessClone)
            center = (width // 2, height // 2)
            img_result = cv2.seamlessClone(img_left, img_result, mask, center, cv2.NORMAL_CLONE)

            img_result = cv2.flip(img_result, 1)     

        else :                                       
            height, width = img_left.shape[:2]
            img_result = cv2.warpPerspective(img_right, H, (width + int(width*1/2), height+200)) # 호모그래피를 활용한 변환 적용 + Warp된 이미지가 잘리지 않도록 검은 여백 제공

            # 포아송 블렌딩을 위한 마스크 생성
            mask = np.zeros((height+200, width + width//2), dtype=np.uint8)
            mask[0:height, 0:width] = 255  # 왼쪽 이미지 부분을 덮는 마스크 생성

            # 포아송 블렌딩 적용 (SeamlessClone)
            center = (width // 2, height // 2)
            img_result = cv2.seamlessClone(img_left, img_result, mask, center, cv2.NORMAL_CLONE)

        return img_result
    

############## Advanced 1 적용 전에 대한 Blending ################


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


# Adaptive Histogram Equalization (CLAHE) 적용
# YCrCb 컬러 스페이스로 변환
Roi_ycrcb = cv2.cvtColor(Roi_result, cv2.COLOR_BGR2YCrCb)

# CLAHE 적용 (밝기에 대한 Y 채널에만 적용)
clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(2, 2))
Roi_ycrcb[:, :, 0] = clahe.apply(Roi_ycrcb[:, :, 0])

# 다시 BGR 컬러 스페이스로 변환
Roi_clahe = cv2.cvtColor(Roi_ycrcb, cv2.COLOR_YCrCb2BGR)

# 결과 표시
cv2.imshow("Left & Mid", result1)
cv2.imshow("Mid & Right", result2)
cv2.imshow("Advanced2 Panoramoa Image(Unprocessed)", result)
cv2.imshow("Advanced2 AHE Panoramoa Image(Unprocessed)", Roi_clahe)
cv2.imshow("Advanced2 Panoramoa ROI Image(Unprocessed)", Roi_result)


############## Advanced 1 적용 후에 대한 Blending ################


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


# Adaptive Histogram Equalization (CLAHE) 적용
# YCrCb 컬러 스페이스로 변환
Roi_ycrcb = cv2.cvtColor(Roi_result, cv2.COLOR_BGR2YCrCb)

# CLAHE 적용 (밝기에 대한 Y 채널에만 적용)
clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(2, 2))
Roi_ycrcb[:, :, 0] = clahe.apply(Roi_ycrcb[:, :, 0])

# 다시 BGR 컬러 스페이스로 변환
Roi_clahe = cv2.cvtColor(Roi_ycrcb, cv2.COLOR_YCrCb2BGR)

# 결과 표시
cv2.imshow("Left & Mid", result1)
cv2.imshow("Mid & Right", result2)
cv2.imshow("Advanced2 Panoramoa Image(processed)", result)
cv2.imshow("Advanced2 AHE Panoramoa Image(processed)", Roi_clahe)
cv2.imshow("Advanced2 Panoramoa ROI Image(processed)", Roi_result)
cv2.waitKey(0)
cv2.destroyAllWindows()