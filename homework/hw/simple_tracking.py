import cv2 as cv

# полезные ссылки по теме:
# https://habr.com/ru/post/244541/
# https://habr.com/ru/post/169055/
# https://habr.com/ru/post/201406/

cap = cv.VideoCapture('car.mp4')
# ShiTomasi corner detection
feature_params = dict(maxCorners=5,
                      qualityLevel=0.3,
                      minDistance=5,
                      blockSize=5)
# LucasKanade optical flow
lk_params = dict(winSize=(5, 5),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

ret, first_frame = cap.read()  # read frame
old_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # corners coords

fourcc = cv.VideoWriter_fourcc(*'MJPG')  # codec
out = cv.VideoWriter('car_track.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while 1:
    ret, frame = cap.read()  # ret == 1 if next frame exist
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    # nextPts == new coords of corner features
    # status == 1 if feature has been found
    nextPts, status, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, corners, None, **lk_params)
    # Select points with status == 1 (i.e. they've been tracked successfully)
    next_feature_points = nextPts[status == 1]
    prev_feature_points = corners[status == 1]

    # draw the tracks
    for i, new in enumerate(prev_feature_points):
        x, y = new.ravel()
        frame = cv.circle(frame, (x, y), 4, (255, 0, 0), -1)
    cv.imshow('frame', frame)
    if ret:
        out.write(frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # update  prev frame and prev points
    old_gray = frame_gray.copy()
    corners = next_feature_points.reshape(-1, 1, 2)

cv.destatusroyAllWindows()
out.release()
cap.release()
