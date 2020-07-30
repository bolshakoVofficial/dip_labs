import cv2 as cv
import numpy as np

# ничего страшного тут нет!

cap = cv.VideoCapture('car.mp4')
# parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=5,
                      qualityLevel=0.3,
                      minDistance=4,
                      blockSize=4)
# parameters for LucasKanade optical flow
lk_params = dict(winSize=(5, 5),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

ret, first_frame = cap.read()  # read frame
old_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # corners coords
original_feature_number = len(corners)

fourcc = cv.VideoWriter_fourcc(*'MJPG')  # codec
out = cv.VideoWriter('car_track_regen.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while 1:
    ret, frame = cap.read()  # ret == 1 if next frame exist
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    # nextPts == new coords of corner features
    # status == 1 if feature has been found (tracked)
    nextPts, status, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, corners, None, **lk_params)

    # Select points with status == 1 (i.e. they've been tracked successfully)
    next_feature_points = nextPts[status == 1]
    prev_feature_points = corners[status == 1]

    # тут начинается самое интересное
    '''Как можно заметить на видео без регенерации, не все точки могут преодалеть препятствие 
    (для понимания причин читать про методы отслеживания оптического потока).
    Такие точки упираюстя в полоску и почти не меняют свою координату х. Это можно детектировать.'''

    '''Учитывая, что все характерные точки принадлежат отслеживаемому объекту, они все должны двигаться вместе с ним.
    А это в свою очередь значит, что если двигается хоть одна точка (но лучше опираться на бОльшее количество),
    то должны двигаться и остальные, даже те, что по какой-то причине застряли на своей координате х.
    '''
    total_shift = 0
    regen_flag = False
    wrong_feature_number = 0
    bad_feature_count = 0

    # считаем среднее перемещение для всех точек, даже если какие-то точки застряли
    for i in range(len(next_feature_points)):
        total_shift += next_feature_points[i][0] - corners[i][0][0]
    avg_shift = total_shift / len(next_feature_points)

    # считаем, сколько точек переместились меньше, чем положено (с коэффициентом можно поиграться, debug в помощь)
    for i in range(len(next_feature_points)):
        if next_feature_points[i][0] - corners[i][0][0] < 0.6 * avg_shift:
            bad_feature_count += 1

        '''Дальше узнав, сколько точек двигалось, а сколько нет, можно найти перемещение каждой из точек.
        Помним, что застрявшие точки ПОЧТИ не портят средний показатель перемещения, значит new_avg_shift - это
        перемещение каждой НЕзастрявшей точки. Однако вот это ПОЧТИ выливается в коэффициент 0.8. Играемся...'''
        # сдвиг     = суммарный сдвиг / (кол-во всех точек - нахлебники)
        new_avg_shift = total_shift / (len(next_feature_points) - bad_feature_count)

        if next_feature_points[i][0] - corners[i][0][0] < 0.6 * new_avg_shift:
            next_feature_points[i][0] += new_avg_shift * 0.8

        '''В самом неприятном случае точки могут теряться (перестают отслеживается, status == 0), тогда генерим заново,
        причем столько, сколько потерялось. В моем случае у объекта всегда будут 5 характерных точек (углов), как
        видно в параметрах ShiTomasi corner detection (feature_params)'''
        if len(next_feature_points) < original_feature_number - 1 and not regen_flag:
            feature_params = dict(maxCorners=original_feature_number - len(next_feature_points),
                                  qualityLevel=0.3,
                                  minDistance=5,
                                  blockSize=5)
            new_corners = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            new_points = np.append(next_feature_points, new_corners)
            regen_flag = True

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
    if regen_flag:
        corners = new_points.reshape(-1, 1, 2)

cv.destatusroyAllWindows()
out.release()
cap.release()
