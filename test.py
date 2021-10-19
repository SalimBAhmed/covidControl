import cv2
source = "videos/20210603_101941.mp4"

vs = cv2.VideoCapture(source)
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(vs.get(cv2.CAP_PROP_FPS))

print(height,width,fps)

while True:
    (grabbed, frame) = vs.read()
    cv2.imshow("image",frame)
    cv2.waitKey(1)
vs.release()
cv2.destroyAllWindows()
