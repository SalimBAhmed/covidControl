# imports
import cv2
import numpy as np
import time
import argparse
import os

import torch
import torch.backends.cudnn as cudnn

# own modules
import utills, plot
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, non_max_suppression, set_logging
from utils.torch_utils import select_device

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] # x center
    y[:, 1] = x[:, 1]  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
    
            
def get_mouse_points(event, x, y, flags, param):
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)            
        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        #print("Point detected")
        #print(mouse_pts)


def calculate_social_distancing(opt,img_size = 640):
    sources = opt.source
    webcam = sources.isnumeric() or sources.endswith('.txt') or sources.lower().startswith(('rtsp://','rtmp://','http://','https://'))
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
    if(sources == "0"):
        sources = 0
    #print("video path",source)
    elif sources.endswith(".txt"):
        f = open(sources, "r")
        sources = f.readlines()
        f.close()
        
    if not isinstance(sources,list):
        sources = [sources]
    
    model_path = opt.model
    
    if model_path[len(model_path) - 1] != '/':
        model_path = model_path + '/'
    
    output_vid = opt.project
    if output_vid[len(output_vid) - 1] != '/':
        output_vid = output_vid + '/'
        
    out_path = ""
    if output_vid[0] == '/':
        out_path = '/'
    dir_list = output_vid.split("/")
    for folder in dir_list :
        
        out_path += folder.strip() + '/'  
        if not(os.path.exists(out_path)):
            os.mkdir(out_path)
    
    # load Yolov3 weights

    weightsPath = model_path + "yolov3.weights"
    configPath = model_path + "yolov3.cfg"
    
    # load Yolov5 weights
    weights = opt.model + "best.pt"
        
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    #Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
        
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


   

    
    global image
    for source in sources:
        if source != 0:
            output_dir = output_vid + source.split("/")[-1].split(".")[0] + "/"
            source = source.strip()
        else :
            output_dir = output_vid + "capture/"
            
        if not(os.path.exists(output_dir)):
            os.mkdir(output_dir)
            os.mkdir(output_dir+"bird_eye_view/")
        
        count = 0
        points = []

        vs = cv2.VideoCapture(source)
        height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(vs.get(cv2.CAP_PROP_FPS))
	
        
        # Set scale for birds eye view
        # Bird's eye view will only show ROI
        scale_w, scale_h = utills.get_scale(width, height)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        bird_movie = cv2.VideoWriter(output_dir + "bird_eye_view/bird_eye_view.mp4", fourcc, fps, (int(width * scale_w), int(height * scale_h)))
        output_movie = cv2.VideoWriter(output_dir + "distancing.mp4", fourcc, fps, (width, height))    
        
        
        while True:
            (grabbed, frame) = vs.read()
            withMask, noMask = 0, 0

            if not grabbed:
                print('here')
                break
                
            (H, W) = frame.shape[:2]    
            # first frame will be used to draw ROI and horizontal and vertical 180 cm distance(unit length in both directions)
            if count == 0:
                while True:
                    #image = cv2.resize(frame, (600, 400))
                    image = frame
                    cv2.imshow("CovidControl", image)
                    cv2.waitKey(1)
                    if len(mouse_pts) == 8:
                        cv2.destroyWindow("CovidControl")
                        break

                points = mouse_pts
        
            # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are 
            # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view. 
            # This bird eye view then has the property property that points are distributed uniformally horizontally and 
            # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are 
            # equally distributed, which was not case for normal view.
            src = np.float32(np.array(points[:4]))
            dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
            prespective_transform = cv2.getPerspectiveTransform(src, dst)

            # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
            pts = np.float32(np.array([points[4:7]]))
            warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

            # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
            # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
            # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
            # which we can use to calculate distance between two humans in transformed view or bird eye view
            distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
            distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
            pnts = np.array(points[:4], np.int32)
            cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)
    
              ####################################################################################
            # YOLO v3
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln1)
            end = time.time()
            boxes = []
            confidences = []
            classIDs = []   

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    # detecting humans in frame
                    if classID == 0:

                        if confidence > confid:

                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)
                
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
            font = cv2.FONT_HERSHEY_PLAIN
            boxes1 = []
            for i in range(len(boxes)):
                if i in idxs:
                    boxes1.append(boxes[i])
                    x,y,w,h = boxes[i]    

            if len(boxes1) == 0:
                count = count + 1
                continue
        
            # Here we will be using bottom center point of bounding box for all boxes and will transform all those
            # bottom center points to bird eye view
            person_points = utills.get_transformed_points(boxes1, prespective_transform)
    
            # Here we will calculate distance between transformed points(humans)
            distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, distance_w, distance_h)
            risk_count = utills.get_count(distances_mat)
        
            frame1 = np.copy(frame)
    
            
    
            #mask detection
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                
            img, ratio, otherThing = letterbox(frame)

            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)
            #img = cv2.resize(frame,(imgsz,imgsz))
            img = torch.from_numpy(img).to(device)
    
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,  max_det=opt.max_det)
    
             # Process detections
            for i, det in enumerate(pred):  # for each prediction in the frame
                if len(det):
                    #det = det.cpu().detach().numpy()
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                    xyxy,conf,state = det[:,:4],det[:,4].tolist(),det[:,5].tolist()
                    xywh = xyxy2xywh(xyxy).tolist() #boxes from xyxy to xywh
                    for ind in range(len(det)):
                        x,y,w,h = int(xywh[ind][0]),int(xywh[ind][1]),int(xywh[ind][2]),int(xywh[ind][3])
                        if(state[ind] == 0):
                            if opt.labels:
                                cv2.putText(frame1,"Mask",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                            cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),opt.line_thickness)
                            withMask += 1
                        else:
                            if opt.labels:
                                cv2.putText(frame1,"No mask",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                            cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),opt.line_thickness)
                            noMask += 1
                        if opt.conf:
                                cv2.putText(frame1,conf,(x,h),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
	
	
            # Draw bird eye view and frame with bouding boxes around humans according to risk factor    
            bird_image = plot.bird_eye_view(frame, distances_mat, person_points, scale_w, scale_h, risk_count)
            frame1 = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count,withMask,noMask)
            
            
            # Show/write image and videos
            if count != 0:
                

                cv2.imshow('Bird Eye View', bird_image)
                cv2.imshow('Covid Controle', frame1)


                if not opt.nosave:
                    bird_movie.write(bird_image)
                    output_movie.write(frame1)
                    
                    cv2.imwrite(output_dir+"frame%d.jpg" % count, frame1)
                    cv2.imwrite(output_dir+"bird_eye_view/b_frame%d.jpg" % count, bird_image)
        
            count += 1
            cv2.waitKey(1)

        vs.release()
        cv2.destroyAllWindows() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', type=str, default='models/', help='model folders path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--project', default='output/', help='save results to project/name')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))
    

    # set mouse callback
    cv2.namedWindow("CovidControl")
    mouse_pts = []
    confid = opt.conf_thres
    thresh = 0.5
    cv2.setMouseCallback("CovidControl", get_mouse_points) # CallBack to get_mouse_points
    np.random.seed(42)
      
    calculate_social_distancing(opt)
