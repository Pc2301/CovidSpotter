#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import glob
import warnings
import cv2
import imutils.video
import numpy as np
import argparse
import os
# import flirimageextractor
# import sys

from PIL                    import Image
from keras.models           import load_model
from models_used            import YOLO, MobileNet, YOLO_MASK, RETINANET
from timeit                 import time
from deep_sort              import preprocessing
from deep_sort.detection    import Detection
from mtcnn.mtcnn            import MTCNN
# from deep_sort              import nn_matching
# from deep_sort.tracker      import Tracker
# from tools                  import generate_detections as gdet
from scipy.spatial          import distance as dist
warnings.filterwarnings('ignore')

def get_facial_embeddings(model, face_pixels):
    ## Function to obtain the facial embeddings from the face-recognition model.
    face_pixels = face_pixels.astype('float32')
    mean, std   = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples     = np.expand_dims(face_pixels, axis=0)
    yhat        = model.predict(samples)
    return yhat[0]

def perform_defaulter_face_recognition(dangerous_facial_bounding_boxes, frame, face_detector, required_size_faces, face_net,
                                       stored_facial_embeddings, meta_data):
    defaulters_embeddings = list()
    for facial_bounding_boxes in dangerous_facial_bounding_boxes:
        sx, sy, ex, ey = facial_bounding_boxes
        facial_image   = frame[int(sy):int(ey), int(sx):int(ex)]
        results        = face_detector.detect_faces(np.asarray(facial_image))
        if len(results) != 0:
            x1, y1, width, height = results[0]['box']
            x1, y1    = abs(x1), abs(y1)
            x2, y2    = x1 + width, y1 + height
            face      = facial_image[y1:y2, x1:x2]
            image     = Image.fromarray(face)
        else:
            image = Image.fromarray(facial_image)
        image = image.resize(required_size_faces)
        face  = np.asarray(image)
        embedding = get_facial_embeddings(face_net,face)
        defaulters_embeddings.append(embedding)

    defaulters_embeddings       = np.asarray(defaulters_embeddings)
    similarity                  = np.matmul(defaulters_embeddings, stored_facial_embeddings.transpose())
    defaulters_labels_predicted = np.argmax(similarity, axis=1)
    defaulters_names_predicted  = meta_data[defaulters_labels_predicted]
    return defaulters_names_predicted

def initial_requirements_width(frame):
    ## This is a function where the user will select two pixel locations in the frame that are 1 FEET away in the real world image.
    print("CLICK and SELECT TWO POINTS WHICH ARE 1 FEET APART in the FRAME")
    print('PRESS "ENTER" TO EXIT After clicking')
    refPt = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.append([x, y])
            cv2.circle(clone, (x, y), 5,  (255, 255, 255), -1)
            cv2.imshow("clone", clone)
        if event == cv2.EVENT_RBUTTONDOWN:
            blue    = clone[y, x, 0]
            green   = clone[y, x, 1]
            red     = clone[y, x, 2]
            font    = cv2.FONT_HERSHEY_SIMPLEX
            strBGR  = str(blue) + ", " + str(green) + "," + str(red)
            cv2.putText(clone, strBGR, (x, y), font, 0.1, (255, 255, 255), 2)
            cv2.imshow("clone", clone)
    clone = frame.copy()
    cv2.imshow("clone", clone)
    cv2.setMouseCallback("clone", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    refPt = np.asarray(refPt)
    if refPt.shape[0] < 2:
        raise ValueError("SELECT AT-LEAST TWO POINTS !!!!")
    xA = refPt[-1, 0]; yA = refPt[-1, 1]
    xB = refPt[-2, 0]; yB = refPt[-2, 1]
    reference_width = dist.euclidean((xA, yA), (xB, yB))
    time.sleep(2)
    return  reference_width

def initial_requirements_temperature(frame, selected_reference_temperature):
    ## This is a function where the user will select a pixel location in the frame which the desired reference temperature i.e "selected_reference_temperature".
    ## Needed to calculate the Facial Temperature of the detected person bounding boxes.
    print("CLICK and SELECT THE POINT FOR THE REFERENCE TEMPERATURE=%s C ----- "
          "IF YOU CLICK on MULTIPLE POINTS, the LAST POINT WILL BE SELECT" % selected_reference_temperature)
    print('PRESS "ENTER" TO EXIT After clicking')
    refPt = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.append([x, y])
            font  = cv2.FONT_HERSHEY_SIMPLEX
            strXY = str(x) + ", " + str(y)
            cv2.putText(clone, strXY, (x, y), font, 0.5, (255, 255, 0), 2)
            cv2.imshow("clone", clone)
        if event == cv2.EVENT_RBUTTONDOWN:
            blue   = clone[y, x, 0]
            green  = clone[y, x, 1]
            red    = clone[y, x, 2]
            font   = cv2.FONT_HERSHEY_SIMPLEX
            strBGR = str(blue) + ", " + str(green) + "," + str(red)
            cv2.putText(clone, strBGR, (x, y), font, 0.5, (0, 255, 255), 2)
            cv2.imshow("clone", clone)

    clone = frame.copy()
    cv2.imshow("clone", clone)
    cv2.setMouseCallback("clone", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    refPt        = np.asarray(refPt)
    reference_px = int(refPt[-1, 0])
    reference_py = int(refPt[-1, 1])
    print("REFERENCE POINT HAS X=%d and Y=%d" % (reference_px, reference_py))
    time.sleep(2)
    return reference_px, reference_py

def find_defaulters(detections, num_detected_persons, reference_width, crowd_max, social_distance_in_feet):
    defaulters_found_flag        = False
    defaulters_crowd             = [] # ids for persons who are gathering in a crowd.
    defaulters_social_distancing = [] # ids for persons who are not-following social-distancing.
    DETECTIONS                   = {'Ref-' + det.id : det for det in detections}
    ''' CHECK for DEFAULTERS. '''
    if num_detected_persons > crowd_max:  ## CROWD COUNTING.
        ''' ALL ARE THE DEFAULTERS. '''
        defaulters_crowd      =  [det.id for det in detections]
        defaulters_found_flag = True

    elif num_detected_persons > 0:  # Create the reference objects: Which will be all the bounding boxes.
        ## Calculting Social Distancing
        references    = {'Ref-' + det.id : det.to_four_corners() for det in detections}
        defaulter_ids = [[], []]
        ids           = [det.id for det in detections]
        for i in ids: #https://www.pyimagesearch.com/2016/04/04/measuring-distance-between-objects-in-an-image-with-opencv/
            ref_box   = references['Ref-' + str(i)]
            refCoords = np.vstack([ref_box.box, ref_box.center])
            J         = ids.copy()
            J.remove(i)
            for j in J:
                other_box = references['Ref-' + str(j)]
                objCoords = np.vstack([other_box.box, other_box.center])
                D = [] # pairwise Distance between the edges of the bounding boxes between 'i' and 'j' bounding boxes.
                for ((xA, yA), (xB, yB)) in zip(refCoords, objCoords):
                    D.append(dist.euclidean((xA, yA), (xB, yB)) / reference_width)

                if min(D) < social_distance_in_feet: # Minimum distance between 'i' and 'j' ids < social_distance_in_feet
                    defaulters_found_flag = True
                    defaulter_ids[0].append(i) # Add i to the defaulters.
                    defaulter_ids[1].append(j) # Add j to the defaulters.

        if len(defaulter_ids[0]) > 0:
            defaulters_social_distancing = defaulter_ids[0] + defaulter_ids[1]
            defaulters_social_distancing = list(set(defaulters_social_distancing)) # Find the unique defaulter ids from the i's and j's
    if defaulters_found_flag:
        defaulter_ids = defaulters_crowd + defaulters_social_distancing
        defaulter_ids = list(set(defaulter_ids)) # Find the unique ids from the defaulters of social-distancing and crowd-gathering.
        for id in defaulter_ids:
            DETECTIONS['Ref-'+id].danger_social_distance = True # Set the danger_social_distance to True.
    return list(DETECTIONS.values())

def find_defaulters_temperature(detections, frame, frame_path, reference_px, reference_py, img_width, img_height, config):
    radiometric    = False
    imageToProcess = frame
    thermal        = None
    '''
    radiometric = True if config.radiometric == "True" else False
    ## Perform the Radiometric Transformations according to IIT-PAVIS
    if radiometric:  # Get radiometric matrix
        flir = flirimageextractor.FlirImageExtractor()# If the input image is from FLIR camera.
        flir.process_image(frame_path)
        thermal = flir.get_thermal_np()
        gray           = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Invert levels
        gray_inverted  = cv2.bitwise_not(gray)  # Convert inverted grayscale to Color RGB format
        imageToProcess = cv2.cvtColor(gray_inverted, cv2.COLOR_GRAY2BGR)
    else:
        imageToProcess = frame
    '''

    count = 0
    for i in range(len(detections)):
        count += 1
        if  detections[i].face_bbox is None:
            continue
        width       = int(detections[i].face_bbox[2])
        height      = int(detections[i].face_bbox[3])
        reference_x = int(detections[i].face_bbox[0] + width/2)
        reference_y = int(detections[i].face_bbox[1] + height/2)
        # Reference point of the facial bounding box is the mid point.
        size_x = int(width/2); size_y = int(height/2)
        counter  = 0; average  = 0; offset = 10
        for y in range(reference_x - size_x + offset, reference_x + size_x + offset):
            for x in range(reference_y - size_y + offset, reference_y + size_y + offset):
                x = max(min(x, img_height-1), 0)
                y = max(min(y, img_width-1), 0)
                if radiometric:
                    average += thermal[x, y]
                else:
                    average += imageToProcess[x, y][0]
                counter += 1
        if counter != 0:
            average = average / counter
        if counter != 0:
            if radiometric:
                temperature = average # Print some data about temperature
                print("Face rect temperature: T:{0:.2f}C".format(temperature))
            else: # Get pixel value of reference point
                reference_temperature = imageToProcess[reference_px, reference_py][0] # Here is the reference temperature. We are getting the pixel value.
                temperature = (average * float(config.selected_reference_temperature)) / reference_temperature
                if temperature > float(config.limit_temperature):
                    detections[i].danger_temperatue = True # Defaulter if the temperature > limit-temperature.
    return detections

def get_defaulters(frame, frame_idx, num_valid_persons, detections):
    dangerous_facial_bounding_boxes = {} # Dictionary to store the ids and the bounding boxes of the faces of the defaulters.
    sd_defaulting_ids   = [] # Ids of persons who are not following social distancing or have gathered in a crowd only.
    temp_defaulting_ids = [] # Ids of persons who have higher temperature of the face then the limit only.
    mask_defaulting_ids = [] # Ids of the persons who are not wearing mask only.
    for det in detections:
        if det.danger_social_distance is True:
            sd_defaulting_ids.append(det.id)
        if det.danger_temperatue is True:
            temp_defaulting_ids.append(det.id)
        if det.danger_mask is True:
            mask_defaulting_ids.append(det.id)

    all_defaulting_ids       = list(set(sd_defaulting_ids).intersection(temp_defaulting_ids).intersection(mask_defaulting_ids)) # All the ids who have violated all the conditions.
    sd_defaulting_ids_only   = [id for id in sd_defaulting_ids if id not in all_defaulting_ids]
    temp_defaulting_ids_only = [id for id in temp_defaulting_ids if id not in all_defaulting_ids]
    mask_defaulting_ids      = [id for id in mask_defaulting_ids if id not in all_defaulting_ids]
    for det in detections:
        bbox = det.to_tlbr()
        id = det.id
        if (id not in sd_defaulting_ids_only) and (id not in temp_defaulting_ids_only) and (id not in mask_defaulting_ids):
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2) # Not defaulter at all - GREEN CODED.
        else:
            if det.face_bbox is not None:
                dangerous_facial_bounding_boxes[id] = det.face_bbox # Facial Bounding box of defaulters.
            if (id in all_defaulting_ids):
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2) # RED COLOR for the people who have violated all the norms.
            elif id in sd_defaulting_ids_only:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2) # BLUE COLOR for the people who have violated only the single norms.

    # Printing the statistics.
    print("In frame %d we have a total of %d persons detected" % (frame_idx, num_valid_persons))
    print("(A) We have found %d people who are violating Social Distancing only." % (len(sd_defaulting_ids_only)))
    print("(B) We have found %d people who are having a higher body temperature only." % (len(temp_defaulting_ids_only)))
    print("(C) We have found %d people who are not wearing the mask only" % (len(mask_defaulting_ids)))
    print("(D) We have found %d who are having a higher body temperature, violating social "
          "distancing norms and not wearing masks." % (len(all_defaulting_ids)))
    return list(dangerous_facial_bounding_boxes.values()), frame # Returning the bounding boxes of the people who are defaulters.

def main(detector, mask_detector, config): # Definition of the parameters
    nms_max_overlap = config.nms_max_overlap # A condition for NMS overlap
    writeVideo_flag = config.writeVideo_flag # To write the output frame as a video.
    use_video_flag  = config.use_video_flag # Flag when True, the input is Video, else the input are images.
    min_confidence  = config.min_confidence # Minimum Confidence of detecting a person bounding box.

    face_detector       = MTCNN() # Face detector model.
    required_size_faces = (160, 160) # Required size of the face.
    face_net            = load_model(config.face_net_file) # Load the Face Detector model.
    stored_data         = np.load(config.stored_facial_embeddings) # Load the stored facial data.
    facial_stored_data  = stored_data['arr_0']  # Stored Embeddings of the Faces
    meta_data           = stored_data['arr_1']  # Names of the Stored Embeddings.

    names_file   = config.video_output_file.split('.')[0] + '.txt'
    txt_file_ptr = open(names_file, "w")

    if use_video_flag:
        file_path     = config.input_file_path if config.input_file_path !='' else 'vid_short.mp4'
        video_capture = cv2.VideoCapture(file_path) # Capture the Video.
        h             = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w             = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps           = int(video_capture.get(cv2.CAP_PROP_FPS))
        max_frame_idx = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    else:
        file_path      = config.input_file_path if config.input_file_path !='' \
                                else '/Users/Apple/Desktop/DECOVIT/Train-YOLOV3/DATASET/MOT16/train/MOT16-02'
        frames         = glob.glob(os.path.join(file_path, 'img1', '*.jpg')) # Obtain the images.
        max_frame_idx  = len(frames)
        info_filename  = os.path.join(file_path, "seqinfo.ini")
        if os.path.exists(info_filename):
            with open(info_filename, "r") as f:
                line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
                info_dict = dict(s for s in line_splits if isinstance(s, list) and len(s) == 2)
            h   = int(info_dict["imHeight"]) # Obtain the Heigth, Width and Frame-Rate (according to MOT-16 dataset.)
            w   = int(info_dict["imWidth"])
            fps = int(info_dict["frameRate"])
        else:
            # Set the Heigth, Width and Frame-Rate (according to MOT-16 dataset.) to default values.
            h   = 1080
            w   = 1920
            fps = 30

    update_ms = 1000 / float(fps)
    if writeVideo_flag:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out    = cv2.VideoWriter(config.video_output_file, fourcc, 30, (w, h))
    fps_imutils = imutils.video.FPS().start()
    frame_idx   = 0
    while True:
        if use_video_flag:
            ret, frame = video_capture.read()  # frame shape 640*480*3
            if ret != True:
                 break
        else:
            frame = cv2.imread(frames[frame_idx], cv2.IMREAD_COLOR)

        frame_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_idx += 1
        if frame_idx == 1:
            if config.click:
                '''
                Here we need the pixel position of reference temperature "reference_px" and "reference_px" (NEEDED for AI-Thermometer)  
                AND the reference width in the image which is 1 FEET AWAY (Needed for Social Distancing)
                '''
                reference_px, reference_py = initial_requirements_temperature(frame=frame,
                                                                              selected_reference_temperature=config.selected_reference_temperature)
                reference_width            = initial_requirements_width(frame=frame)
            else:
                print("NO clicking is needed")
                reference_px = 300
                reference_py = 300
                reference_width = 20
                print("The X and Y coordinate of the reference temperature pixel is %d and %d respectively."%(reference_px, reference_py))
                print("The value of the reference width is %d "%(reference_width))

        if (frame_idx%config.frames_skip == 0) or (frame_idx == 1):
            print("\n Frame %05d/%05d" % (frame_idx, max_frame_idx), end=' ')
            t1          = time.time()
            U1          = detector.detect_image(frame)
            boxes       = U1[0] # person bounding boxes.
            confidence  = U1[1] # Confidence of the person bounding boxes.
            if len(boxes) == 0:
                continue
            '''
            ## You should consider each detection in "detections" as a person.
            '''
            detections  = [Detection(bbox, confidence, None, config.ref_width) for bbox, confidence in zip(boxes, confidence)] # Person bounding boxes.
            detections  = [d for d in detections if d.confidence >= min_confidence] # Prune the bounding boxes which are of low confidence.
            boxes       = np.array([d.tlwh for d in detections])
            scores      = np.array([d.confidence for d in detections])
            indices     = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)  # Run non-maxima suppression.
            detections  = [detections[i] for i in indices] # Prune of NMS.
            num_valid_persons = len(detections) #Number of valid person bounding boxes.
            if num_valid_persons > 0:
                detections = mask_detector.detect_mask(image=frame_mask, num_detected_person_boxes=num_valid_persons, detections=detections)

                for i in range(0, num_valid_persons):
                    detections[i].id = str(i) # Assigning unique id for each person bounding boxes.

                for det in detections:
                    bbox = det.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    if det.face_bbox is not None:
                        face_bbox = det.face_bbox
                        cv2.rectangle(frame, (int(face_bbox[0]), int(face_bbox[1])), (int(face_bbox[2]), int(face_bbox[3])), (255, 255, 255), 2)
                    '''
                    ## I need to get the facial bounding boxes from vijay and design an algorithm which to assign the faces to the desired persons.
                    ## If that facial bounding box is not wearing a mask, make sure that the flag "danger_mask" is set to True.
                    '''
                detections = find_defaulters(detections=detections, crowd_max=config.crowd_max,
                                             social_distance_in_feet=config.social_distance_in_feet,
                                             reference_width=reference_width, num_detected_persons=num_valid_persons)

                detections = find_defaulters_temperature(detections=detections, frame=frame,
                                                         frame_path=None, config=config,
                                                         reference_px=reference_px, reference_py=reference_py,
                                                         img_width=w, img_height=h) # TODO FRAME PATH
                dangerous_facial_bounding_boxes, frame = get_defaulters(frame=frame, frame_idx=frame_idx,
                                                                        detections=detections, num_valid_persons=num_valid_persons)
                if len(dangerous_facial_bounding_boxes) > 0:
                    defaulter_names = perform_defaulter_face_recognition(dangerous_facial_bounding_boxes=dangerous_facial_bounding_boxes,
                                                                          frame=frame_mask, face_net=face_net,
                                                                          required_size_faces=required_size_faces,
                                                                          face_detector = face_detector,
                                                                          stored_facial_embeddings= facial_stored_data,
                                                                          meta_data=meta_data)
                    LINE = [str(frame_idx)]
                    for name in defaulter_names:
                       LINE.append(str(name))
                    LINE = ' '.join(LINE)
                    txt_file_ptr.write(LINE)
                    txt_file_ptr.write("\n")

            if writeVideo_flag:
                out.write(frame)
            fps_imutils.update()
            fps = (fps + (1./(time.time()-t1))) / 2
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    txt_file_ptr.close()
    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))
    video_capture.release()
    if writeVideo_flag:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('DECOVIT')
    # parser.add_argument('--max_cosine_distance', type = float,          default=0.3)
    # parser.add_argument("--nn_budget",           type=int,              default=None,   help="Maximum size of the appearance descriptors gallery. If None, no budget is enforced.")

    parser.add_argument("--writeVideo_flag",     action='store_true',   default = True) # Change to False
    parser.add_argument("--input_file_path",     type=str,              default='') ## Add required true here
    parser.add_argument("--use_video_flag",      action='store_true',   default = False)  # Argument
    parser.add_argument('--video_output_file',   type=str,              required=True)

    parser.add_argument('--choose_detector',         type=int,          default=0)
    parser.add_argument("--min_confidence",          type=float,        default=0.3,
                        help="Detection confidence threshold. Disregard all detections that have a confidence lower than this value.")
    parser.add_argument("--nms_max_overlap",         type=float, default=1.0)  # Argument
    # parser.add_argument("--model_filename",          type=str,          default='model_data/mars-small128.pb')  # Argument # Deep SORT)
    parser.add_argument('--click', action='store_true', default=False)  # Argument
    parser.add_argument("--ref_width",               type=float,        default=1.0)
    parser.add_argument("--social_distance_in_feet", type=int,          default=4)
    parser.add_argument("--crowd_max",               type=int,          default=10)
    parser.add_argument("--frames_skip",             type=int,          default=5)

    parser.add_argument("--selected_reference_temperature",   default="25.5",    help="Reference temperature")
    parser.add_argument("--limit_temperature",                default="37.5",    help="Limit temperature")
    # parser.add_argument("--radiometric",                      default="False",   help="User radiometric temperature, else reference temperature is used")

    parser.add_argument('--face_net_file',              type=str,            default = 'model_data/facenet_keras.h5')
    parser.add_argument('--stored_facial_embeddings',   type=str,            default = 'data/5-celebrity-faces-embeddings.npz')
    parser.add_argument('--mask_detector_file',         type=str,            default = "model_data/mask-detector-final.h5", help="THE .h5 file should be stored in model_data folder.")
    parser.add_argument('--mask_backbone',              type=str,            default = 'resnet50')

    config = parser.parse_args()
    if config.choose_detector == 0:
        print("WILL BE CHOOSING THE YOLO DETECTOR trained on COCO-Dataset.")
        # main(YOLO(), YOLO_MASK(), config)
        main(YOLO(), RETINANET(filename=config.mask_detector_file, backbone=config.mask_backbone), config)
    elif config.choose_detector == 1:
        print("WILL BE CHOOSING THE MOBILE-NET Detector trained on COCO-Dataset.")
        # main(MobileNet(), YOLO_MASK(), config)
        main(MobileNet(), RETINANET(), config)
    else:
        raise ValueError("WRONG DETECTOR SELECTED")


'''

    # metric              = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # tracker             = Tracker(metric)
    # max_cosine_distance = config.max_cosine_distance
    # nn_budget = config.nn_budget
    # model_filename      = config.model_filename ## For generating the features.
    # encoder             = gdet.create_box_encoder(model_filename, batch_size=1)  ## LOOK HERE FOR INCREASING THE SPEED.


                # tracker.predict() # Call the tracker
                # tracker.update(detections)
                # for track in tracker.tracks:
                #     if not track.is_confirmed() or track.time_since_update > 1:
                #         continue
                #     bbox = track.to_tlbr()
                #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                #     cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
                


'''