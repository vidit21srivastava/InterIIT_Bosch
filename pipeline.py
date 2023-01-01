import cv2
from deepface import DeepFace
from deepface.DeepFace import build_model
from deepface.detectors.FaceDetector import detect_faces
from deepface.detectors.FaceDetector import build_model as build_face_detector
from deepface.commons import functions, realtime, distance as dst
import numpy as np
import csv

# Our Libraries
from age_and_gender import AgeAndGenderDetection_YU4U, AgeAndGenderEstimator_FairFace


class AgeAndGenderDetector():
  def __init__(self):
    self.MAX_CENTROID_THRESOLD = 100
    self.FACE_DETECTOR_BACKEND = 'retinaface'
    self.DB_FACE_SIZE_THRESOLD = 100
    self.DB_FRAME_SIZE_THRESOLD = 100
    self.DST_EUCLIDEAN_L2_THRESHOLD = 0.5

    self.deepface_face_detector = build_face_detector(
        self.FACE_DETECTOR_BACKEND)
    self.deepface_vggface = build_model('VGG-Face')
    self.age_and_gender_estimator_FF = AgeAndGenderEstimator_FairFace()
    self.age_and_gender_estimator_Y = AgeAndGenderDetection_YU4U()

  '''
  Draws bounding boxes with age, gender and Tracking ID
  '''

  def draw_rectangle(self, img, pos, gender, age, id):
    xmin, ymin, xmax, ymax = pos
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (36, 255, 12), 1)
    face_label = '[' + str(id) + ']Age:' + str(age) + ' Gender: ' + str(gender)
    cv2.putText(img, face_label, (xmin, ymin+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return img

  '''
  Computes distance between centres of 2 bounding boxes
  '''

  def centroid_score(self, boxA, boxB):
    x_cA, y_cA = boxA
    x_cB, y_cB = boxB
    c_score = np.sqrt((x_cB-x_cA)**2 + (y_cB-y_cA)**2)
    return c_score

  def write_to_csv(self, rows, writer):
    for data in rows:
      xmin = data['bbox'][0]
      ymin = data['bbox'][1]
      w = data['bbox'][2]-data['bbox'][0]
      h = data['bbox'][3]-data['bbox'][1]
      # row = str(data['frame']) + ',' + str(data['id']) + ',' + str(xmin) + ',' + str(ymin) + ',' + str(h) + ',' + str(w) + ',' + str(
      #     data['age_min']) + ',' + str(data['age_max']) + ',' + str(data['age_max']) + ',' + str(data['age']) + ',' + str(data['gender'])
      row = [data['frame'], data['id'], xmin, ymin, h, w, data['age_min'],
             data['age_max'], data['age_max'], data['age'], data['gender']]
      writer.writerow(row)

  """Annotate the faces in the input video with age and gender.

  Args:
    video_path (str): Path to the input video (if None, then webcam video feed is used).
    output_path (str): Path to store the output video.
    csv_path (str): Path to store the output csv file
  """

  def annotate_age_and_gender_video_mode(self, video_path, output_path, csv_path):
    if video_path is None:
      video_path = 0
    # read from the video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(
        output_path,
        fourcc,
        fps, (width, height)
    )
    csv_file = open(csv_path, 'w')
    csv_writer = csv.writer(csv_file)

    frame_number = 1
    db_faces = []
    db_frames = []
    id = 1
    db_ids = {}
    count = 0

    while True:
      success, image = cap.read()

      if(not success):
        break

      count = count + 1
      if count % 100 == 0:
        print(count)

      faces = detect_faces(self.deepface_face_detector,
                           self.FACE_DETECTOR_BACKEND, image)
      faces_in_frame = []
      new_faces = []
      for face in faces:
        x, y, w, h = face[1]

        x1 = max(0, int(x-1*w))
        x2 = min(image.shape[1], int(x+2*w))
        y1 = max(0, int(y-1*h))
        y2 = min(image.shape[0], int(y+2*h))

        face_img = image[y1:y2, x1:x2]
        curr_centroid = np.array([(x1+x2)/2, (y1+y2)/2])
        face_img = cv2.resize(face_img, (224, 224))
        face_embd = self.deepface_vggface.predict(
            np.array([face_img]))[0].tolist()

        curr_id = id
        if len(db_faces) == 0:
          # print(id)
          new_faces.append({
              'embd': face_embd,
              'id': id,
              'img': face_img,
              'centroid': curr_centroid,
          })
          id = id+1
        else:
          min_face_score = None
          for idx, db_embd in enumerate(db_faces):

            prev_centroid = db_embd['centroid']
            score = self.centroid_score(curr_centroid, prev_centroid)
            if score > self.MAX_CENTROID_THRESOLD:
              continue

            distance = dst.findEuclideanDistance(dst.l2_normalize(
                db_embd['embd']), dst.l2_normalize(face_embd))
            if distance > self.DST_EUCLIDEAN_L2_THRESHOLD:
              continue

            if min_face_score == None or min_face_score['distance'] > distance:
              min_face_score = {
                  'distance': distance,
                  'id': idx
              }

          if min_face_score is None:
            # No match found
            if len(db_faces) >= self.DB_FACE_SIZE_THRESOLD:
              db_faces.pop(0)
            new_faces.append({
                'id': curr_id,
                'embd': face_embd,
                'img': face_img,
                'centroid': curr_centroid
            })
            id = id+1
            continue

          db_embd = db_faces[min_face_score['id']]
          curr_id = db_embd['id']
          db_faces.pop(idx)
          new_faces.append({
              'id': curr_id,
              'embd': face_embd,
              'img': face_img,
              'centroid': curr_centroid,
          })

        genderFF, ageFF, gender_scoreFF, age_scoreF = self.age_and_gender_estimator_FF.predict_age_and_gender(
            face_img)
        genderY, ageY, gender_scoreY, age_scoreY = self.age_and_gender_estimator_Y.predict_age_and_gender(
            face_img)

        if curr_id not in db_ids.keys():
          db_ids[curr_id] = {}
          db_ids[curr_id]['age'] = ageY
          db_ids[curr_id]['gender'] = gender_scoreFF
          db_ids[curr_id]['num_occurences'] = 1
          db_ids[curr_id]['num_active'] = 1
        else:
          db_ids[curr_id]['age'] = (db_ids[curr_id]['age']*db_ids[curr_id]
                                    ['num_occurences']+ageY)/(db_ids[curr_id]['num_occurences']+1)
          db_ids[curr_id]['gender'] = (db_ids[curr_id]['gender']*db_ids[curr_id]
                                       ['num_occurences']+gender_scoreFF)/(db_ids[curr_id]['num_occurences']+1)
          db_ids[curr_id]['num_occurences'] = db_ids[curr_id]['num_occurences']+1
          db_ids[curr_id]['num_active'] = db_ids[curr_id]['num_active']+1  # to do

        faces_in_frame.append({
            'id': curr_id,
            'bbox': [x1, y1, x2, y2]
        })

      for new_face in new_faces:
        db_faces.append(new_face)

      if len(db_frames) >= self.DB_FRAME_SIZE_THRESOLD:
        out_frame = db_frames[0]
        db_frames.pop(0)
        # fill info for out_frame
        out_img = out_frame['image']
        frame_csv_data = []
        for face_info in out_frame['faces']:
          face_id = face_info['id']
          age = db_ids[face_id]['age']
          gender = db_ids[face_id]['gender']
          gender_pred = np.argmax(gender)
          predicted_gender = "M" if gender_pred == 0 else "F"
          frame_csv_data.append({
              'frame': frame_number,
              'id': face_info['id'],
              'gender': predicted_gender,
              'age': age,
              'age_min': age,
              'age_max': age,
              'bbox': face_info['bbox']
          })
          self.draw_rectangle(out_img, face_info['bbox'],
                              predicted_gender, int(age), face_id)
          if db_ids[face_info['id']]['num_active'] <= 0:
            db_ids.pop(face_info['id'], None)
          db_ids[face_info['id']
                 ]['num_active'] = db_ids[face_info['id']]['num_active']-1
        out_video.write(out_img)
        # print(frame_number)
        frame_number = frame_number+1
        self.write_to_csv(frame_csv_data, csv_writer)

      db_frames.append({
          'image': image,
          'faces': faces_in_frame
      })

    # print(db_ids)
    for out_frame in db_frames:
      # fill info for out_frame
      out_img = out_frame['image']
      frame_csv_data = []
      for face_info in out_frame['faces']:
        face_id = face_info['id']
        age = db_ids[face_id]['age']
        gender = db_ids[face_id]['gender']
        gender_pred = np.argmax(gender)
        predicted_gender = "M" if gender_pred == 0 else "F"
        frame_csv_data.append({
            'frame': frame_number,
            'id': face_info['id'],
            'gender': predicted_gender,
            'age': age,
            'age_min': age,
            'age_max': age,
            'bbox': face_info['bbox']
        })
        self.draw_rectangle(out_img, face_info['bbox'],
                            predicted_gender, int(age), face_id)
        if db_ids[face_info['id']]['num_active'] <= 0:
          db_ids.pop(face_info['id'], None)
        db_ids[face_id]['num_active'] = db_ids[face_id]['num_active']-1
      out_video.write(out_img)
      self.write_to_csv(frame_csv_data, csv_writer)

    out_video.release()
    cap.release()
    csv_file.close()


  """Annotate the faces in the input image with age and gender.

  Args:
    input_image (image/str): Image/Path to the input image.
    output_path (str): Path to store the output image.
    csv_path (str): Path to store the output csv file
  """
  def annotate_age_and_gender_image(self, image, output_path, csv_path):
    if(type(image) == str):
      image = cv2.imread(image)

    csv_file = open(csv_path, 'w')
    csv_writer = csv.writer(csv_file)

    faces = detect_faces(self.deepface_face_detector,
                         self.FACE_DETECTOR_BACKEND, image)

    id = 1
    frame_number = 1
    frame_csv_data = []

    for face in faces:
      x, y, w, h = face[1]

      x1 = max(0, int(x-1*w))
      x2 = min(image.shape[1], int(x+2*w))
      y1 = max(0, int(y-1*h))
      y2 = min(image.shape[0], int(y+2*h))

      face_img = image[y1:y2, x1:x2]

      genderFF, ageFF, gender_scoreFF, age_scoreF = self.age_and_gender_estimator_FF.predict_age_and_gender(
          face_img)
      genderY, ageY, gender_scoreY, age_scoreY = self.age_and_gender_estimator_Y.predict_age_and_gender(
          face_img)
      age = ageY
      gender_pred = np.argmax(gender_scoreFF)
      gender = "Male" if gender_pred == 0 else "Female"
      self.draw_rectangle(image, [x1, y1, x2, y2],
                          gender, int(age), id)
      frame_csv_data.append({
          'frame': frame_number,
          'id': id,
          'gender': gender,
          'age': age,
          'age_min': age,
          'age_max': age,
          'bbox': [x1, y1, x2, y2]
      })
      id = id + 1

    self.write_to_csv(frame_csv_data, csv_writer)
    if output_path is not None:
      cv2.imwrite(output_path, image)

    csv_file.close()
    return image
