from pipeline import AgeAndGenderDetector

# example code to run our api
if __name__ == '_main_':
  detector = AgeAndGenderDetector()
  # Uncomment the following to run the model on video

  # input_video_path = '<path of the video (including the file name)>'
  # output_video_path = '<path of the output video (including the file name)>'
  # csv_video_path = '<path of the csv file(including the file name)>'
  # detector.annotate_age_and_gender_video(input_video_path, output_video_path, output_video_path)

  # Uncomment the following to run the model on image
  # detector = AgeAndGenderDetector()
  # input_image_path = '<path of the image (including the file name)>'
  # output_image_path = '<path of the output image (including the file name)>'
  # csv_image_path = '<path of the csv file(including the file name)>'
  # detector.annotate_age_and_gender_image(input_image_path, output_image_path, csv_image_path)