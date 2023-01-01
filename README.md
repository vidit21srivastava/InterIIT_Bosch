# InterIIT 2022 Age Gender detection IITBBS

## Dependencies
---
Use python 3.5+ (We used 3.7.12)

To install the dependencies run the following commands:

```
pip install -r requirements.txt
git clone https://github.com/yu4u/age-gender-estimation.git
mv "age-gender-estimation" "yu4u"
```

## How to run the models?

### Test for video:

The following lines are to be uncommented the *main.py* file:
```python
  detector = AgeAndGenderDetector()
  input_video_path = '<path of the video (including the file name)>'
  output_video_path = '<path of the output video (including the file name)>'
  csv_video_path = '<path of the csv file(including the file name)>'
  detector.annotate_age_and_gender_video(
	input_video_path, 
	output_video_path, 
	output_video_path
)
```


### Test for image:

The following lines are to be uncommented the *main.py* file:
```python
  detector = AgeAndGenderDetector()
  input_image_path = '<path of the image (including the file name)>'
  output_image_path = '<path of the output image (including the file name)>'
  csv_image_path = '<path of the csv file(including the file name)>'
  detector.annotate_age_and_gender_image(
	input_image_path, 
	output_image_path, 
	csv_image_path
)
```

## How does it work?
---
The general pipeline is as follows:

<!-- //Image should come here -->
![Pipeline](pipeline.svg)
### Face Detection
We use [**Retina face**](https://github.com/serengil/retinaface) [[Paper](https://arxiv.org/abs/1905.00641)] for face detection in Dense crowds.(Briefly mention about how it works)

### Age Detection
We use [**DEX: Deep EXpectation of Apparent Age from a Single Image**](https://github.com/yu4u/age-gender-estimation) [[Paper](https://ieeexplore.ieee.org/document/7406390)] for Age detection. This work uses a VGG-16 based architecture and the Expectation of a 100 node output to estimate the age of a person.

### Gender Detection
We use [**FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age**](https://github.com/dchen236/FairFace) [[Paper](https://arxiv.org/abs/1908.04913)] to identify the Gender of a person. This work trains a ResNet-34 model on their collected dataset.

### Face Tracking
We use two approaches, Centroid based and Face recognition to track faces in a video. 

#### Centroid based tracking
We find the bounding boxes of faces in 2 consecutive frames, T and T+1 and we find the bounding box from T+1, whose centroid closest to a given bounding box in frame T.

#### Face Recognition based Tracking
We find the bounding boxes of faces in 2 consecutive frames, T and T+1 and we find the bounding box from T+1, whose embedding is closest in distance to a given bounding box in T. We use cosine similarity for measuring distances and a pretrained VGG-Face to embed the image.

#### How to merge the two?
We first apply face-reidentification algorithm
We first apply Person Reidentification based tracking on detected faces, and then we apply Centroid based tracking to make sure that we have identified the right person. Centroid based tracking ensures that we have have a spatial correlation in detected faces, while face recognition helps in semantically mapping two faces.

### Averaging age and Gender Probability over the Tracked Trajectories

The age and gender is being calculated by averaging the results over the series of 100 frames. 

Once we have the identified a person and we have tracked the person, the age is calculated by averaging the values of age obtained from the age detection function. The gender of the person is determined by taking the maximum count of the classification obtained for a given person.


