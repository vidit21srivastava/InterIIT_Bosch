import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import numpy as np
from tensorflow.keras.utils import get_file
from pathlib import Path
from omegaconf import OmegaConf
from yu4u.src.factory import get_model
import cv2


class AgeAndGenderEstimator_FairFace():
  def __init__(self):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.model_fair_7 = torchvision.models.resnet34(pretrained=True)
    self.model_fair_7.fc = nn.Linear(self.model_fair_7.fc.in_features, 18)
    self.model_path='./models/res34_fair_align_multi_7_20190809.pt'
    self.model_fair_7.load_state_dict(torch.load(self.model_path))
    self.model_fair_7 = self.model_fair_7.to(self.device)
    self.model_fair_7.eval()
  
  def predict_age_and_gender(self, im):
      trans = transforms.Compose([
          transforms.ToPILImage(),
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
      
      im = trans(im)
      im = im.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
      im = im.to(self.device)
      outputs = self.model_fair_7(im)
      outputs = outputs.cpu().detach().numpy()
      outputs = np.squeeze(outputs)

      gender_outputs = outputs[7:9]
      age_outputs = outputs[9:18]

      gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
      age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

      gender_pred = np.argmax(gender_score)
      age_pred = np.argmax(age_score)

      predicted_gender = "Male" if gender_pred == 0 else "Female"

      dict = {0: '0-2 yrs', 1: '3-9 yrs', 2: '10-19 yrs', 3: '20-29 yrs',
              4: '30-39 yrs', 5: '40-49 yrs', 6: '50-59 yrs', 7: '60-69 yrs',
              8: '70+ yrs'}
      predicted_age = dict[age_pred]
      
      return predicted_gender, predicted_age, gender_score, age_score

class AgeAndGenderDetection_YU4U():
  def __init__(self):
    pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
    modhash = '6d7f7b7ced093a8b3ef6399163da6ece'
    weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
                                  file_hash=modhash, cache_dir=str(Path('./models/').resolve().parent))
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    self.model = get_model(cfg)
    self.model.load_weights(weight_file)

  def predict_age_and_gender(self, im):
    im = cv2.resize(im, (224, 224))
    # print(im.shape)
    results = self.model.predict(np.array([im]))
    gender_score = results[0][0][0]
    predicted_gender = "Male" if gender_score<0.5 == 0 else "Female"
    # print(gender_score)
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_age = results[1].dot(ages).flatten()[0]
    # ages_var = (np.arange(0, 101).reshape(101, 1)-predicted_age)**2
    # variance = results[1].dot(ages_var).flatten()[0]
    return predicted_gender, predicted_age, gender_score, None
