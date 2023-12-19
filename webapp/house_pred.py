from flask import Flask, render_template, jsonify, request
app = Flask(__name__)
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from skimage.transform import rescale, resize        
from skimage import io
import torch.nn as nn  
import pdb

@app.route('/')  
def home():  
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def get_predictiction():
    image = request.files['image']
    new_img = io.imread(image) 
    io.imsave('static/user_img.jpg', new_img)

    #identify a house gestalt
    label = check_imagenet(new_img)
    acceptable_categories = ['house', 'home', 'palace', 'church', 'mosque', 'castle', 'barn']
    is_house = any([(AC in label) for AC in acceptable_categories])

    #classify houses only
    if is_house == True:
        prediction = predict(new_img)
        print(prediction, sep = "\n")
    else:
        prediction  = ' It appears you may have uploaded an image of a ' + label.upper() +'.\n Please make sure to upload an image of a HOUSE.\n\n'

    store_image(new_img)
    #io.imsave('static/user_img.jpg', new_img)

    #return prediction
    return render_template('predict.html', image = new_img, p1=prediction.split('\n')[0],
                           p2=prediction.split('\n')[1], p3=prediction.split('\n')[2])



def transform_image(image, test):
        #image = io.imread(img_path)
        if test == 'IMAGENET':
            apply_TF= transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),                                        
                                        transforms.Normalize( # Normalize for RESNET50(?)
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])

            image = Image.fromarray(np.uint8(image)).convert('RGB')      
            tf_image = apply_TF(image)

        else:
            image = resize(image,(232,320),anti_aliasing=True) # for RESNET50 model
            tf_image = torch.tensor(image.T)

        #conform data shape to model
        rtf_image = tf_image.unsqueeze(0).float()        
        return rtf_image #return transformed image



def predict(img_path):
    rtf_image = transform_image(img_path, test ='fine_tune') # get it into image form
    outputs = model.eval()
    scores = model.forward(rtf_image)
    #feature_names = ['fair market value (usd)', 'year built']
    feature_names = ['fair market value (usd)' , 'year built', 'finished living area (sqft.)']
    feature_idxs = [torch.arange(5), torch.arange(5,10), torch.arange(10,15)] # 5-class features
    y_hats = []
    out = ''

    inflation_multiplier = 1.43 #adjust for inflation
    CLV_multiplier = 1.23 #adjust for common level ratio

    import json 
    class_index = json.load(open('PghHousClass.json')) #load labels
    from numpy import array 
    labels = array(list(class_index.values()))
    for feature in torch.arange(len(feature_idxs)):
        y_hats.append(scores[0, feature_idxs[feature]].max(0)[1].item()) # which cat
        out = ''.join([out, 'The ' +feature_names[feature] + ' is likely between: ' + labels[feature_idxs[feature]][y_hats[-1]][0] + ' and ' +labels[feature_idxs[feature]][y_hats[-1]][1] + '.\n'])
    print(out)
    return out

def check_imagenet(new_img):
    inet_model = models.resnet50(pretrained=True)
    inet_model.eval()

    image = new_img
    img_tensor = transform_image(image, 'IMAGENET')
    inet_model.eval()
    outputs = inet_model.forward(img_tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    import json
    imagenet_class_index = json.load(open('imagenet_class_index.json'))    
    object_id = imagenet_class_index[predicted_idx][1]

    return object_id    

def store_image(new_img):
    #resize image prior to saving
    if new_img.shape[0]>500 or new_img.shape[1]>500:
        if new_img.shape[0]>1.2*new_img.shape[1]:
            new_img = resize(new_img, (232, int(232*1.3)),anti_aliasing=True)
        elif new_img.shape[1]>1.2*new_img.shape[0]:
            new_img = resize(new_img, (int(232*1.3), 232),anti_aliasing=True)
        else:
            new_img = resize(new_img, (232, 232),anti_aliasing=True)
    io.imsave('static/user_img.jpg', new_img)
# load model
# zipped dict.
#zip_dict = '../saved_mods/saved_mod_resenet50_FMV_YBT_FLA_64bs_e7.zip'
#import shutil
#shutil.unpack_archive(zip_dict)
#saved_dict = 'saved_mod_resenet50_FMV_YBT_FLA_64bs_e7'
saved_dict = '../saved_mods/PA_dl'

state_dict = torch.load(saved_dict, map_location=torch.device('cpu'))

num_y = 15
#model = models.resnet50(weights='IMAGENET1K_V1') # for new torch 
model = models.resnet50(pretrained=True) # deprecated old torch 

model.fc=nn.Sequential(
        nn.Linear(in_features=2048,out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024,out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512,out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128,out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32,out_features=num_y,bias=True)
        )

model.load_state_dict(state_dict)
model.eval()


if __name__ == "__main__":
    # Please do not set debug=True in production
    app.run(host="0.0.0.0", port=5050, debug=True)
