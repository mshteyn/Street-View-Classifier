from flask import Flask, render_template, jsonify, request
app = Flask(__name__)
import torch
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
    prediction = predict(new_img)
    print(prediction, sep = "\n")
    io.imsave('static/user_img.jpg', new_img)

    #return prediction
    return render_template('predict.html', image = new_img, p1=prediction.split('\n')[0],
                           p2=prediction.split('\n')[1], p3=prediction.split('\n')[2])



def transform_image(image):
        #image = io.imread(img_path)
        image = resize(image,(232,320),anti_aliasing=True) # for RESNET50 model
        #image=resize(image,(384,384),anti_aliasing=True) #for vit_B model
        #image=resize(image,(256,256),anti_aliasing=True) #for vit_L model
        # apply_TF= transforms.Compose([transforms.Normalize( # Normalize for Nrm network
        #                                     [0.485, 0.456, 0.406],
        #                                     [0.229, 0.224, 0.225])])
        # tf_image = apply_TF(torch.tensor(image.T))
        tf_image = torch.tensor(image.T)

        #conform data shape to model
        rtf_image = tf_image.unsqueeze(0).float()        
        return rtf_image #return transformed image



def predict(img_path):
    rtf_image = transform_image(img_path) # get it into image form
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
