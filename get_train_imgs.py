#harvest streetview images from western pennsylvania regional data center online database
import pickle
import numpy as np
import urllib.request
import os
import pdb
import datetime

fname  = 'parIDs'
#load in parcel IDs
with open(fname, 'rb') as f:
	parcel_IDs = pickle.load(f)

#'https://iasworld.alleghenycounty.us/iasworld/iDoc2/Services/GetPhoto.ashx?parid=0176F00135000000&jur=002&Rank=1&size=350x263'
img_url = 'https://iasworld.alleghenycounty.us/iasworld/iDoc2/Services/GetPhoto.ashx?parid='
img_params = '&jur=002&Rank=1&size=350x263'
problem_imgs =[] #couldnt load the photos
success_imgs = []
char015_imgs = []
nids = len(parcel_IDs)
benchmarks = np.linspace(0,len(parcel_IDs),int(nids/500)).astype('int')
ct =0
for parcel in parcel_IDs:
	try:
		get_img = img_url + parcel + img_params
		urllib.request.urlretrieve(get_img, 'images/'+parcel+'.jpg')	
		success_imgs.append(parcel)	
	
	except:
		try: # try 15-digit instead of 16-digit parcel ID
			subpar = parcel[:-1] # some images 15-digit, not 16-digit!
			get_img = img_url + subpar + img_params
			urllib.request.urlretrieve(get_img, 'images/'+parcel+'.jpg')	
			success_imgs.append(parcel)			
			char015_imgs.append(parcel)	
		except:
			problem_imgs.append(parcel)

	if (ct==benchmarks[0]):
		print('Attempted to downloaded: '+str(ct)+ ' photos to cwd. '+ str(datetime.datetime.now()))
		with open('success_dict', 'wb') as f:
			parcel_IDs = pickle.dump(success_imgs, f)
		
		with open('fail_dict', 'wb') as f:
			parcel_IDs = pickle.dump(problem_imgs, f)
		print('Successes =  ' + str(len(success_imgs)) + ' Fails = ' + str(len(problem_imgs)))
		
		with open('15char_dict', 'wb') as f: #note which are missing a digit -> These may be problem images 
			# For instance -> no photo avaialble, e.g. parid: 0399B00018000000
			# These are likely to new construction (built after 2007)
			parcel_IDs = pickle.dump(char015_imgs, f)	

		benchmarks = benchmarks[1:] # advance benchmarks


	ct+=1
