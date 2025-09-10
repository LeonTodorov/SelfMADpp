import albumentations as alb
import numpy as np
import cv2

class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
	def apply(self,img,**params):
		return self.randomdownscale(img)

	def randomdownscale(self,img):
		keep_ratio=True
		keep_input_shape=True
		H,W,C=img.shape
		ratio_list=[2,4]
		r=ratio_list[np.random.randint(len(ratio_list))]
		img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
		if keep_input_shape:
			img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

		return img_ds


def get_source_transforms():
		return alb.Compose([
				alb.Compose([
						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),
	
				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),
				
			], p=1.)

		
def get_gam_transforms():
    return alb.Compose([
        alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
        alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
        alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
        alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
        
    ], 
    additional_targets={f'image_': 'image'},
    p=1.)
    
    
def randaffine(img,mask):
    f=alb.Affine(
            translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
            scale=[0.95,1/0.95],
            fit_output=False,
            p=1)
        
    g=alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

    transformed=f(image=img,mask=mask)
    img=transformed['image']
    
    mask=transformed['mask']
    transformed=g(image=img,mask=mask)
    mask=transformed['mask']
    return img,mask


def crop_face(img=None, label=None, landmark=None, bbox=None, phase='train'):
    assert phase in ['train', 'test']
    H,W=len(img),len(img[0])

    assert landmark is not None
    x0,y0=landmark[:,0].min(),landmark[:,1].min()
    x1,y1=landmark[:,0].max(),landmark[:,1].max()

    w=x1-x0
    h=y1-y0
    if phase == 'train':
        w_margin = w * np.random.uniform(0.02, 0.1)
        h_margin = h * np.random.uniform(0.02, 0.1)
    else:
        w_margin = w * 0.0625
        h_margin = h * 0.0625
        
    x0_new = max(0, int(x0 - w_margin))
    x1_new = min(W, int(x1 + w_margin))
    y0_new = max(0, int(y0 - h_margin))
    y1_new = min(H, int(y1 + h_margin))
    
    img_cropped=img[y0_new:y1_new,x0_new:x1_new] if img is not None else None
    label_cropped=label[y0_new:y1_new,x0_new:x1_new] if label is not None else None

    if landmark is not None:
        landmark_cropped=np.zeros_like(landmark)
        for i,(p,q) in enumerate(landmark):
            landmark_cropped[i]=[p-x0_new,q-y0_new]
    else:
        landmark_cropped=None

    if bbox is not None:
        bbox_cropped=np.zeros_like(bbox)
        for i,(p,q) in enumerate(bbox):
            bbox_cropped[i]=[p-x0_new,q-y0_new]
    else:
        bbox_cropped=None

    return img_cropped, label_cropped, landmark_cropped, bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1), (y0_new,y1_new,x0_new,x1_new)

def reorder_landmark(landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark

def hflip(img=None, mask=None,landmark=None,bbox=None,label=None):
    H,W=img.shape[:2]

    if landmark is not None:
        landmark_new=np.zeros_like(landmark)
        landmark_new[:17]=landmark[:17][::-1]
        landmark_new[17:27]=landmark[17:27][::-1]

        landmark_new[27:31]=landmark[27:31]
        landmark_new[31:36]=landmark[31:36][::-1]

        landmark_new[36:40]=landmark[42:46][::-1]
        landmark_new[40:42]=landmark[46:48][::-1]

        landmark_new[42:46]=landmark[36:40][::-1]
        landmark_new[46:48]=landmark[40:42][::-1]

        landmark_new[48:55]=landmark[48:55][::-1]
        landmark_new[55:60]=landmark[55:60][::-1]

        landmark_new[60:65]=landmark[60:65][::-1]
        landmark_new[65:68]=landmark[65:68][::-1]
        if len(landmark)==68:
            pass
        elif len(landmark)==81:
            landmark_new[68:81]=landmark[68:81][::-1]
        else:
            raise NotImplementedError
        landmark_new[:,0]=W-landmark_new[:,0]
        
    else:
        landmark_new=None

    if bbox is not None:
        bbox_new=np.zeros_like(bbox)
        bbox_new[0,0]=bbox[1,0]
        bbox_new[1,0]=bbox[0,0]
        bbox_new[:,0]=W-bbox_new[:,0]
        bbox_new[:,1]=bbox[:,1].copy()
        if len(bbox)>2:
            bbox_new[2,0]=W-bbox[3,0]
            bbox_new[2,1]=bbox[3,1]
            bbox_new[3,0]=W-bbox[2,0]
            bbox_new[3,1]=bbox[2,1]
            bbox_new[4,0]=W-bbox[4,0]
            bbox_new[4,1]=bbox[4,1]
            bbox_new[5,0]=W-bbox[6,0]
            bbox_new[5,1]=bbox[6,1]
            bbox_new[6,0]=W-bbox[5,0]
            bbox_new[6,1]=bbox[5,1]
    else:
        bbox_new=None

    mask_new = mask[:,::-1].copy() if mask is not None else None
    img_new = img[:,::-1].copy() if img is not None else None
    label_new = label[:,::-1].copy() if label is not None else None

    return img_new, mask_new,landmark_new,bbox_new,label_new


def zoom_at_point(img, x, y, zoom_factor):
	height, width, _ = img.shape
	new_width = int(width / zoom_factor)
	new_height = int(height / zoom_factor)
	left = max(0, x - new_width // 2)
	top = max(0, y - new_height // 2)
	right = min(width, x + new_width // 2)
	bottom = min(height, y + new_height // 2)

	if left == 0:
		right = new_width
	if right == width:
		left = width - new_width
	if top == 0:
		bottom = new_height
	if bottom == height:
		top = height - new_height

	cropped_image_array = img[top:bottom, left:right]
	zoomed_image_array = cv2.resize(cropped_image_array, (width, height), interpolation=cv2.INTER_LANCZOS4)
	return zoomed_image_array
