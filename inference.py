import torch
from model.sagnn import SAGNN
from util.transform import ToTensor,Normalize,Compose,Resize
import cv2
import matplotlib.pyplot as plt


def find_new_hw(ori_h, ori_w, test_size):
    if ori_h >= ori_w:
        ratio = test_size * 1.0 / ori_h
        new_h = test_size
        new_w = int(ori_w * ratio)
    elif ori_w > ori_h:
        ratio = test_size * 1.0 / ori_w
        new_h = int(ori_h * ratio)
        new_w = test_size

    if new_h % 8 != 0:
        new_h = (int(new_h / 8)) * 8
    else:
        new_h = new_h
    if new_w % 8 != 0:
        new_w = (int(new_w / 8)) * 8
    else:
        new_w = new_w
    return new_h, new_w

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]
inf_transform = [Resize(473),ToTensor(),Normalize(mean, std)]

model = SAGNN(device_number=1)
model.load_state_dict(torch.load("best_model_path",map_location="cuda:1")["state_dict"])
model = model.cuda(1)
model.eval()
transforms = Compose(inf_transform)
support_image = cv2.imread("./mysamples/support1.jpg",cv2.IMREAD_COLOR)
support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
support_gt = cv2.imread("./mysamples/support1_gt.png",cv2.IMREAD_GRAYSCALE)
query_image = cv2.imread("./mysamples/query1.jpg",cv2.IMREAD_COLOR)
query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
s_inp, smask = transforms(support_image,support_gt)
q_inp, qmask = transforms(query_image,support_gt)
s_inp = s_inp.cuda(device=1)
smask = smask.cuda(device=1)
q_inp = q_inp.cuda(device=1)
s_inp = s_inp.unsqueeze(0).unsqueeze(0)
smask = smask.unsqueeze(0).unsqueeze(0)
q_inp = q_inp.unsqueeze(0)
output= model(s_x=s_inp,s_y=smask,x=q_inp,y=None)
new_h,new_w = find_new_hw(query_image.shape[0],query_image.shape[1], 473)
output = torch.nn.functional.interpolate(output[:,:,:,:], size=max(query_image.shape[:2]),mode='bilinear',align_corners=True)
output = output.max(1)[1]
fig1, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(query_image)
ax1.imshow(output.cpu()[0,:,:],'jet',alpha=0.5)
ax1.set_title("Query and its prediction")
ax2.imshow(support_image)
ax2.imshow(support_gt,'jet',alpha=0.5)
ax2.set_title("Support and its GT Mask")
plt.show()
