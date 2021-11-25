import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import os
# https://www.kaggle.com/kmader/show-3d-nifti-images
import numpy as np
import tqdm as tqdm

data_path = './sample/data'
label_path = './sample/label'
data_file_list = sorted(os.listdir(data_path))
label_file_list = sorted(os.listdir(label_path))

ivh = 0
ich = 0

for d in range(len(data_file_list)):

    ct_path = './sample/data/'+data_file_list[d]
    ct = nib.load(ct_path)
    ct = ct.get_fdata()

    mask_path = './sample/label/'+label_file_list[d]
    mask = nib.load(mask_path)
    mask = mask.get_fdata()

    ct = np.transpose(ct,(2,0,1))
    mask = np.transpose(mask,(2,0,1))

    # ct data slice
    ct = ct[:32, :512, :512]
    mask = mask[:32, :512, :512]

    c, w, h = ct.shape
    if w!=512 or h!=512:
        print(data_file_list[d])

    if c < 32:
        z_padding = 32-c
        ct = np.pad(ct, ((0, z_padding), (0, 0), (0, 0)), 'constant')
        mask = np.pad(mask, ((0, z_padding), (0, 0), (0, 0)), 'constant')

    for i in range(len(ct)):
        ct[i] = np.where(ct[i] < 0, 0, ct[i])
        ct[i] = np.where(ct[i] > 140, 255, ct[i])
        ct[i] = np.where(ct[i] == 255, ct[i]/140*255, ct[i])

        c_path = './sample_preprocessing/data/' + data_file_list[d][:3]
        m_path = './sample_preprocessing/label/' + label_file_list[d][:4]

        # print(m_path, i+1)
        # print(np.where(mask[i] == 1, True, False).sum())

        # ich_check = np.where(mask[i] == 1, True, False)
        # if ich_check.sum() != 0:
        #     ich += 1
        #
        # ivh_check = np.where(mask[i] == 2, True, False)
        # if ivh_check.sum() != 0:
        #     print(m_path, i+1)
        #     ivh += 1

        if not os.path.exists(c_path):
            os.makedirs(c_path)
        if not os.path.exists(m_path):
            os.makedirs(m_path)

        c_path2 = c_path + '/{:03d}.png'.format(i+1)
        m_path2 = m_path + '/m{:03d}.png'.format(i+1)


        # cv2.imwrite(c_path2, ct[i])
        # cv2.imwrite(m_path2, mask[i])

        # # 시각화해서 검증시
        plt.imsave(c_path2, ct[i])
        plt.imsave(m_path2, mask[i])

