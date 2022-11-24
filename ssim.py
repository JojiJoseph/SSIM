import numpy as np

def ssim(img1, img2, L=255, K1=0.01, K2=0.03):
    size = img1.shape[0] * img1.shape[1]

    img1_mean = img1.mean()

    img2_mean = img2.mean()

    img1_std = np.sqrt(((img1-img1_mean)**2).sum()/(size-1))
    img2_std = np.sqrt(((img2-img2_mean)**2).sum()/(size-1))

    C1 = (L*K1)**2
    C2 = (L*K2)**2
    C3 = C2/2
    luminance_measure = (2*img1_mean*img2_mean + C1)/(img1_mean**2 + img2_mean**2 + C1)
    contrast_measure = (2*img1_std*img2_std + C2)/(img1_std**2 + img2_std**2 + C2)
    img12_std = (((img1-img1_mean)*(img2-img2_mean)).sum()/(size-1))

    structural_measure = (img12_std + C3)/(img1_std * img2_std + C3)

    res = luminance_measure * contrast_measure * structural_measure
    return res
