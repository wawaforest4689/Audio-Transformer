"""
Date:20250408
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def hsv_show_and_adjust(img,hvary_max=32,svary_max=32,ivary_max=32):
    if hvary_max-int(hvary_max)!=0 or hvary_max>90 or hvary_max<=0:
        print("Error: Invalid input of parameter hvary_max.")
        return
    if svary_max-int(svary_max)!=0 or svary_max>128 or svary_max<=0:
        print("Error: Invalid input of parameter svary_max.")
        return
    if ivary_max-int(ivary_max)!=0 or ivary_max>128 or ivary_max<=0:
        print("Error: Invalid input of parameter ivary_max.")
        return

    hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsi)
    plt.figure(f'image with size {img.shape}')
    plt.subplot(131)
    plt.imshow(H,cmap='gray')
    plt.title('Hue')
    plt.subplot(132)
    plt.imshow(S,cmap='gray')
    plt.title('Saturation')
    plt.subplot(133)
    plt.imshow(V,cmap='gray')
    plt.title('Intensity')
    print(np.max(H))
    print("H:\n",H)
    print("S:\n",S)
    print("V:\n",V)

    S=np.asarray(S/255,dtype=np.float32)
    V=np.asarray(V/255,dtype=np.float32)
    print("S:\n",S)
    print("V:\n",V)
    print(np.max(H),np.max(S),np.max(V))
    H2=H+np.random.randint(-hvary_max,hvary_max-1,size=H.shape)
    if np.min(H2)<0:
        H2-=np.min(H2)
    if np.max(H2)>180:
        H2=np.asarray(180*H2/np.max(H2),dtype=int)
    print("adjusted H: ",H2)

    # np.stack() often runs quicker
    H_hsi=cv2.merge([np.asarray(H2,dtype=np.uint8),np.asarray(255*S,dtype=np.uint8),np.asarray(255*V,dtype=np.uint8)])

    S2=S+np.random.randint(-svary_max,svary_max-1,size=S.shape)/255
    if np.min(S2)<0:
        S2-=np.min(S2)
    if np.max(S2)>1:
        S2=np.asarray(255*S2/np.max(S2),dtype=int)
    else:
        S2=np.asarray(255*S2,dtype=int)
    print("adjuested S: ",S2/255)
    S_hsi=cv2.merge([np.asarray(H,dtype=np.uint8),np.asarray(S2,dtype=np.uint8),np.asarray(255*V,dtype=np.uint8)])

    V2=V+np.random.randint(-ivary_max,ivary_max-1,size=V.shape)/255
    if np.min(V2)<0:
        V2-=np.min(V2)
    if np.max(V2)>1:
        V2=np.asarray(255*V2/np.max(V2),dtype=int)
    else:
        V2=np.asarray(255*V2,dtype=int)
    print("adjuested V: ",V2/255)
    V_hsi=cv2.merge([np.asarray(H,dtype=np.uint8),np.asarray(255*S,dtype=np.uint8),np.asarray(V2,dtype=np.uint8)])

    plt.figure(f'h,s,v adjustion-image with size {img.shape}')
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(H_hsi,cv2.COLOR_HSV2RGB))
    plt.title(f'Hue varies with maximum value {hvary_max}')
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(S_hsi,cv2.COLOR_HSV2RGB))
    plt.title(f'Saturation varies with maximum value {svary_max}')
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(V_hsi,cv2.COLOR_HSV2RGB))
    plt.title(f'Intensity varies with maximum value {ivary_max}')


"""
adapt to monochromatic(gray-like) images
"""
def pseudocolor_enhance(img_array,color_ch=1,levels=5,brightness=0.5):
    if levels>7 or levels<3 or levels-int(levels)>0:
        print("levels invalid, should be a integer within [3,7].")
        return
    if color_ch>4 or color_ch<1 or color_ch-int(color_ch)>0:
        print("color_ch invalid,should be a interger in [1,4].")
        return
    if brightness<0.1 or brightness>1:
        print("brightness invalid,should be within [0.1,1].")


    # 红紫蓝组合，橙黄复合，绿黄青复合，青蓝组合（特殊统计地图的展示效果最好）
    colors_d=[128,4,35,89]
    colors_u=[183,34,85,124]

    pseudo_imgs=[]
    for img in img_array:
        try:
            h,w,ch=img.shape
            print("RGB Image")
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        except:
            print("Gray Image")
            pass
        pseudocolor_hsi = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        divider = dict()
        step = int(256 / levels)
        for i in range(256):
            divider[i] = (colors_d[color_ch - 1] + int(
                (i // step) * (colors_u[color_ch - 1] - colors_d[color_ch - 1]) / levels + 0.5)) % 180
        h, w = img.shape
        for i in range(h):
            for j in range(w):
                pseudocolor_hsi[i][j][0] = divider[img[i][j]]
                pseudocolor_hsi[i][j][1] = 255
                pseudocolor_hsi[i][j][2] = int(255 * brightness + 0.5)
        # print("pseudocolor_hsi:\n", pseudocolor_hsi)
        # print(pseudocolor_hsi.shape)
        pseudo_bgr = cv2.cvtColor(pseudocolor_hsi, cv2.COLOR_HSV2BGR)
        pseudo_imgs.append(pseudo_bgr)
        # cv2.imshow(f'pseudocolor enhancing, size {img.shape}', pseudo_bgr)

    return pseudo_imgs


if __name__=="__main__":
    l_house=cv2.imread('light_house.jpg')
    # hsv_show_and_adjust(l_house,16,16,16)
    # plt.show()

    indonesia1=cv2.imread("Indonesia1.jpg")
    # hsv_show_and_adjust(indonesia1,90,128,128)
    # plt.show()


    hsv_show_and_adjust(indonesia1,16,5,5)
    gray1=cv2.imread('China.jpg',cv2.IMREAD_GRAYSCALE)
    gray2 = cv2.imread('Russian.jpeg', cv2.IMREAD_GRAYSCALE)
    gray3 = cv2.imread('Germany.jpeg', cv2.IMREAD_GRAYSCALE)
    gray4 = cv2.imread('Global.png', cv2.IMREAD_GRAYSCALE)
    gray5 = cv2.imread('COVID19.jpg', cv2.IMREAD_GRAYSCALE)
    pseudocolor_enhance([gray5],2,7,0.9)
    cv2.waitKey(0)


