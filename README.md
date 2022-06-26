# Classification-prior-to-Restoration
Weather image noise (rain, snow, haze) removal with classification 

Classification: Mobilenet
Denoising: U-Net

Overall Architecture 

### Classifier: Edge and Luminance metric are used for better classification. 

* Notice: Rainy and snowy days tended to have low luminance and high edge components 


![image](https://user-images.githubusercontent.com/58447982/175798354-ee64143e-eab3-4815-97f3-6d06fc941f76.png)

### Restoration: Pyramidal U-Net

* U-Net has proven its power in denoising. Base on U-Net architecture, we adopted pyramid structure to enhance receptive field acceptance.


![image](https://user-images.githubusercontent.com/58447982/175798417-303cbf04-8760-4ed8-90d1-20121af02bbf.png)


For more detail, please check "Weather Classification and Image Restoration Algorithm Attentive to Weather Conditions in Autonomous Vehicle.pdf"

Our demo is on youtube.


## Dehaze: https://youtu.be/sFpYvlNCd0s


## Desnow: https://youtu.be/tdBYfQB3czU


## Derain: https://youtu.be/knX-aSEZKbE


![image](https://user-images.githubusercontent.com/58447982/175050139-2d277607-43c9-4137-a679-626585835b5b.png)
