# Real-time deraining 
First, install all requirements using
!pip install -r requirements.txt

To train the deraining model, use 


python train.py --architecture CAE --mode raindrops --data_dir ../Data --traininghardness light --output_dir ./weights --size 512 --displayfigures True

architecture could be CAE or MMAE
mode could be raindrops or rainstreaks
data_dir is the directory where pair of images are stored
traininghardness could be heavy, light, or meduim. It specifies how many images you want to use for train. Heavy is only recommanded for CPU with ~1TB of ram
output_dir is where to store the weights
size is the size of the images 512*512*2
displayfigures is to display image samples from the dataset or not

There is also a jupyter notebook version called train.ipynb


To test the deraining model, use 


python test.py --type images --mode evaluate --input ./input --object_detection_test True --arch CAE --rain rainstreak --output_dir ./output  --weights ./weights --enhancer GAN

type could be images or video. It specify if we want to test the deraining on images or videos
mode could be evaluate or demo. Evaluate will compute MSE, FSIM, VIF, SSIM,  PSNR and all the needed metrics while demo will only test the model on the input
input is the directory where the input video/images to be tested are found
object_detection_test specifiy if we want to run the object detection on the images before and after deraining model 
arch is the  architecture. It could be either MMAE or GAN
rain is the type of rain to remove, it could be rainstreaks or raindrops
output_dir is where to output the resultant derained images/videos
weights is where  the enhancer and deraining model is found
enhancer specify which type of enahncer we want to use, it can be GAN or Autoencoder

There is also a jupyter notebook version called test.ipynb


The notebook train-AE-enhancer.ipynb trains the AUTOENCODER enhancer 
The notebook train_raindrops.ipynb trains the CAE for raindrop removal 




