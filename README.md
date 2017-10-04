Submissions for [Kaggle nips 2017 adversarial image competition](https://www.kaggle.com/c/nips-2017-targeted-adversarial-attack).

## How to run Attack and Targeted Attack 
1. Run [cleverhans](https://github.com/tensorflow/cleverhans)'s weight downloader. 
2. Copy corresponding weights to attack/fgsm_tweak_dual and targeted_attack/iter_target_class_tweak_dual.

## How to run Defense 
1. Download keras and h5py .whl files into defense/resnet_xception_vgg19_dual 
2. Download corresponding keras pre-trained weights and put them into defense/resnet_xception_vgg19_dual/common.  
3. Somehow fine-tune the above pre-trained weights with adversarial images, and put them under defense/resnet_xception_vgg19_dual.


