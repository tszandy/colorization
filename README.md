instruction:
requirement install:
    pytorch
    torchvision
    opencv_python
    numpy
    cuda (for training)
put resized image into training data inside ./train_image/dir/picture
then run any trainUnet*.py to train different network

trainUnet2 predict UV int8 layer with 128 mean
trainUnet3 predict UV float32 layer with 0 mean
trainUnet4 grey scale predict RGB
trainUnet5 line sketch predict RGB
trainUnet6 line sketch predict grey scale
trainUnet7 pretrain line sketch predict RGB as generator(trainUnet5 model) then use gan discriminate picture

after training and get unet*.pth
put image that want to be predicted into ./input_image folder then run convert.py to convert image into grey scale and line sketch

then run print_result.ipynb it will send output to different ./output_image* folder
