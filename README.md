# Image retrieval System

We are thrilled to introduce our cutting-edge image retrieval system that is powered by advanced algorithms and state-of-the-art machine learning techniques. This remarkable system represents a significant leap forward in the field of image searches, offering users effortless access to highly precise results. It is with great pleasure that we bid farewell to the cumbersome task of endless scrolling and welcome an exciting new era of image discovery.

## Features


- Utilize your own dataset for image retrieval.
- Provide a variety of feature descriptors.
- Offer a diverse range of search methods.
- Experiment with your own camera.
- Visualize your results.

## Technology

Here are the techs we implemented for this project

-  [Python](https://nodejs.org/): 
-  [Python](https://nodejs.org/): 
-  [Python](https://nodejs.org/): 

## Installation
Clone the repository

```sh
git clone https://github.com/lechithinh/Image-Retrival.git
```

Install the libraries

```sh
pip install -r requirement.txt
```
## Structure

Please organize your dataset following this structure: 

```
IMAGE_RETRIEVAL/
│
├── Asset/ 
|   |   ├── asset1.png
|   |   ├── asset2.png
|   |   ├── asset3.png
├── dataset/ 
|   |   ├── Black_dress (1).png
|   |   ├── Black_dress (2).png
|   |   ├── Black_dress (3).png
|   |   ├── ....
|   |   ├── Red_dress(30).png
├── feature/
|   |   ├── LBP.index.bin
|   |   ├── LBP.csv
|   |   ├── RGBHistogram.index.bin
|   |   ├── RGBHistogram.csv
|   |   └── VGG16.index.bin
|   |   └── VGG16.csv
├── app.py
├── ...
├─ main.py
└── ...
```

## Dataset

The dataset for this project is [Fashion Dataset](https://nodejs.org/)
+ There are 24 class
+ Each class contains 30 images
+ The total images is 720

The test dataset for this project [Test Dataset](https://nodejs.org/)
+ There are 24 class
+ Each class contains 7 images
+ The total images is 168

Be sure follow `The same structure` .


## License

MIT


