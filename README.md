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

## Structure

Please organize your code following this structure: 

```
Main-folder/
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

## How to run
1. Create the virtual environment
    ```sh
    python -m venv venv
    
    #activate the virtual environment
    .\venv\Scripts\activate.bat
    ```
2. Install required libraries.
    ```sh
    pip install -r requirement.txt
    ```
3. Follow the structure above to configure the dataset.
4. Run the web.
    ```sh
     streamlit run main.py
    ```
5. Configure your username and password on `credentials.json` to login.
## Try your own dataset
1. Create your dataset.
2. Follow the structure to configure the dataset.
3. Move into `Extract Features` tab on the website.
4. Select the `Dataset path`.
5. Select  the `Feature Descriptor`.

## Try search system
1. Move into `Search System`.
2. Select either full image or crop image method.
3. Upload your image,
4. Configure the `dataset path`
5. Select the `feature descriptor` 
6. Choose the number of image results.

## Try Camera system
1. Move into `Camera System`.
2. Select either full image or crop image method.
3. Capture your picture.
4. Configure the `dataset path`
5. Select the `feature descriptor` 
6. Choose the number of image results.

## License

MIT


