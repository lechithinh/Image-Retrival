<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>
<h1 align="center"><b>TÍNH TOÁN ĐA PHƯƠNG TIỆN</b></h>

## THÀNH VIÊN NHÓM

| STT |   MSSV   |       Họ và Tên |                     Github                                 |            Email                        |
| --- | :------: | --------------: |  --------------------------------------------------------: | ---------------------------------------:|
| 1   | 21522634 |  Lê Chí Thịnh   |  [lechithinh](https://github.com/lechithinh)               | 21522634@gm.uit.edu.vn                  |
| 2   | 21522621 | Huỳnh Công Thiện|  [HuynhThien1](https://github.com/HuynhThien1)             | 21522621@gm.uit.edu.vn                  |
| 3   | 21522706 | Nguyễn Minh Trí |  [MinhTri17](https://github.com/MinhTri17)                 | 21522706@gm.uit.edu.vn                  |


## GIỚI THIỆU MÔN HỌC

-   **Tên môn học:** Tính toán đa phương tiện
-   **Mã môn học:** CS232
-   **Mã lớp:** CS232.N21.KHCL
-   **Năm học:** HK2 (2022 - 2023)
-   **Giảng viên**: TS.Đỗ Văn Tiến


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

-  [Python](https://docs.python.org/3/)
-  [Streamlit](https://docs.streamlit.io/)
-  [Pytorch](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html#torchvision.models.VGG16_Weights)

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

The dataset for this project is [Fashion Dataset](https://drive.google.com/file/d/1k6467Iv7us87foba77YGtrY36fuLz9G-/view?usp=drive_link)
+ There are 24 class
+ Each class contains 30 images
+ The total images is 720



Be sure follow `The same structure` .

## Extracted Features
+ These are features that we extracted based on the dataset provided above  [ Feature](https://drive.google.com/drive/folders/1syWNHdNG1BcOM_3YlHOLbxFdposJ3U6E?usp=drive_link)
+ You you download this one and run your code immediately or you can use the dataset above and extract these features on your own.

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
