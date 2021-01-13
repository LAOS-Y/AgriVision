# AgriVis-Seg

## Data Preparation
1. Put the compressed dataset file "Agriculture-Vision.tar.gz" in data/
2. ```
   cd data && tar -xvf Agriculture-Vision.tar.gz
   ```
3. Generate odgt files in data/ (if you want to use the provided odgts, skip this step)
   ```
   python gen_odgt.py -r data -d Agriculture-Vision/train -o data/agri-trn.odgt
   python gen_odgt.py -r data -d Agriculture-Vision/val -o data/agri-val.odgt
   python gen_odgt.py -r data -d Agriculture-Vision/test -o data/agri-test.odgt -t
   head -n 100 data/agri-trn.odgt > data/agri-debug.odgt
   ```
4. Files in data should look like
    ```
    data
    |-- Agriculture-Vision
    |   |-- Agriculture-Vision\ Workshop\ Terms\ and\ Conditions.pdf
    |   |-- test
    |   |-- train
    |   `-- val
    |-- agri-debug.odgt
    |-- agri-test.odgt
    |-- agri-trn.odgt
    `-- agri-val.odgt
    ```

## Train
```
./train.sh 0,1 29500 config/agri-resnet101dilated-ibn@a-deeplab-low_feat@3-bce+dice+lovasz-aug-warmup@2000.yaml
```

## Test
```
python test.py --cfg config/agri-test.yaml
```

## Bibtex

```
@InProceedings{Yang_2020_CVPR_Workshops,
author = {Yang, Siwei and Yu, Shaozuo and Zhao, Bingchen and Wang, Yin},
title = {Reducing the Feature Divergence of RGB and Near-Infrared Images Using Switchable Normalization},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
}
```

## Contribution 

- Bingchen Zhao proposed the idea of measuring feature divergence between modalities which motivated this research and designed some of the experiments.
- Siwei Yang wrote most of the code and ran part of the experiments to validate the idea.
- Shaozuo Yu shared part of coding and assisted Siwei Yang with some experiments.
- Yin Wang supervises this reseach and provide the resources used by this research.
