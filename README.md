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
@inproceedings{yang2020CVPRW,
  title={Reducing the feature divergence of RGB and near-infrared images using Switchable Normalization},
  author={ {Siwei Yang, Shaozuo Yu, Bingchen Zhao} and Yin Wang},
  booktitle={Proceedings of IEEE CVPR(2020) Workshop on Agriculture-Vision},
  year={2020}
}
```
