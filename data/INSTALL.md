# Data Installation

## 1. Download datasets

Download the required datasets and ground truth file:

- Train dataset: https://challengedata.ens.fr/media/public/train-images.zip  
- Test dataset: https://challengedata.ens.fr/media/public/test-images.zip  
- Ground truth (CSV): https://challengedata.ens.fr/media/public/label_Hnl61pT.csv  

You can download them using `wget`:

```bash
wget https://challengedata.ens.fr/media/public/train-images.zip
wget https://challengedata.ens.fr/media/public/test-images.zip
wget https://challengedata.ens.fr/media/public/label_Hnl61pT.csv -O y.train.csv
```
## 2. Organize directory structure

We recommend the following project structure:

```
data/
├── train/
├── test/
└── y_train.csv
```

Then,
```
mkdir -p data/train data/test
```

## 3. Unzip and rename
unzip train-images.zip -d data/train
unzip test-images.zip -d data/test
mv label_Hnl61pT.csv data/y_train.csv

Finally, you can remove zip files 
```
rm train-images.zip test-images.zip
```

