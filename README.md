# LatentGoal
Code accompanying the IEEE WACV 2022 paper ["Action anticipation using latent goal learning"](https://openaccess.thecvf.com/content/WACV2022/papers/Roy_Action_Anticipation_Using_Latent_Goal_Learning_WACV_2022_paper.pdf)

# Features
Download RGB features from [RULSTM](https://github.com/fpv-iplab/rulstm) project, specifically this script

https://github.com/fpv-iplab/rulstm/blob/master/RULSTM/scripts/download_data_ek55.sh

and these lines

```
mkdir -p data/ek55/rgb
curl https://iplab.dmi.unict.it/sharing/rulstm/features/rgb/data.mdb -o data/ek55/rgb/data.mdb
```

The data is now in <pwd>/data/ek55/rgb. Next, fetch the *training.csv* and *validation.csv* from the RULSTM project [ek55 directory](https://github.com/fpv-iplab/rulstm/tree/master/RULSTM/data/ek55)

You are all set to run the code. Please cite this work if you use this code

```
@inproceedings{
wacv22,
title={Action anticipation using latent goal learning},
author={Debaditya Roy and Basura Fernando},
booktitle={WACV},
year={2022},
}
```
# Acknowledgment

This research/project is supported in part by the National Research Foundation, Singapore under its AI Singapore Program (AISG Award No: AISG2-RP-2020-016) and the National Research Foundation Singapore under its AI Singapore Program (Award Number: AISG-RP-2019-010).

  
In case of issues, please write to roy_debaditya [at] ihpc [dot] a-star [dot] edu [dot] sg
