# Action Anticipation using Latent Goal Learning
Code accompanying the IEEE WACV 2022 paper ["Action anticipation using latent goal learning"](https://openaccess.thecvf.com/content/WACV2022/papers/Roy_Action_Anticipation_Using_Latent_Goal_Learning_WACV_2022_paper.pdf)

# EPIC-KITCHENS 55
* For training the verb anticipation network

  Download RGB features from [RULSTM](https://github.com/fpv-iplab/rulstm) project, specifically this script

  https://github.com/fpv-iplab/rulstm/blob/master/RULSTM/scripts/download_data_ek55.sh

  and these lines

  ```
  mkdir -p data/ek55/rgb
  curl https://iplab.dmi.unict.it/sharing/rulstm/features/rgb/data.mdb -o data/ek55/rgb/data.mdb
  ```

  The data is now in <pwd>/data/ek55/rgb. Next, fetch the *training.csv* and *validation.csv* from the RULSTM project [ek55 directory](https://github.com/fpv-iplab/rulstm/tree/master/RULSTM/data/ek55)

  * For training the noun anticipation network
   Download OBJ features from [RULSTM](https://github.com/fpv-iplab/rulstm) project, specifically this script

  https://github.com/fpv-iplab/rulstm/blob/master/RULSTM/scripts/download_data_ek55.sh

  and these lines

  ```
  mkdir -p data/ek55/rgb
  curl https://iplab.dmi.unict.it/sharing/rulstm/features/obj/data.mdb -o data/ek55/obj/data.mdb
  ```

  The data is now in <pwd>/data/ek55/obj. 
  
  

# Breakfast and 50 Salads

I3D features were obtained from this repo for both 50Salads and Breakfast.

https://github.com/yabufarha/ms-tcn

Download the [data](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) folder, which contains the features and the ground truth labels. (~30GB) (If you cannot download the data from the previous link, try to download it from [here](https://zenodo.org/record/3625992#.Xiv9jGhKhPY))

Then run [i3d_latent_goal_bf.py]
(https://github.com/debadityaroy/LatentGoal/blob/main/i3d_latent_goal_bf.py)

# Clarifications
* The CSV format is different in *training.csv* and *test_seen.csv*. For *training.csv*, the columns are - ```segment_id, video_id, start_frame, end_frame, verb, noun, action``` For *test_seen.csv*, the columns are - ```segment_id, video_id, start_frame, end_frame```
* We have actions instead of verb and nouns for Breakfast and 50Salads.

Please cite this work if you use this code

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
