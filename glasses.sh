# CHANGE THIS FOR YOU
declare -r path_FirstAid=/Users/yidarvin/Desktop/FirstAid/train_CNNclassification.py

declare -r path_train=$PWD/h5_data/training
declare -r path_val=$PWD/h5_data/testing

declare -r name=glasses

declare -r path_model=$PWD/model_state/$name.ckpt
declare -r path_log=$PWD/logs/$name.txt
declare -r path_vis=$PWD/graphs

python $path_FirstAid --pTrain $path_train --pVal $path_val --name $name --pModel $path_model --pLog $path_log --pVis $path_vis --nGPU 0 --bs 8 --ep 50 --nClass 2 --lr 0.001 --do 0.5
