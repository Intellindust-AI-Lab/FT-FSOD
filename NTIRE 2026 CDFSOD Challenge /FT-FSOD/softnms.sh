for dataset in dataset1 dataset2 dataset3; do
  for shot in 1shot 5shot 10shot; do
  python ./softnms.py \
    --pred1 ./exp_cdfsodchallenge26_results_trainval1/tta/${dataset}/${shot}/swinL_${dataset}_${shot}/${dataset}_${shot}_coco_preds.json \
    --pred2 ./exp_cdfsodchallenge26_results_trainval2/tta/${dataset}/${shot}/swinL_${dataset}_${shot}/${dataset}_${shot}_coco_preds.json \
    --output ./softnms_results/${dataset}_${shot}.json 
  done
done
