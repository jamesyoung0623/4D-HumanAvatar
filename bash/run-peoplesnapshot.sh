experiment="fitting"
SEQUENCES=("male-3-casual")
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="peoplesnapshot/$SEQUENCE"
    # bash scripts/custom/process-sequence.sh ./data/PeopleSnapshot/$SEQUENCE neutral
    python fit.py --config-name SNARF_NGP_fitting dataset=$dataset experiment=$experiment deformer=smpl train.max_epochs=200
    # python train.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment train.max_epochs=200
    # python eval.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment
done