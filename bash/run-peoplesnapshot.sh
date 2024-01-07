experiment="test_refine_pose"
# SEQUENCES=("female-3-casual" "female-4-casual" "male-3-casual" "male-4-casual")
# SEQUENCES=("female-3-casual")
# SEQUENCES=("female-4-casual")
SEQUENCES=("male-3-casual")
# SEQUENCES=("male-4-casual")
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="peoplesnapshot/$SEQUENCE"
    # bash scripts/custom/process-sequence.sh ./data/PeopleSnapshot/$SEQUENCE neutral
    # python fit.py --config-name SNARF_NGP_fitting dataset=$dataset experiment=$experiment deformer=smpl train.max_epochs=200
    # python train.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment train.max_epochs=1000
    python eval.py --config-name SNARF_NGP_refine dataset=$dataset experiment=$experiment train.max_epochs=6300
    # python eval.py --config-name SNARF_NGP_refine dataset=$dataset experiment=$experiment train.max_epochs=1000
done