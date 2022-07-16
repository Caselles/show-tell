Typical command is: mpirun -n 24 python train_learner_show-tell.py --n-epochs 250 --save-dir output_learner/ --teacher-action-mode naive --learner-from-demos True --sqil True --teacher-language-mode colors_preference_R1 --learner-language-mode pragmatic --compute-statistically-significant-results True --predictability True --reachability True --no-biased-init 1 --n-cycles 2

then: python plots_v3.py
