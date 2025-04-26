for baseline_model in 'ToMnet' 'BC'
do
    for seed in {0..5}
    do
        for lr in 1e-2 1e-3 1e-4 1e-5
        do
            sbatch train_mtom.slurm $baseline_model $seed $lr
        done
    done
done