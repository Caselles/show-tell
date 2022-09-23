#!/bin/sh
#SBATCH --account=kcr@gpu
#SBATCH --job-name=isl_experiment_debug
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=isl_experiment_%j.out
#SBATCH --error=isl_experiment_%j.out
#SBATCH --time=01:59:59
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --array=0-0

module load pytorch-gpu/py3/1.4.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib
export LIBRARY_PATH=$LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib
export CPATH=$CPATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/genisi01/uhx75if/.mujoco/mujoco200/bin
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_btl_openib_warn_default_gid_prefix=0
export OMPI_MCA_mpi_warn_on_fork=0

srun python -u -B train_learner.py --cuda --learner-from-demos True --teacher-mode pedagogical --pragmatic-learner True --sqil True  --compute-statistically-significant-results True --predictability True --reachability True 2>&1 