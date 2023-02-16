#SBATCH --output='output.%j.out'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=3:00:00

#SBATCH --mem-per-cpu=3G
#SBATCH --job-name=xTB
#SBATCH --output=output.%J.txt
#SBATCH --array=1-100
export MKL_NUM_THREADS=40

module load xtb/ml-hessian
which xtb

clean_up () {
  echo "clean_up is used at $(date)."
  echo "Files are copied to cwd. '$tdir' is cleard."
  rm $tdir/*tmp*
  mkdir $SLURM_SUBMIT_DIR/temp
  cp ./* $SLURM_SUBMIT_DIR/temp && rm -rf $tdir
}

ZERO_NUMBER=$(printf "7%03d" $SLURM_ARRAY_TASK_ID)

dir=${ZERO_NUMBER}

#dir=$SLURM_ARRAY_TASK_ID

cd $dir

dir=$(pwd)

echo $dir

trap 'clean_up_function' USR1 SIGTERM SIGKILL

tdir=$(mktemp -d $TMPDIR/XXXXXXXXXXXXXXXXXXX)

echo 'temp dir for computations:' $tdir

echo $SLURM_SUBMIT_DIR
for f in $(ls *) ; do
        cp $f $tdir
done

cd $tdir

xtb xtbopt.xyz --gfn2 --hess

shopt -s extglob
cp -r ./!(*tmp*) $dir  && rm -rf $tdir

cd $SLURM_SUBMIT_DIR/
