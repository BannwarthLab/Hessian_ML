for file in F2_*/
do
	cd $file
	xtb coord.xyz --hess --gfn2
	~/git/xtb/xtb_ml/build/xtb coord.xyz --ml_feature
	cd ..
done
