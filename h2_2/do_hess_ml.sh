for file in H2_*/
do
	cd $file
	xtb coord.xyz --hess
	~/git/xtb/xtb_ml/build/xtb coord.xyz --ml_feature
	cd ..
done
