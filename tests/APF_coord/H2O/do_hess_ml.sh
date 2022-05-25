for file in H2O_*/
do
	cd $file
	cd apf_coord/
	for files in atoms_*/
	do
		cd $files
		xtb coord.xyz --hess --gfn2
		~/git/xtb/xtb_ml/build/xtb coord.xyz --ml_feature
		cd ..
	done
	cd ..
	cd ..
done
