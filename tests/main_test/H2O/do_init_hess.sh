for file in H2O_*/
do 
	cd $file
	cd init_coord
		xtb coord.xyz --hess --gfn2
                ~/git/xtb/xtb_ml/build/xtb coord.xyz --ml_feature
	cd ..
	cd ..
done
