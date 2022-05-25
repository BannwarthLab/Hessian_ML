for file in HF_*/
do 
	cd $file
	cd init_coord
		xtb coord.xyz --hess --gfn2
                ~/git/xtb/xtb_ml/build/xtb coord.xyz --ml_feature
	cd ..
	cd ..
done
