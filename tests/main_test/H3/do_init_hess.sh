for file in H3_*/
do 
	cd $file
	cd init_coord
		xtb coord.xyz --chrg 0 --etemp 300 --hess --gfn2
                ~/git/xtb/xtb_ml/build/xtb coord.xyz --chrg 0 --etemp 300 --ml_feature
	cd ..
	cd ..
done
