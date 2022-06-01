for file in NH3_*/
do 
	cd $file
	cd init_coord
		rm charges chempot_ext.csv g98.out hessian ml_feature.csv vibspectrum wbo xtbhess.xyz xtbrestart xtbtopo.mol xyz_dipm.csv
	cd ..
	cd ..
done
