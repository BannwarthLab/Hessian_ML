for file in H3_*/
do 
	cd $file
		python3.8 ~/git/hessian_ml/rotation_coord.py
	cd ..
done
