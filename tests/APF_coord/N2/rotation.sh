for file in N2_*/
do 
	cd $file
		python3.8 ~/git/hessian_ml/rotation_coord.py
	cd ..
done
