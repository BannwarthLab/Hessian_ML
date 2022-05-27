length=('0.25' '0.3' '0.35' '0.37' '0.4' '0.45' '0.5' '0.55' '0.6' '0.65' '0.7') 
for file in "${length[@]}"
do	
	mkdir "H2_$file"
	cd "H2_$file"
	mkdir "init_coord"
	cd "init_coord"
	> coord.xyz
	echo "2" > coord.xyz
	echo "Energy =" >> coord.xyz
	echo "H     -$file    0.0000000    0.0000000" >> coord.xyz
	echo "H     $file    0.0000000    0.0000000" >> coord.xyz
	cd ..
	cd ..
done
