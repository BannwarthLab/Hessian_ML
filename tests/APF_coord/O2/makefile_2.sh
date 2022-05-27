length=('0.45' '0.5' '0.55' '0.6' '0.65' '0.7' '0.75' '0.8' '0.85' '0.9' '0.95' '1.0' '1.05' '1.1' '1.15' '1.2' '1.25')
for file in "${length[@]}"
do	
	mkdir "O2_$file"
	cd "O2_$file"
	mkdir "init_coord"
	cd "init_coord"
	> coord.xyz
	echo "2" > coord.xyz
	echo "Energy =" >> coord.xyz
	echo "O     -$file    0.0000000    0.0000000" >> coord.xyz
	echo "O     $file    0.0000000    0.0000000" >> coord.xyz
	cd ..
	cd ..
done
