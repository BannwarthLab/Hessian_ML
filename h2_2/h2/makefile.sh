length=('0.25' '0.3' '0.35' '0.37' '0.4' '0.45' '0.5') 
for file in "${length[@]}"
do	
	mkdir "H2_$file"
	cd "H2_$file"
	> coord.xyz
	echo "2" > coord.xyz
	echo "Energy =" >> coord.xyz
	echo "H     0.0000000    0.0000000 -$file" >> coord.xyz
	echo "H     0.0000000    0.0000000 $file " >> coord.xyz
	cd ..
done
