#!/bin/bash
i=0
while IFS= read -r line ; do 
((i++))
(./main_parallel.py $line &> ./logfiles/log.${i} & ) ; 
done < "/users/nrietze/Desktop/Verknüpfung mit MA/data/glogem_centerpoints/parallel.txt"

#while IFS= read -r RGI_ID NAME X Y GLAMOS_ID ; do 
#(./parallel_test.py $NAME $X $Y $GLAMOS_ID &> ./logfiles/log.${NAME} & ) ; 
#done < "/users/nrietze/Desktop/Verknüpfung mit MA/data/RGI/parallel.txt"

#for i in {1..4} ; do 
#( ./parallel_test.py /users/nrietze/Desktop/Verknüpfung mit MA/data/RGI/#parallel.txt${i} & ) ; 
