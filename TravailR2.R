Basis<-function(t,d,noeuds){
  
  #Bi,0(t)
  k<-c(1:(length(noeuds)-1))
  B<-matrix(ncol = d+1, nrow = length(k), data = 0) #ncol=l+1 puisque l'on a le degré 0
  b<-0
  
  for(i in k){
    if(t>=noeuds[i]&t<=noeuds[i+1])
      b[i]<-1
    else
      b[i]<-0
  }
  B[1,]<-b
  #Bi,2
  
  # for(i in k-1)
  #   b[i]=((t-noeuds[i])/(noeuds[i+1]-noeuds[i]))*b[i]+((noeuds[i+1+1]-t)/(noeuds[i+1+1]-noeuds[i+1]))*b[i+1]
  
  #Bi,d
  for(j in 2:d)
    for(i in 1:(length(noeuds)-j))
      B[j,i]=((t-noeuds[i])/(noeuds[i+j]-noeuds[i]))*B[i,j-1]+((noeuds[i+j+1]-t)/(noeuds[i+j+1]-noeuds[i+1]))*B[i+1, j-1]
  
  
}
