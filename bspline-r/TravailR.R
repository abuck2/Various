Basis<-function(t,d,noeuds){
  
          #Bi,0(t)
    k<-c(1:(length(noeuds)-1))
    l<-c(1:d)
    B<-matrix(ncol = d+1, nrow = length(k), data = 0) #ncol=l+1 puisque l'on a le degré 0
    b<-0
  
    for(i in k){
      if(t>=noeuds[i]&t<=noeuds[i+1])
        b[i]<-1
      else
        b[i]<-0
    }
    print(b)
    #Bi,d
   
    B[,1]<-b
    for(j in l) {           # chaque degré
      m<-k-j                #sinon on dépasse la longueur de noeuds
      for(i in m){          #chaque noeud
        
        
        B[j+1, i]=((t-noeuds[i])/(noeuds[i+j])*B[i,j])+((noeuds[i+j+1]-t)/(noeuds[i+t+1]-noeuds[i+1])*B[i+1,j])
      }
    }
    print(B)
    
     
    
}
