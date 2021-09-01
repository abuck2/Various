
#Setup
setwd("/run/media/alexis/6917a5e2-e4da-439f-ada9-c2d93a4db183/alexis/stats/sondage/")
rm(list=ls())
load("/run/media/alexis/6917a5e2-e4da-439f-ada9-c2d93a4db183/alexis/stats/sondage/FacebookPopu.Rda")
load("/run/media/alexis/6917a5e2-e4da-439f-ada9-c2d93a4db183/alexis/stats/sondage/OfficialPopu.Rda")
library(sampling)
library(samplingbook)
library(survey)
library(DBI)
library(dplyr)
dim(FacebookPopu)[1]
set.seed(5)

#Official Cluster?
View(OfficialPopu)
Probasc<-OfficialPopu$NEleve/sum(OfficialPopu$NEleve)
S<-cluster(data = OfficialPopu, clustername = c("NumEcoleUnique"), size = 18, method = "srswor", pik=("Probasc"))
Echant6 <- getdata(OfficialPopu, S)
table(Echant6$Ville, Echant6$Pays)
View(Echant6)
dim(Echant6)
save(Echant6, file = "Echantillon_official.Rda")
write.table(Echant6$NumEcoleUnique, "AlexisBuckens_Official.txt", sep="\t", row.names = F, quote = F, col.names = F) 
test<-read.table("Echantillon_officiel.txt", sep="\t")


#Facebook PESR
S<-srswor(29000, dim(FacebookPopu)[1])
Echantillon<-FacebookPopu[S!=0,]
View(Echantillon)
dim(Echantillon)
save(Echantillon, file = "Echantillon_facebook.Rda")
write.table(Echantillon$NumEleveUnique, "AlexisBuckens_Facebook.txt", sep="\t",  row.names = F, quote = F, col.names = F) 
length(Echantillon$NumEleveUnique)

## Seconde partie
load("/run/media/alexis/6917a5e2-e4da-439f-ada9-c2d93a4db183/alexis/stats/sondage/AlexisBuckens_FacebookSample.Rda")
load("/run/media/alexis/6917a5e2-e4da-439f-ada9-c2d93a4db183/alexis/stats/sondage/AlexisBuckens_OfficialSample.Rda")
### Facebook
## Facebook: Observation et structure
str(FacebookSample)
FacebookSample$Sexe<-as.factor(FacebookSample$Sexe)
FacebookSample$Fume<-as.factor(FacebookSample$Fume)
FacebookSample$Pays<-as.factor(FacebookSample$Pays)
FacebookSample$Ville<-as.factor(FacebookSample$Ville)
FacebookSample$Annee<-as.factor(FacebookSample$Annee)
str(FacebookSample)
View(FacebookSample)
##design
FacebookSample$Prob<-dim(FacebookSample)[1]/dim(FacebookPopu)[1]
Data1 <- svydesign( ids=~1 , fpc=~Prob , data=FacebookSample  )
m<-mean(FacebookSample$Argent)
svymean(~Argent, design=Data1, deff=T)
##Calcul de SE
f<-dim(FacebookSample)[1]/dim(FacebookPopu)[1]
sc<-sd(FacebookSample$Argent)^2
n<-dim(FacebookSample)[1]
var<-(1-f)*(sc/n)
se<-sqrt(var)
## Intervalle de confiance et incertitude relative
up<-m+(1.96*se)
lo<-m-(1.96*se)
1.96*se/m



### Official
#officialSample Observation et structure
str(OfficialSample)
OfficialSample$Pays<-as.factor(OfficialSample$Pays)
OfficialSample$Ville<-as.factor(OfficialSample$Ville)
OfficialSample$Ecole<-as.factor(OfficialSample$Ecole)
OfficialSample$Sexe<-as.factor(OfficialSample$Sexe)
OfficialSample$Fume<-as.factor(OfficialSample$Fume)
OfficialSample$Cinema<-as.integer(OfficialSample$Cinema)
str(OfficialSample)
View(OfficialSample)
#officialSample Design survey
tail(OfficialPopu)
OfficialSample$M<-dim(OfficialPopu)[1]
#Prob<-(OfficialSample$NEleve/(sum(OfficialPopu$NEleve)*sum(OfficialSample$Prob)))
#N<-sum(OfficialPopu$NEleve)
Datao<-svydesign(ids=~NumEcoleUnique, data=OfficialSample, fpc = ~M)
Datao
svymean(~Argent, design=Datao, deff=T)
OfficialSample$propor<-(sum(OfficialPopu$NEleve))
Datao<-svydesign(ids=~NumEcoleUnique, data=OfficialSample, fpc = ~propor)
Datao
svymean(~Argent, design=Datao, deff=T)
## Calcul de la variance intra et inter --- NON
Off<-list(Off)
Off1<-filter(OfficialSample, NumEcoleUnique==17)$Argent
Off2<-filter(OfficialSample, NumEcoleUnique==25)$Argent
Off3<-filter(OfficialSample, NumEcoleUnique==27)$Argent
Off4<-filter(OfficialSample, NumEcoleUnique==30)$Argent
Off5<-filter(OfficialSample, NumEcoleUnique==36)$Argent
Off6<-filter(OfficialSample, NumEcoleUnique==46)$Argent
Off7<-filter(OfficialSample, NumEcoleUnique==54)$Argent
Off8<-filter(OfficialSample, NumEcoleUnique==57)$Argent
Off9<-filter(OfficialSample, NumEcoleUnique==65)$Argent
Off10<-filter(OfficialSample, NumEcoleUnique==68)$Argent
Off11<-filter(OfficialSample, NumEcoleUnique==70)$Argent
Off12<-filter(OfficialSample, NumEcoleUnique==89)$Argent
Off13<-filter(OfficialSample, NumEcoleUnique==99)$Argent
Off14<-filter(OfficialSample, NumEcoleUnique==110)$Argent
Off15<-filter(OfficialSample, NumEcoleUnique==111)$Argent
Off16<-filter(OfficialSample, NumEcoleUnique==112)$Argent
Off17<-filter(OfficialSample, NumEcoleUnique==145)$Argent
Off18<-filter(OfficialSample, NumEcoleUnique==156)$Argent
Off<-list(Off1, Off2, Off3, Off4, Off5, Off6, Off7, Off8, Off9, Off10, Off11, Off12, Off13, Off14, Off15, Off16, Off17, Off18)
rm(Off1, Off2, Off3, Off4, Off5, Off6, Off7, Off8, Off9, Off10, Off11, Off12, Off13, Off14, Off15, Off16, Off17, Off18)

tot<-rep(0,18)
for(i in 1:18){
  tot[i]<-var(Off[[i]])*length(Off[[i]])
  }
total<-sum(tot)/dim(OfficialSample)[1]

moy<-mean(OfficialSample$Argent)
moyt<-rep(0,18)
for(i in 1:18){
  moyt[i]<-(mean(Off[[i]])-moy)^2*(length(Off[[i]])/dim(OfficialSample)[1])
}
inter<-sum(moyt)
inter/(total)
#Si on avait GRTE
moyt<-rep(0,18)
for(i in 1:18){
  moyt[i]<-(mean(Off[[i]])-moy)^2
}
inter<-(sum(moyt)/dim(OfficialPopu)[1])
inter/sd(OfficialSample$Argent)
#GRTE? Non
ng<-rep(0,18)
for(i in 1:18){
  ng[i]<-length(Off[[i]])
}
min(ng)
max(ng)
## Erreur de l'estimateur, recalculé
sc<-sd(OfficialSample$Argent)^2/17
fac<-(dim(OfficialPopu)[1]^2)/(dim(OfficialSample)[1]^2)
facb<-(1-18/dim(OfficialPopu)[1])
var<-sc*fac*facb
se<-sqrt(var)
#intervalle de confiance
up<-m+(1.96*se)
lo<-m-(1.96*se)
1.96*se/m
#Comparaison avec swsor
f<-dim(OfficialSample)[1]/sum(OfficialPopu$NEleve)
sc<-sd(OfficialSample$Argent)^2
n<-dim(OfficialSample)[1]
var<-(1-f)*(sc/n)
sqrt(var)

#Redressement : Facebook
#correction des infos : Fumeurs: 58.46%, Cinema:3.11séances, écart-type de population=10.76
sd(FacebookSample$Argent)
sum(FacebookSample$Fume==1)/(sum(FacebookSample$Fume==1)+sum(FacebookSample$Fume==0))
mean(FacebookSample$Cinema)
propo<-sum(FacebookSample$Fume==1)/(sum(FacebookSample$Fume==1)+sum(FacebookSample$Fume==0))
m2<-mean(FacebookSample$Argent)*propo/0.5846
m2
mean(filter(FacebookSample, Fume==1)$Argent)
mean(filter(FacebookSample, Fume==0)$Argent)
cor(FacebookSample$Argent, FacebookSample$Fume, method = 'spearman')

#Poststrata - Pays
eff<-c(0,0,0)
eff[1]<-dim(filter(FacebookPopu, Pays==1))[1]
eff[2]<-dim(filter(FacebookPopu, Pays==2))[1]
eff[3]<-dim(filter(FacebookPopu, Pays==3))[1]
eff
Effectifs <- data.frame(Pays=1:3, Freq=eff)
design2<-postStratify(design= Data1, strata= ~Pays, population = Effectifs)
#poststrata-Age
eff<-rep(0,7)
for(i in 1:7){
  eff[i]<-dim(filter(FacebookPopu, Age==(i+12)))[1]
}
eff
Effectifs <- data.frame(Age=13:19, Freq=eff)
design3<-postStratify(design= Data1, strata= ~Age, population = Effectifs)
#poststrata-Sexe
eff<-c(0,0)
eff[1]<-dim(filter(FacebookPopu, Sexe==0))[1]
eff[2]<-dim(filter(FacebookPopu, Sexe==1))[1]
eff
Effectifs <- data.frame(Sexe=c(0,1), Freq=eff)
design4<-postStratify(design= Data1, strata= ~Sexe, population = Effectifs)
#comparaison
svymean(~Argent, design4)
svymean(~Argent, design3)
svymean(~Argent, design2)
svymean(~Argent, design=Data1)
se<-0.0611
m<-22.161
## Intervalle de confiance et incertitude relative
up<-m+(1.96*se)
lo<-m-(1.96*se)
1.96*se/m

#Redressement : Official
#correction des infos : Fumeurs: 58.46%, Cinema:3.11séances, écart-type de population=10.76
sd(OfficialSample$Argent)
sum(OfficialSample$Fume==1)/(sum(OfficialSample$Fume==1)+sum(OfficialSample$Fume==0))
svymean(~Cinema, design=Datao, deff=T)
#Poststrata - Pays
eff<-c(0,0,0)
eff[1]<-dim(filter(FacebookPopu, Pays==1))[1]
eff[2]<-dim(filter(FacebookPopu, Pays==2))[1]
eff[3]<-dim(filter(FacebookPopu, Pays==3))[1]
eff
Effectifs <- data.frame(Pays=1:3, Freq=eff)
designo2<-postStratify(design= Datao, strata= ~Pays, population = Effectifs)
svymean(~Argent, designo2)
svymean(~Argent, Datao, Deff=T)
#reech
designo2.rep <- as.svrepdesign(designo2)
svymean(~Argent, designo2.rep)

#comparaison distributions
hist(OfficialSample$Argent, freq = F, xlab = "Argent de poche : Official", main = "Distribution de l'argent de poche : Official")
hist(FacebookSample$Argent, freq = F, xlab = "Argent de poche : Facebook", main = "Distribution de l'argent de poche : Facebook")
p1 <- hist(OfficialSample$Argent, freq = F)
p2 <- hist(FacebookSample$Argent, freq = F)
plot( p1, col='blue', xlim=c(0,60), freq = F, main="Official", xlab="Argent")  # first histogram
plot( p2, col='red', add=T, freq=F)  # second
plot( p2, col='blue', xlim=c(0,60), freq = F, main="Facebook", xlab="Argent")  # first histogram
plot( p1, col='red', add=T, freq=F)  # second
