/*A1*/
%LET lieu = C:\Users\abuckens\Documents\donnees_travailSAS\donnees_travailSAS;
libname projet "&lieu";

title "Statistiques descriptives sur les r?sultats de l'examen";

/*A2*/
proc sort data=projet.etudiants;
    by Noma;
run;

proc sort data=projet.notes;
    by Noma;
run;

data projet.partieA;
    merge projet.etudiants projet.notes;
    by Noma;
run;

/*A3*/
proc sort data=projet.partieA;
    by Annee_etude;
run;

proc means data=projet.partieA n mean median min max var maxdec=2;
    
    var grade;
    by Annee_etude;

run;

/*A4*/
proc freq data=projet.partieA;
    tables annee_inscrip*Annee_etude /nopercent norow nocol ;
run;

/*A5-6*/
/*data projet.notesBIS;
     set projet.details;
    total_mauvaises_rep = ?;

run;*/







/*A7*/
proc format;
 value grade 0-1 = 'echec'
             10-12='reussite'
            12-14='satisfaction'
            14-16='distonction'
            16-18='grande-distinction'
            18-20='la plus grande distinction';

run;
/*proc print data=projet.partieA;
run;*/

/*A8*/
title "%sysfunc(date(),worddate.)";
ods pdf ;
    title2 'reussite';
    proc print data=projet.partieA noobs;
        where Grade >= 12;
        var firstname lastname;
    run;
    title2 'echec';
    proc print data=projet.partieA noobs;
        where Grade < 12;
        var firstname lastname;
    run;
    
ods pdf close;



/*PartieB*/
title;

/*B1*/
data projet.etudiants2;
    infile " &lieu\etudiants.txt" firstobs=2 dsd missover;
    input
        @1 Noma 8.
        @10 Prenom $20.
        @30 Nom $40.;
run;

data projet.details2;
    infile "&lieu\details.txt" dlm=',' dsd missover;
    input
        Noma
        Version:1.
        Question_1 $
        Question_2 $
        Question_3 $
        Question_4 $
        Question_5 $
        Question_6 $
        Question_7 $
        Question_8 $
        Question_9 $
        Question_10 $
        Question_11 $
        Question_12 $
        Question_13 $
        Question_14 $
        Question_15 $
        Question_16 $
        Question_17 $
        Question_18 $
        Question_19 $
        Question_20 $;
run;

data projet.notes2;
    infile "&lieu\notes.txt" firstobs=2 delimiter='09'x;
    input
        Noma
        Annee_etude $
        Open
        Score:commax4.2
        Grade:commax4.2;
run;
/*B2*/
/*pas utilis? tranwrd, mais ca marche? moins bien peut-?tre*/
data projet.etudiants2;
    set projet.etudiants2;
    length user_email $ 70;
    length mail $ 13;
    mail = "uclouvain.be";
    Nom=compress(Nom, " ");
    user_email=catx(".",Prenom, Nom);
    /*user_email=cat(user_email, '@uclouvain.be');*/
    user_email=catx("@", user_email, mail);
    drop mail;
run;


/*B3*/
data projet.notes2;
    set projet.notes2;
    length annee_inscrip 4;
    annee_inscrip=cats('20', substr(Noma, 9, 2));
run;


