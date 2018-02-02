load data2;
figure
plot(X,Yb,'b.')
hold on
grid on
NbreblocksApp=4;
[Xapp,Yapp]=GetSamples(NbreblocksApp,X,Yb);
plot(Xapp,Yapp,'ro')
NbreblocksTest=3;
[Xtest,Ytest]=GetSamples(NbreblocksTest,X,Yb);
plot(Xtest,Ytest,'go')
legend(': donnees de depart', 'donnees apprentissage',':donnees test')
disp('SAUVEGARDE!!')
save base1 X Y Yb Xapp Yapp Xtest Ytest