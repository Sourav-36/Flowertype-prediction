from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data     
Y=iris.target   
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=11) #splitting is done randomly, so random_state is taken to tune the randomness
acc_knn=0
acc_knn1=0
acc_knn2=0
acc_knn3=0

def KNN():
    global acc_knn
    global K
    from sklearn.neighbors import KNeighborsClassifier
    K=KNeighborsClassifier(n_neighbors=5)
    K.fit(X_train,Y_train)
    Y_pred=K.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc_knn=accuracy_score(Y_test,Y_pred)
    acc_knn=round(acc_knn*100,2)
    m.showinfo(title="KNN",message="You have selected KNN and got accuracy score of"+ str(acc_knn)+"%")

def LG():
    global acc_knn1
    from sklearn.linear_model import LogisticRegression
    LG=LogisticRegression(solver='liblinear',multi_class='auto')
    LG.fit(X_train,Y_train)
    Y_pred=LG.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc_knn1=accuracy_score(Y_test,Y_pred)
    acc_knn1=round(acc_knn1*100,2)
    m.showinfo(title="LG",message="You have selected LG and got accuracy score of"+ str(acc_knn1)+"%")

def DT():
    global acc_knn2
    from sklearn.tree import DecisionTreeClassifier
    DT=DecisionTreeClassifier()
    DT.fit(X_train,Y_train)
    Y_pred=DT.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc_knn2=accuracy_score(Y_test,Y_pred)
    acc_knn2=round(acc_knn2*100,2)
    m.showinfo(title="DT",message="You have selected DT and got accuracy score of"+ str(acc_knn2)+"%")

def NB():
    global acc_knn3
    from sklearn.naive_bayes import GaussianNB
    NB=GaussianNB()
    NB.fit(X_train,Y_train)
    Y_pred=NB.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc_knn3=accuracy_score(Y_test,Y_pred)
    acc_knn3=round(acc_knn3*100,2)
    m.showinfo(title="NB",message="You have selected NB and got accuracy score of"+ str(acc_knn3)+"%")

def compare():
    import matplotlib.pyplot as plt
    '''bottom=0
    # x-coordinate
    left=[1,2,3,4]

    #y-coordinate
    height=[acc_knn,acc_knn1,acc_knn2,acc_knn3]

    #label bar
    tick_label=['KNN','LR,','DT','NB']

    #plotting graph
    plt.bar(left,height,tick_label=tick_label,width=0.8,color=['red','green'])

    #labeling axis
    plt.xlabel=('MODEL')
    plt.ylabel=('ACCURACY')
    plt.title('FLOWER ML')
    plt.show()'''
    
    model=["KNN","LG","NB","DT"]
    accuracy=[acc_knn,acc_knn1,acc_knn2,acc_knn3]
    plt.bar(model,accuracy,color=["orange","blue","green","red"])
    plt.xlabel("MODELS")
    plt.ylabel("ACCURACY")
    plt.show()


def submit():
    s1=float(v1.get())
    s2=float(v2.get())
    s3=float(v3.get())
    s4=float(v4.get())
    result=K.predict([[s1,s2,s3,s4]])

    if result==0:
        flower="Setosa"
    elif result==1:
        flower="Versicolor"
    else:
        flower="Virginica"
    m.showinfo(title="IRIS FLOWER",message=flower)

def reset():
    v1.set("0")
    v2.set("0")
    v3.set("0")
    v4.set("0")



from tkinter import *
import tkinter.messagebox as m
w=Tk()
w.title("Machine Learning")
v1=StringVar()
v2=StringVar()
v3=StringVar()
v4=StringVar()
    

B1=Button(w,text="KNN",font=("arial",20,"bold"),command=KNN)
B2=Button(w,text="LG",font=("arial",20,"bold"),command=LG)
B3=Button(w,text="DT",font=("arial",20,"bold"),command=DT)
B4=Button(w,text="NB",font=("arial",20,"bold"),command=NB)
B5=Button(w,text="Compare",font=("arial",20,"bold"),command=compare)
B6=Button(w,text="Submit",font=("arial",20,"bold"),command=submit)
B7=Button(w,text="Reset",font=("arial",20,"bold"),command=reset)
L=Label(w,text="Entry for flower data",font=("arial",20,"bold"))
L1=Label(w,text="SL",font=("arial",20,"bold"))
L2=Label(w,text="SW",font=("arial",20,"bold"))
L3=Label(w,text="PL",font=("arial",20,"bold"))
L4=Label(w,text="PW",font=("arial",20,"bold"))
E1=Entry(w,font=("arial",20,"bold"),justify="left",textvariable=v1)
E2=Entry(w,font=("arial",20,"bold"),justify="left",textvariable=v2)
E3=Entry(w,font=("arial",20,"bold"),justify="left",textvariable=v3)
E4=Entry(w,font=("arial",20,"bold"),justify="left",textvariable=v4)

B1.grid(row=2,column=1)
B2.grid(row=3,column=1)
B3.grid(row=4,column=1)
B4.grid(row=5,column=1)
B5.grid(row=6,column=1)
B6.grid(row=6,column=2)
B7.grid(row=6,column=3)
L.grid(row=1,column=2)
L1.grid(row=2,column=2)
L2.grid(row=3,column=2)
L3.grid(row=4,column=2)
L4.grid(row=5,column=2)
E1.grid(row=2,column=3)
E2.grid(row=3,column=3)
E3.grid(row=4,column=3)
E4.grid(row=5,column=3)


w.mainloop()
