import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
import re
from sklearn.metrics import confusion_matrix 
# from sklearn.cross_validation import train_test_split 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn import metrics 
import matplotlib.pyplot as plt
#importing wx files
import wx
#import the newly created GUI file
import gui
from sklearn import svm
import webbrowser
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,classification,roc_curve
from sklearn import preprocessing
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import log_loss
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,classification,roc_curve



#ROC Curve
def plot_roc_curve(fpr, tpr):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


#extract single feature
def extract_feature_usertest(url):
    
    
    #length of url
    l_url=len(url)
    if(l_url > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    
    return length_of_url,http_has,suspicious_char,prefix_suffix,dots,slash,phis_term,sub_domain,ip_contain




#extract single feature
def extract_feature_usertest_stack(url):
    
    
    #length of url
    l_url=len(url)
    if(l_url > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    res=[]

    res.append(length_of_url)
    res.append(http_has)
    res.append(suspicious_char)
    res.append(prefix_suffix)
    res.append(dots)
    res.append(slash)
    res.append(phis_term)
    res.append(sub_domain)
    res.append(ip_contain)
    
    return res



#extract testing feature
def extract_feature_test(url,output):
    
    
    #length of url
    l_url=len(url)
    if(l_url > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    #output
    yn = output
    
    return yn,length_of_url,http_has,suspicious_char,prefix_suffix,dots,slash,phis_term,sub_domain,ip_contain
#extract training feature
def extract_feature_train(url,output):
    
    
    #length of url
    l_url=len(url)
    if(l_url > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    #output
    yn = output

   
        
    return yn,length_of_url,http_has,suspicious_char,prefix_suffix,dots,slash,phis_term,sub_domain,ip_contain
#import train data
def importdata_train(): 
    balance_data = pd.read_csv('id3.csv',sep= ',', header = 1,usecols=range(1,11),encoding='utf-8') 
      
      
    # Printing the dataswet shape 
    print ("Dataset Lenght: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data 
#import test data
def importdata_test(): 
    balance_data = pd.read_csv('feature_test.csv',sep= ',', header = 1,usecols=range(1,11),encoding='utf-8') 
      
      
    # Printing the dataswet shape 
    print ("Dataset Lenght: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data 
#split data into train and test
def splitdataset(balance_data): 
  
    # Seperating the target variable 
    X = balance_data.values[:, 1:10]
    Y = balance_data.values[:, 0] 
  
    # Spliting the dataset into train and test 
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 2, min_samples_leaf = 10) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    #print("Predicted values:") 
    #print(y_pred) 
    return y_pred 
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred))

    return accuracy_score(y_test,y_pred)*100

#roc
def plot_roc_curve(fpr, tpr ):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()



#main funcation
def main():
    excel_file= 'training.xlsx'
    df=pd.DataFrame(pd.read_excel(excel_file))
    excel_file_test= 'test1.xlsx'
    df1=pd.DataFrame(pd.read_excel(excel_file_test))

    a=[]
    b=[]
    a1=[]
    b1=[]
    for url in df['url']:
        a.append(url)

    for output in df['phishing']:
        b.append(output)

    for url1 in df1['url']:
        a1.append(url1)

    for output in df1['result']:
        b1.append(output)

    c=[]
    d=[]
    for url1,output1 in zip(a,b):       
        url=url1
        output=output1
        c.append(extract_feature_train(url,output))

    for url1,output1 in zip(a1,b1):           
        url=url1
        output=output1
        d.append(extract_feature_test(url,output))



    df=pd.DataFrame(c,columns=['r','length_of_url','http_has','suspicious_char','prefix_suffix','dots','slash','phis_term','sub_domain','ip_contain'])

    df.to_csv('id3.csv', sep=',', encoding='utf-8')

    df_test=pd.DataFrame(d,columns=['r','length_of_url','http_has','suspicious_char','prefix_suffix','dots','slash','phis_term','sub_domain','ip_contain'])

    df_test.to_csv('feature_test.csv', sep=',', encoding='utf-8')  
    
    data_train=importdata_train()
    data_test=importdata_test()
    X, Y = splitdataset(data_train) 
    X1, Y1 = splitdataset(data_test)  

    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)
    
    model=RandomForestClassifier()
    model.fit(X,Y)
    
    gnb = DecisionTreeClassifier()
    gnb.fit(X, Y)
    
    


#RandomForest
    class Predict1(gui.MainFrame):

        def __init__(self,parent):
            gui.MainFrame.__init__(self,parent)
        
        def click(self,event): 
            try:
                url = self.text1.GetValue()
                e=np.array([extract_feature_usertest(url)])
                userpredict1 = model.predict(e.reshape(1,-1)) 
                if(userpredict1[0]=='no'):
                    self.text2.SetValue(str("Legitimate"))
                    print('Legitimate')
                    webbrowser.open(url)

                else:
                    self.text2.SetValue(str("Phising"))
                    print('Phising')
            except Exception:
                print ('error')
        def clearFunc(self,event):
            self.text.SetValue(str(''))

    app1 = wx.App(False)
    
    
    frame = Predict1(None)
    frame.Show(True)
    app1.MainLoop() 

#SVM
    class Predict2(gui.MainFrame):
        clf = svm.SVC(kernel='linear',probability=True)
        clf.fit(X, Y)
        def __init__(self,parent):
            gui.MainFrame.__init__(self,parent)

        def click(self,event):
            try:
                url2 = self.text1.GetValue()
                e2=np.array([extract_feature_usertest(url2)])
                userpredict2 = clf.predict(e2.reshape(1,-1))
                if(userpredict2[0]=='no'):
                    self.text2.SetValue(str("Legitimate"))
                    webbrowser.open(url2)
                    print('Legitimate')
                   

                else:
                    self.text2.SetValue(str("Phising")) 
                    print('Phising')
            except Exception:
                print ('error')
        def clearFunc(self,event):
            self.text.SetValue(str(''))
    
    app2 = wx.App(False)
    frame = Predict2(None)
    frame.Show(True)
    app2.MainLoop()



#DecisionTree
    class Predict3(gui.MainFrame):

        def __init__(self,parent):
            gui.MainFrame.__init__(self,parent)

        def click(self,event):
            try:
                url3 = self.text1.GetValue()
                e3=np.array([extract_feature_usertest(url3)])
                userpredict3 = gnb.predict(e3.reshape(1,-1))
                if(userpredict3[0]=='no'):
                    self.text2.SetValue(str("Legitimate"))
                    webbrowser.open(url3)
                    print('Legitimate')
                else:
                    self.text2.SetValue(str("Phising")) 
                    print('Phising')
            except Exception:
                print ('error')
        def clearFunc(self,event):
            self.text.SetValue(str(''))
    
    app3 = wx.App(False)
    frame = Predict3(None)
    frame.Show(True)
    app3.MainLoop()





    print("___________________________RandomForst__________________________________________") 
    model=RandomForestClassifier()
    model.fit(X,Y)
    y_pred1 = model.predict(X1)
    print("_____________Report___________________")
    acc1=cal_accuracy(Y1, y_pred1)
    # print("_____________user input ___________________")
    #confusion Matrix
    import matplotlib.pyplot as plt1
    matrix =confusion_matrix(Y1, y_pred1)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt1.xticks(tick_marks, class_names)
    plt1.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt1.tight_layout()
    plt1.title('Confusion matrix', y=1.1)
    plt1.ylabel('Actual label')
    plt1.xlabel('Predicted label')
    fig.canvas.set_window_title('RandomForest')
    plt.show()

    #ROC_AUC curve
    probs = model.predict_proba(X1) 
    probs = probs[:, 1]  
    auc = roc_auc_score(Y1, probs)  
    print('AUC: %.2f' % auc)
    le = preprocessing.LabelEncoder()
    y_test1=le.fit_transform(Y1)
    fpr1, tpr1, thresholds = roc_curve(y_test1, probs)
    #fig.canvas.set_window_title('XGBoost')
    plot_roc_curve(fpr1, tpr1)


    #Classification Report
    target_names = ['Yes', 'No']
    prediction=model.predict(X1)
    print(classification_report(Y1, prediction, target_names=target_names))
    classes = ["Yes", "No"]
    visualizer1 = ClassificationReport(model, classes=classes, support=True)
    visualizer1.fit(X, Y)  
    visualizer1.score(X1, Y1)
    #fig.canvas.set_window_title('XGBoost')  
    g = visualizer1.poof()




    
    
    
    
    
    
    
    
    
    
    print("___________________________SVM__________________________________________") 
    clf = svm.SVC(kernel='linear',probability=True)
    clf.fit(X, Y)
    print("_____________Report___________________")
    y_pred = clf.predict(X1)
    #print(cal_accuracy(Y1, y_pred))
    acc2=cal_accuracy(Y1, y_pred)
    #print("_____________user input ___________________")

    #confusion Matrix
    matrix =confusion_matrix(Y1, y_pred)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    fig.canvas.set_window_title('SVM')
    plt.show()

    #ROC_AUC curve
    probs = clf.predict_proba(X1) 
    probs = probs[:, 1]  
    auc = roc_auc_score(Y1, probs)  
    print('AUC: %.2f' % auc)
    le = preprocessing.LabelEncoder()
    y_test1=le.fit_transform(Y1)
    fpr, tpr, thresholds = roc_curve(y_test1, probs)
    #fig.canvas.set_window_title('SVM')
    plot_roc_curve(fpr, tpr)


    #Classification Report
    target_names = ['Yes', 'No']
    prediction=clf.predict(X1)
    print(classification_report(Y1, prediction, target_names=target_names))
    classes = ["Yes", "No"]
    visualizer = ClassificationReport(clf, classes=classes, support=True)
    visualizer.fit(X, Y)  
    visualizer.score(X1, Y1) 
    #fig.canvas.set_window_title('SVM') 
    g = visualizer.poof()






   

    
    

    print("___________________________Decison Tree__________________________________________") 
    gnb = DecisionTreeClassifier() 
    gnb.fit(X, Y)
    print("_____________Report___________________")
    y_pred = gnb.predict(X1)
    #print(cal_accuracy(Y1, y_pred))
    acc3=cal_accuracy(Y1, y_pred)
    #print("_____________user input ___________________")
   
    #confusion Matrix
    matrix =confusion_matrix(Y1, y_pred)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    fig.canvas.set_window_title('Decision Tree')
    plt.show()

    #ROC_AUC curve
    probs = gnb.predict_proba(X1) 
    probs = probs[:, 1]  
    auc = roc_auc_score(Y1, probs)  
    print('AUC: %.2f' % auc)
    le = preprocessing.LabelEncoder()
    y_test1=le.fit_transform(Y1)
    fpr, tpr, thresholds = roc_curve(y_test1, probs)
    #fig.canvas.set_window_title('NB')
    plot_roc_curve(fpr, tpr)


    #Classification Report
    target_names = ['Yes', 'No']
    prediction=gnb.predict(X1)
    print(classification_report(Y1, prediction, target_names=target_names))
    classes = ["Yes", "No"]
    visualizer = ClassificationReport(gnb, classes=classes, support=True)
    visualizer.fit(X, Y)  
    visualizer.score(X1, Y1) 
    #fig.canvas.set_window_title('NB') 
    g = visualizer.poof()



    labels = ['RandomForest','SVM','Decision Tree']
    #sizes = [5, neg_per, neu_per]
    sizes = [acc1,acc2,acc3]
    index = np.arange(len(labels))
    plt.bar(index, sizes)
    plt.xlabel('Algorithm', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(index, labels, fontsize=10, rotation=0)
    plt.title('comparative study')
    plt.show()













  
if __name__== "__main__":
  main()





    
