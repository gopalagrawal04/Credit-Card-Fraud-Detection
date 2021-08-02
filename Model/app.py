from flask  import*
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
credit_df=pd.read_csv('cleanedcsv.csv')

#df - Selects data by the label of the rows and columns.
x_data=credit_df[credit_df.columns[0:6]]
y_data=credit_df[credit_df.columns[6]]

#Read the .csv file
creditdata=pd.read_csv('credits.csv')


#List - Get Column Names as List in Pandas DataFrame
labels=list(creditdata.columns)
def enc(label,in_t=False):
        #LabelEncoder - Encode target labels with value between 0 and n_classes-1.
	lbe=LabelEncoder()
	#fit - It accepts an input for the sample data ( X ) and for supervised models it also accepts an argument for labels (i.e. target data y ).
	lbe.fit(creditdata[label])
	res =list(lbe.classes_)
	trans=list(lbe.transform(res))
	if(in_t==False):
		return trans
	else:
                #pass - is a null statement
		pass
	
def inverse(label):
        #LabelEncoder - Encode target labels with value between 0 and n_classes-1.
	lbe=LabelEncoder()
	lbe.fit(creditdata[label])
	res =list(lbe.classes_)
	return res
e_card_company,e_trans_freq,e_persons,e_amount,e_age,e_education,e_class=list(map(enc,creditdata.columns))
i_card_company,i_trans_freq,i_persons,i_amount,i_age,i_education,i_class_=list(map(inverse,creditdata.columns))

#LabelEncoder - transformer should be used to encode target values
card_company=LabelEncoder()
trans_freq=LabelEncoder()
persons=LabelEncoder()
age=LabelEncoder()
amount=LabelEncoder()
education=LabelEncoder()
class_=LabelEncoder()
##List - Get Column Names as List in Pandas DataFrame
card_company=list(card_company.fit_transform(credit_df['card_company']))
trans_freq=list(trans_freq.fit_transform(credit_df['trans_freq']))
persons=list(persons.fit_transform(credit_df['persons']))
amount=list(amount.fit_transform(credit_df['amount']))
age=list(age.fit_transform(credit_df['age']))
education=list(education.fit_transform(credit_df['education']))
class_=list(class_.fit_transform(credit_df['class']))
x_data=list(zip(card_company,trans_freq,persons,amount,age,education))
y_data=class_
names=['It seems to be very dangerous','It might not be safe','it is safe','It is very safe and secure']

#Make 20% data to test model
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2)
#Apply ML model
#K-NN algorithm stores all the available data and classifies a new data point based on the similarity
#The k-nearest neighbors (KNN) algorithm is a simple, supervised machine learning algorithm that can be used to solve both classification and regression problems.
model=KNeighborsClassifier(n_neighbors=11)
model.fit(x_train,y_train)
print('Accuracy Score : ', model.score(x_test,y_test)*100,'%')
#Persist an arbitrary Python object into one file.
joblib.dump(model,'model')
jmodel=joblib.load('model')

#pre=jmodel.predict([[0, 2, 3, 2, 1, 0]])
#print(names[pre[0]])

#import flask
app=Flask(__name__)
#/ - Make new page to open
@app.route('/')
def home():
	return render_template('index.html')
#Method POST will send the request to the model
@app.route('/',methods=['POST'])
def result():
        #index - which searches for a given element from the start of the list and returns the lowest index where the element appears
        #Decides which output case has major chances
	card=i_card_company.index(request.form['card_company'])
	trans=i_trans_freq.index(request.form['trans_freq'])
	per=i_persons.index(request.form['persons'])
	amount=i_amount.index(request.form['amount'])
	age=i_age.index(request.form['age'])
	edu=i_education.index(request.form['education'])

	pre=list(jmodel.predict([[card,trans,per,amount,age,edu]]))
	res=i_class_[pre[0]]

	if res=="It seems to be very dangerous":
	
	  return render_template('indexd.html',result=res)

	elif res=="It might not be safe":
	  return render_template('index.html',result=res) 
	
	else :
		return render_template('indexs.html',result=res) 

#Given a port address to run
if(__name__=='__main__'):
	app.run(debug=True,port=8080)
