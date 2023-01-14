import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

print(X)
print(Y)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])],     remainder='passthrough')

l1 = LabelEncoder() # ulkeyi kodlamak için 1 numarali etiket kodlayici nesnesi oluşturma (ozelliklerde dizin 1)
X[:, 1] = l1.fit_transform(X[:, 1])# dizeden sadece 3 numaraya kadar kodlama bölgesi. s 0,1,2
l2 = LabelEncoder() #cinsiyeti kodlamak için 2 numarali etiket kodlayici nesnesi oluşturma (özelliklerdeki dizin 2)
X[:, 2] = l2.fit_transform(X[:, 2]) #cinsiyetin dizeden sadece 2 numaraya kodlanması.s 0,1(erkek,kadın)


onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Keras kitapliklarini ve paketlerini ice aktarma
from keras.models import Sequential # Sinir Agini katman katman olusturmak icin
from keras.layers import Dense, Dropout# Agirliklari 0'a yakin küçük sayilara rastgele baslatmak icin(0 degil)

# ANN baslatma
# Yani aslinda derin bir ögrenme modelini baslatmanin 2 yolu var
#1) Her katmani tek tek tanimlama
#2) Grafik Tanimlama
clf = Sequential()
clf.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))
clf.add(Dropout(rate=0.15))
clf.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))
clf.add(Dropout(rate=0.15))
clf.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

clf.fit(X_train, Y_train, batch_size=10, epochs=100)

Y_pred = clf.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Karisiklik Matrisini Yapmak
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(Y_test, Y_pred)
print(cm)
#print("report:  /n", classification_report(Y_test, Y_pred))



#modeli dosyaya ekleme
clf.save('my_model.h5')

model_json = clf.to_json()
with open("./model.json", "w") as json_file:
    json_file.write(model_json)

clf.save_weights("./model.h5")
print("DOSYA HAZIR")