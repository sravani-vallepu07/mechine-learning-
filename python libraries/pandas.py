import pandas as pd

# l=[1,2,3,4,5]
# vs=pd.DataFrame(l)
# print(vs)
# print(type(vs)) 


d={'a':[1,2,3,4,5,6,7],"s":[2,3,3,4,4,4,5],1:[1,2,3,4,9,0,9]}
k=pd.DataFrame(d,columns=["a",1,'s'],index=['s','r','a','v','a','n','i'])
print(k)
print(k[1]['v'])


# list1=[[1,2,3,4,5],[11,12,13,1,15]]
# vr=pd.DataFrame(list1)
# print(vr)
# print(type(vr)) 


# sr={'s':pd.Series([1,2,3,4]),'r':pd.Series([1,2,3,4])}
# ma=pd.DataFrame(sr)
# print(ma)
# print(type(ma)) 

###arthematic operators in python

# var=pd.DataFrame({'a':[1,2,3,4],'b':[5,6,7,8]})
# var['c']=var['a']+var['b']
# print(var)

# var1=pd.DataFrame({'a':[1,2,3,4],'b':[5,6,7,8]})
# var1['c']=var1['a']-var1['b']
# print(var1)

# var2=pd.DataFrame({'a':[1,2,3,4],'b':[5,6,7,8]})
# var2['c']=var2['a']*var2['b']
# print(var2)

# var3=pd.DataFrame({'a':[1,2,3,4],'b':[5,6,7,8]})
# var3['c']=var3['a']/var3['b']
# print(var3)

# var1["python"]=var1['c']>=20
# print(var1)

###handling missing data (replace and interpolate)
# d=pd.read_csv("C:\\Users\\valle\\OneDrive\\Desktop\\IPL IMB381IPL2013.csv")
# df=pd.DataFrame(d)
# df.replace(to_replace=1,value=400)
# print(df)
# df.insert(1,"python",df["Sl.NO."])
# print(df)
# df["hi"]=df["AGE"][:3]
# print(df)
# df1=df.pop("hi")
# print(df)

# del df["hi"]
# print(df)

### python pandas csv files[write csv files]
# dis={"a":[1,2,3,4,5,6],"s":[6,7,8,9,8,0],"d":[3,6,9,12,15,18]}
# d=pd.DataFrame(dis)
# print(d)
# d.to_csv("test_new.csv",index=False,header=[1,2,3])

### READ CSV FILES
# csv_1=pd.read_csv("C:\\Users\\valle\\OneDrive\\Desktop\\IPL IMB381IPL2013.csv")
# print(csv_1)
# csv_2=pd.read_csv("C:\\Users\\valle\\OneDrive\\Desktop\\IPL IMB381IPL2013.csv",nrows=3)
# print(csv_2)
# print(type(csv_2))
# csv_3=pd.read_csv("C:\\Users\\valle\\OneDrive\\Desktop\\IPL IMB381IPL2013.csv",usecols=["AGE","PLAYER NAME"])
# print(csv_3)
# csv_4=pd.read_csv("C:\\Users\\valle\\OneDrive\\Desktop\\IPL IMB381IPL2013.csv",usecols=[2,1])
# print(csv_4)
# csv_5=pd.read_csv("C:\\Users\\valle\\OneDrive\\Desktop\\IPL IMB381IPL2013.csv",index_col=["PLAYER NAME"])
# print(csv_5)
# csv_6=pd.read_csv("C:\\Users\\valle\\OneDrive\\Desktop\\IPL IMB381IPL2013.csv",header=2)
# print(csv_6)
# csv_7=pd.read_csv("C:\\Users\\valle\\OneDrive\\Desktop\\IPL IMB381IPL2013.csv",dtype={"AGE":float})
# print(csv_7)


###python csv file reading and writing complete tutorial


# dis={"a":["srav","durgi","meghana","madhu","prathusha","sarayu","kavya"],"s":[6,7,8,9,8,0,9],"d":[3,6,9,12,67,15,18]}
# d=pd.DataFrame(dis)
# d.to_csv("oyy.csv")
# csv_8=pd.read_csv("C:\\Users\\valle\\OneDrive\\Desktop\\python\\oyy.csv")
# print(csv_8)
# print(csv_8.index)
# print(csv_8.columns)
# print(csv_8.describe)
# print(csv_8.head())
# print(csv_8.head(2))
# print(csv_8.tail())
# print(csv_8.tail(2))
# print(csv_8[6:])#slicing
# print(type(csv_8))
# print(csv_8.index.array)
# print(csv_8.to_numpy())
# print(csv_8.sort_index(axis=0,ascending=False))
# csv_8["a"][0]="cr"
# print(csv_8)
# import numpy as np
# v=np.array(csv_8)
# print(v)

# csv_8.loc[0,"a"]="venky"
# print(csv_8)
# print(csv_8.loc[[2,3],["s","a"]])
# print(csv_8.iloc[0,1])
# print(csv_8.drop("d",axis=1))
# print(csv_8.drop(0,axis=0))

###handling missing values (dropna &fillna )in python pandas

# dis={"a":[1,2,3,4,5,6],"s":[6,7,8,9,8,0],"d":[3,6,9,12,15,18]}
# d=pd.DataFrame(dis)
# print(d)
# d.to_csv("test_new.csv",index=False,header=["a","b","c"])
# csv_8=pd.read_csv("C:\\Users\\valle\\OneDrive\\Desktop\\python\\test_new.csv")
# print(csv_8.dropna())
# print(csv_8.dropna(axis=1))#delete a col if any element in col has null
# print(csv_8.dropna(axis=0))#delete a row if any element in row has null
# print(csv_8.dropna(how="any"))#deleting row containg null
# print(csv_8.dropna(how="all"))#deleting the row which should containhg all null values
# csv_8.dropna(inplace=True)
# print(csv_8)
# print(csv_8.dropna(thresh=3))
# print(csv_8.fillna("potti"))
# print(csv_8.fillna({"a":"srav","b":"prema","c":"anitha"}))
# print(csv_8.fillna(method="ffill"))
# print(csv_8.fillna(method="bfill",axis=1))
# print(csv_8.fillna(12,inplace=True))
# print(csv_8.fillna(12,limit=3))

###PANDAS TUTORIAL FOR HANDLING MISIING DATA(replace and interpolate)


# print(csv_8.replace(to_replace=8,value=1))
# print(csv_8.replace([1,2,3,4,5,6,7,8,9,18,11,12,13,14],22))
# print(csv_8.replace([1-9],3,regex=True))
# print({"a":'[A-Z]'},22,regex=True)
# print(csv_8.replace(8,method="ffill"))
# print(csv_8.replace(8,method="bfill",limit=3,inplace=True))
# print(csv_8.interpolate())
# print(csv_8.interpolate(method="linear"))
# print(csv_8.interpolate(limit_direction="both",limit=2,axis=1))
# print(csv_8.interpolate(limit_area="inside"))

###merge
# var1=pd.DataFrame({"d":[1,2,3,4]})
# var2=pd.DataFrame({"A":[1,2,3,4],"B":[21,22,23,24]})
# print(pd.merge(var1,var2,on="A"))
# print(pd.merge(var1,var2,how="inner"))
# print(pd.merge(var1,var2,how="left"))
# print(pd.merge(var1,var2,how="right"))
# print(pd.merge(var1,var2,how="outer",indicator=True))
# print(pd.merge(var1,var2,left_index=True,right_index=True,suffixes=("name","python")))

# sr1=pd.Series([1,2,3,4])
# sr2=pd.Series([11,21,31,41])
# print(pd.concat([sr1,sr2]))
# print(pd.concat([var1,var2],axis=1,join="outer"))
# print(pd.concat([var1,var2],axis=1,join="inner"))
# print(pd.concat([var1,var2],axis=1,keys=["var1","var2"]))
# print(pd.concat([var1,var2]))

###pandas groupby
# k=pd.DataFrame({"name":["a","b","c","d","a","b","a","b","a","c","c","d"],"s_1":[12,13,14,12,13,14,15,23,25,16,10,34],"s2":[23,24,25,26,27,28,29,30,25,34,35,56]})
# print(k)
# k_new=k.groupby("name")
# print(k_new)
# for x,y in k_new:
#     print(x)
#     print(y)
#     print()
# print(k_new.get_group('a'))
# print(k_new.min())
# print(k_new.max())
# print(k_new.mean())
# print(list(k_new)) 

###join and append
# k1=pd.DataFrame({"a":[1,2,3,4],"b":[11,12,13,14]})
# k2=pd.DataFrame({"b":[10,20],"d":[33,44]})
# print(k1.join(k2))
# print(k2.join(k1))
# print(k2.join(k1,how="left"))
# print(k2.join(k1,how="right"))
# print(k2.join(k1,how="outer"))
# print(k2.join(k1,how="inner",lsuffix="_12"))
# print(k2.join(k1,how="outer",lsuffix="_12",rsuffix="_123"))
# l=k12)
# print(l)

###pivot table and melt function in pandas
# w=pd.DataFrame({"days":[1,1,1,1,2,2],"st_name":["a","b","a","a","b","b"],"eng":[10,12,14,15,16,12],"maths":[17,18,19,13,14,16]})
# print(pd.melt(w,id_vars=["eng"],var_name="python",value_name="wscube"))
# print(w.pivot(index="days",columns="st_name"))
# print(w.pivot(index="days",columns="st_name",values="eng"))
# print(w.pivot_table(index="st_name",columns="days",aggfunc="mean",margins="True"))