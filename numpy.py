import numpy as np
# x = np.array([1, 2, 3, 4])
# print(type(x))
# print(x)
# x=[1,2,3,4]
# y=np.array(x)
# print(y)

# y=np.array([1,2,3,4,5])
# print(y)
# print(y.ndim)
# print(type(y))

# l=[]
# for i in range(1,5):
#     int_1=input("enter:")
#     l.append(int_1)
# print(np.array(l))

# ar2=np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
# print(type(ar2))
# print(ar2.ndim)

# ar3=np.array([[[1,2,3,4],[1,2,3,4],[1,2,3,4]]])
# print(ar3)
# print(ar3.ndim)

# arn=np.array([1,2,3,4],ndmin=10)
# print(arn)
# print(arn.ndim)
 
####how to create numpy array using numpy function
# ar_zero=np.zeros(5)
# print(ar_zero)
# ar_zero1=np.zeros((3,4))
# print(ar_zero1)
 
# ar_one=np.ones(4)
# print(ar_one)
# ar_empty=np.empty(4)
# print(ar_empty)
# ar_rn=np.arange(4)
# print(ar_rn)
# dia=np.eye(3)
# print(dia)
# dia=np.eye(3,5)
# print(dia)

# ar_line=np.linspace(1,10,num=5)
# print(ar_line)

# var1=np.random.rand(3)
# print(var1)
# var2=np.random.rand(3,5)
# print(var2)
# var3=np.random.randn(3)
# print(var3)
# var4=np.random.ranf(7)
# print(var4)
# var5=np.random.randint(3,10,8)
# print(var5)

#### datatypes in numpy
# x1=np.array([1,2,3,4])
# print("DATA TYPE:",x1.dtype)
# x2=np.array([1.0,2.9,3.5,4.3])
# print("DATA TYPE:",x2.dtype)
# x3=np.array(["1.6sd","2.5d","3k34","4jj"])
# print("DATA TYPE:",x3.dtype)
# x4=np.array([1,2,"3kl","jk4"])
# print("DATA TYPE:",x4.dtype)
# x5=np.array([1,2.9,"3kl","jk4"])
# print("DATA TYPE:",x5.dtype)
# x6=np.array([1.8,2.56,"3ff","894"])
# print("DATA TYPE:",x6.dtype)
# x7=np.array([1,2,3,4],dtype=np.int64)
# print("DATA TYPE:",x7.dtype)
# x8=np.array([1,2,3,4],dtype="f")
# print("DATA TYPE:",x8.dtype)
# print(x8)

# x9=np.array([1,2,3,4])
# new=np.float_(x9)
# new_one=np.int_(new)
# print(x9)
# print(new)
# print(new_one)
# print(x9.dtype)
# print(new.dtype)
# print(new_one.dtype)

# x9=np.array([4,5,6,7])
# n1=x9.astype(float)
# print(n1)

#### arthematic operations
# k1=np.array([1,2,3,4])
# print(k1+2)
# print(np.add(k1,2))

# k1=np.array([1,2,3,4])
# print(k1-2)
# print(np.subtract(k1,2))

# k1=np.array([1,2,3,4])
# print(k1*2)
# print(np.multiply(k1,2))

# k1=np.array([1,2,3,4])
# print(k1/2)
# print(np.divide(k1,2))

# k1=np.array([1,2,3,4])
# print(k1%2)
# print(np.mod(k1,2))

# k1=np.array([1,2,3,4])
# print(k1**2)
# print(np.power(k1,2))



# k1=np.array([1,2,3,4])
# k2=np.array([4,5,6,7])
# print(k1+k2)
# print(np.add(k1,k2))

# k3=np.array([1,2,3,4])
# k4=np.array([4,5,6,7])
# print(k3-k4)
# print(np.subtract(k3,k4))
 
# s1=np.array([1,2,3,4])
# s2=np.array([4,5,6,7])
# print(s1*s2)
# print(np.multiply(s1,s2))

# d1=np.array([1,2,3,4])
# d2=np.array([4,5,6,7])
# print(d1/d2)
# print(np.divide(d1,d2))

# k1=np.array([1,2,3,4])
# k2=np.array([4,5,6,7])
# print(k1%k2)
# print(np.mod(k1,k2))

# k1=np.array([1,2,3,4])
# k2=np.array([4,5,6,7])
# print(k1**k2)
# print(np.power(k1,k2))

# k2=np.array([4,5,6,7])
# print(1/k2)
# print(np.reciprocal(k2))

#### ARTHEMATIC FUNCTIONS
# v1=np.array([5,6,7,2,8])
# print(np.min(v1),np.argmin(v1))
# v2=np.array([5,6,7,2,8])
# print(np.max(v2),np.argmax(v2))
# print(np.sqrt(v2))
# print(np.sin(v2))
# print(np.cos(v2))
# n8=np.array([[3,2,5,6,7],[4,9,6,8,8]])
# print(np.min(n8,axis=0))

####shape and reshape in numpy arrays  
# l=np.array([[3,2,5,6,7],[4,9,6,8,8]])
# print(l.shape)
# v2=np.array([5,6,7,2,8],ndmin=3)
# print(v2.shape)
# k4=np.array([1,2,3,4,5,6])
# print(k4)
# print(k4.ndim)
# l=k4.reshape(3,2)
# print(l)
# print(l.ndim)
# o=np.array([5,6,7,5,6,7,7,8,9,2,8,4])
# print(o)
# print(o.ndim)
# print()
# j=o.reshape(2,3,2)
# print(j)
# print(j.ndim)
# print()
# m=j.reshape(-1)
# print(m)
# print(m.ndim)

###broadcasting in numpy arrays
# v1=np.array([1,2,3])
# v2=np.array([[4],[5],[6],[5]])
# print(v1+v2)
# x=np.array([[1],[2]])
# print(x.shape)
# y=np.array([[1,2,3],[4,5,6]])
# print(y.shape)
# print(x+y)

###indexing and slicing
# v1=np.array([1,2,3])
# print(v1[2])
# y=np.array([[1,2,3],[4,5,6]])
# print(y[0][2])
# z=np.array([[[1,2],[6,7]]])
# print(z[0][1][0])

# v1=np.array([1,2,3,4,5,6,7])
# print("2 to 5",v1[1:5])
# print("2 to end",v1[1:])
# print("start to 5",v1[:5])
# print("step2",v1[2:5:2])
# y=np.array([[1,2,3,4,5],[9,8,7,6,5],[11,12,13,14,15]])
# print(y[2,1:])

###iterating numpy arrays
# v1=np.array([1,2,3,4,5,6,7])
# for i in v1:
#     print(i)
# y=np.array([[1,2,3,4,5],[9,8,7,6,5],[11,12,13,14,15]])
# for i in y:
#     for j in i:
#         print(j)
# z=np.array([[[1,2],[6,7]]])
# for i in z:
#     for j in i:
#         for k in j:
#             print(k)

# z=np.array([[[1,2,8],[6,7,4]]])
# for i in np.nditer(z):
#     print(i)

# z=np.array([[[1,2,9],[6,7,4]]])
# for i in np.nditer(z,flags=["buffered"],op_dtypes=["S"]):
#     print(i)

# z=np.array([[[1,2,8],[6,7,4]]])
# for i,d in np.ndenumerate(z):
#     print(i,d)



######COPY VS VIEW IN NUMPY
# o1=np.array([1,2,3,4,5,6,7])
# c1=o1.copy()
# c1[1]=99
# print(o1)
# print(c1)

# o2=np.array([2,3,4,5,6])
# v1=o2.view()
# o2[1]=99
# print(o2)
# print(v1)


######joining and split numpy arrays
# o2=np.array([2,3,4,5,6])
# o1=np.array([1,2,3,4,5,6,7])
# a=np.concatenate((o1,o2))
# print(a)

# y=np.array([[1,2],[3,4]])
# w=np.array([[9,8],[7,6]])
# a=np.concatenate((y,w),axis=0)
# print(a)

# y=np.array([[1,2],[3,4]])
# w=np.array([[9,8],[7,6]])
# a=np.stack((y,w))
# print(a)

# o2=np.array([2,3,4,5,6])
# o1=np.array([1,2,3,4,5,])
# a=np.vstack((o1,o2))
# b=np.hstack((o1,o2))
# c=np.stack((o1,o2),axis=1)
# b=np.dstack((o1,o2))
# print(a)
# print(b)
# print(c)

# o2=np.array([1,2,3,4,5,6])
# ar=np.array_split(o2,3)
# print(ar)
# print(type(ar))
# print(ar[1])

# w=np.array([[9,8],[7,6],[6,7]])
# ar=np.array_split(w,3)
# ar1=np.array_split(w,3,axis=1)
# print(ar)
# print(ar1)
# print(type(ar))
# print(ar[1])
# print(ar1[1])



#####NUMPY FUNCTIONS
# n=np.array([1,2,3,4,9,3,2,1,2,3])
# x=np.where((n%2)==0)
# print(x)

# n=np.array([1,2,3,4,9,3,2,1,6,7,8,3])
# x=np.searchsorted(n,11,side="right") 
# print(x)

# n=np.array([1,2,3,4,9,3,2,1,6,7,8,3])
# print(np.sort(n))
# k=np.array(['a','e','d','f'])
# print(np.sort(k))

# w=np.array([[9,6,4,4,8],[2,2,2,4,6]])
# print(np.sort(w)

# k=np.array(['a','e','d','f'])
# f=[True,False,False,True]
# n=k[f]         #filter
# print(n)
# print(type(n))
# n=np.array([1,2,3,4,9,3,2,1,6,7,8,3])
# np.random.shuffle(n)
# print(n)
# x=np.unique(n,return_index=True,return_counts=True)
# print(x)
# y=np.resize(n,(3,4))
# print(y)
# k=np.array([[3,4],[5,6]])
# print(k.flatten(order="F"))
# print(np.ravel(k,order="c"))



####insert and delete
# v1=np.array([[3,4,5],[8,5,6]])
# s=np.insert(v1,2,78,axis=1)
# p=np.insert(v1,2,78,axis=0)
# x=np.insert(v1,2,[23,24,25],axis=0)
# print(s)
# print(p)
# print(x)


# n=np.array([1,2,3,4,9,3,2,1,2,3])
# x=np.append(n,6.5)
# x1=np.insert(n,2,66)
# print(x)
# print(x1)

# v1=np.array([[1,2,3],[1,2,3]])
# v=np.append(v1,[[23,24,25]],axis=0)
# print(v)

# n=np.array([1,2,3,4,9,3,2,1,2,3])
# d=np.delete(n,2)
# print(d)

####matrix in python
v1=np.matrix([[1,2],[3,4]])
v2=np.matrix([[1,2],[1,2]])
# print(v1+v2)
# print(v1*v2)
# print(v1.dot(v2))
# print(type(v1))
# print(np.transpose(v1))
# print(v1.T)
# print(np.swapaxes(v1,0,1))
# print(np.linalg.inv(v1))
# print(np.linalg.matrix_power(v1,-2))
print(np.linalg.det(v1))
