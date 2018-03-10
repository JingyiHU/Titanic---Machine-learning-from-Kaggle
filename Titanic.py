# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np   #科学计算
import pandas as pd  

from pandas import Series, DataFrame

data_train = pd.read_csv("~/Desktop/kaggle/train.csv")

data_train

data_train.info() #我们发现一些列比如 Cabin客舱有缺失值（缺的很多） age也是 

data_train.describe()

import matplotlib.pyplot as plt


#乘客各属性分布
fig = plt.figure()
fig.set(alpha = 0.2)# 设定图表颜色alpha参数

plt.subplot2grid((2,3),(0,0)) # 在一张大图里分列几个小图  
data_train.Survived.value_counts().plot(kind = 'bar') #柱状图
plt.title(u"survive or dead (1 correspond survive)") # 标题
plt.ylabel(u"number of people")  

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind = 'bar')
plt.title(u"class for passenger")
plt.ylabel(u"number of people")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"age")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y') #这句不懂 好像去掉也没事
plt.title(u"survive or not regards to age (1 is survive)")

plt.subplot2grid((2,3),(1,0), colspan = 2)
data_train.Age[data_train.Pclass == 1].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 2].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 3].plot(kind = 'kde')
plt.xlabel(u"age")
plt.ylabel(u"densite")
plt.title(u"class - age")
plt.legend((u'1st',u'2st',u'3st'),loc = 'best') #为图设置legends

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind = 'bar')
plt.title(u"embaked people of ports")
plt.ylabel(u"number of people")

plt.show()
#由图可以看出，获救总人数大约只有300多人。
#3等级的人数非常多
#遇难的人和获救的人跨度都很广
#2，3等舱 20多岁的人最多，3等舱三四十的人比较多
#登舱口s的人比较多


#################属性与获救结果的关联统计
#看看各乘客等级的获救情况

fig = plt.figure()
fig.set(alpha = 0.2)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'survived':Survived_1,u'dead':Survived_0})
df.plot(kind = 'bar',stacked = True)
plt.title(u'survive - class')
plt.xlabel(u'class')
plt.ylabel('nb of people')  #不加u也可以
plt.show()

#图中可以看出，明显1等级的人获救的概率最大

##看看堂兄妹，父母/孩子有几人对是否获救的影响
g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print (df)

g = data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print (df)

#ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴把
#cabin只有204个乘客有值，我们先看看它的一个分布
print(data_train.Cabin.value_counts())

#先在有无Cabin信息这个粗粒度上看看Survived的情况

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
df.plot(kind='bar')
plt.title(u"按Cabin有无看获救情况")
plt.xlabel(u"Cabin有无") 
plt.ylabel(u"人数")
plt.show()

# =============================================================================
# 
# ######################################简单数据预处理
# #看完了感性的数据，就应该处理一下这些数据，为机器建模做准备
# #这就是很多kaggler津津乐道的特征工程过程
# 
# 
# # #####先处理丢失的数据
# # #从cabin和age入手  按照cabin有无数据分为yes或者no两种情况
# # #对于age，遇到缺省值一般的处理情况：
# # #1.如果缺值的样本占总数比例极高，我们可能就直接舍弃了，作为特征加入的话，可能反倒带入noise，影响最后的结果了
# # #2.如果缺值的样本适中，而该属性非连续值特征属性(比如说类目属性)，那就把NaN作为一个新类别，加到类别特征中
# # #3.如果缺值的样本适中，而该属性为连续值特征属性，有时候我们会考虑给定一个step(比如这里的age，我们可以考虑每隔2/3岁为一个步长)，然后把它离散化，之后把NaN作为一个type加到属性类目中
# # #4.有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上
# # 
# # #这里用scikit-learn中的RandomForest来拟合一下缺失的年龄数据(注：RandomForest是一个用在原始数据中做不同采样，
# # #建立DecisionTrees，再进行average等等来降低过拟合现象，提高结果的机器学习算法
# =============================================================================


from sklearn.ensemble import RandomForestRegressor

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
     # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    
    # y即目标年龄
    y = known_age[:,0]#df index从0开始
    # X即特征属性值
    X = known_age[:,1:]
    
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:,1::])
    
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull(), 'Age')] = predictedAges
    
    return df, rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

# =============================================================================
# 因为逻辑回归建模时，需要输入的特征都是数值型特征，我们通常会先对类目型的特征因子化。 
# 什么叫做因子化呢？举个例子：
# 
# 以Cabin为例，原本一个属性维度，因为其取值可以是[‘yes’,’no’]，而将其平展开为’Cabin_yes’,’Cabin_no’两个属性
# 
# 原本Cabin取值为yes的，在此处的”Cabin_yes”下取值为1，在”Cabin_no”下取值为0
# 原本Cabin取值为no的，在此处的”Cabin_yes”下取值为0，在”Cabin_no”下取值为1   
# 
# =============================================================================

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix = 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix = 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix = 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix = 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis = 1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)

# =============================================================================
# #这样看起来我们需要的属性值都有了 并且它们都是数值型属性
# 但是仔细观察fare和age两个属性就会发现它们的数值变化程度很大，将对收敛速度造成很大的影响，
# 甚至不收敛。所以我们先用scikit-learn里面的preprocessing模块对这俩货做一个scaling，
# 所谓scaling，其实就是将一些变化幅度较大的特征化到[-1,1]之内。
# =============================================================================

import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()

scale_param = scaler.fit(df[['Age', 'Fare']])  #############################focus!!

df['Age_scaled'] = scaler.fit_transform(df[['Age', 'Fare']], scale_param)[:, 0]
df['Fare_scaled'] = scaler.fit_transform(df[['Age', 'Fare']], scale_param)[:, 1]

#print(df['Age_scaled']) #ok!

# =============================================================================
# 万事俱备，只欠建模，我们把需要的属性值取出来转成scikit-learn里面LogisticRegression可以处理的格式。
# =============================================================================

# =============================================================================
# 逻辑斯蒂建模，我们把需要的feature取出来，转换成numpy形式
# =============================================================================
from sklearn import linear_model

train_df = df.filter(regex = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:,0]

# X即特征属性值
X = train_np[:,1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

# =============================================================================
# test要和train做像前面一样的处理才可以放进model里面。
# =============================================================================

data_test = pd.read_csv("~/Desktop/kaggle/test.csv")

data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0

# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]#多个feature -->[[]]

null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


scale_param = scaler.fit(df_test[['Age', 'Fare']])

df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age', 'Fare']], scale_param)[:, 0]
df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Age', 'Fare']], scale_param)[:, 1]


# =============================================================================
# 做预测取结果
# =============================================================================

test = df_test.filter(regex = 'Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
#输出一个dataframe 一列是乘客id 一列是预测结果
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("~/Desktop/kaggle/logistic_regression_predictions.csv", index=False)

# =============================================================================
# =============================================================================
# # 逻辑回归系统优化
# =============================================================================
# =============================================================================

#这才是刚刚做好一个baseline model还需要进行优化
#需要分析模型现在是欠拟合还是过拟合？所以需要相应的增加特征还是增加数据？

#不过现在我们不急于做这个事情，因为我们现在这模型还是有些粗糙的，需要再挖掘一下：
##第一，我们舍弃了两个属性，name和ticket，因为这两个每一个记录都是一个完全不同的值，我们没有找到好的处理方式
##第二，我们用rf处理年龄也不一定正确，因为我们根据其他属性，其实无法很好的预测出未知的年龄。
##再一个，以我们的日常经验，小盆友和老人可能得到的照顾会多一些，这样看的话，年龄作为一个连续值，
#给一个固定的系数，应该和年龄是一个正相关或者负相关，似乎体现不出两头受照顾的实际情况，
#所以，说不定我们把年龄离散化，按区段分作类别属性会更合适一些。

#为了证实猜测，我们把model系数和feature结合起来看看：
pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})

# =============================================================================
# 根据逻辑斯蒂回归，这些系数为正的特征，和最后结果是一个正相关，反之为负相关。
# 我们先看看那些权重绝对值非常大的feature，在我们的模型上：
# 1.sex：女性会增加最后获救的概率，而男性就相对较低
# 2.pclass：1等就会增加这个概率 而3等会严重降低获救概率
# 3.有Cabin值会很大程度拉升最后获救概率(这里似乎能看到了一点端倪，事实上从
# 最上面的有无Cabin记录的Survived分布图上看出，即使有Cabin记录的乘客也有一部分遇难了，
# 估计这个属性上我们挖掘还不够)
# 4.Age是一个负相关，意味着在我们的模型里，年龄越小，越有获救的优先权(还得回原数据看看这个是否合理）
# 5.有一个登船港口S会很大程度拉低获救的概率，另外俩港口压根就没啥作用(这个实际上非常奇怪，
# 因为我们从之前的统计图上并没有看到S港口的获救率非常低，所以也许可以考虑把登船港口这个feature去掉试试)。
# 6.船票Fare有小幅度的正相关(并不意味着这个feature作用不大，有可能是我们细化的程度还不够，
# 举个例子，说不定我们得对它离散化，再分至各个乘客等级上？)
# 
# 我们怎么知道哪个feature是更有用的呢？test数据也没有一列叫做survived
#总不能做一次更改就提交一次结果吧？！
# 
# =============================================================================





# =============================================================================
# =============================================================================
# # #交叉验证
# =============================================================================
# =============================================================================
# =============================================================================
# 
# 我们通常情况下，这么做cross validation：把train.csv分成两部分，
# 一部分用于训练我们需要的模型，另外一部分数据上看我们预测算法的效果。
# 我们用scikit-learn的cross_validation来帮我们完成小数据集上的这个工作。
# 先简单看看cross validation情况下的打分
# 
# =============================================================================


from sklearn import cross_validation

 #简单看看打分情况
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.as_matrix()[:,1:]
y = all_data.as_matrix()[:,0]
print(cross_validation.cross_val_score(clf, X, y, cv=5))

#result:[0.81564246 0.81564246 0.78651685 0.78651685 0.81355932]

#既然我们要做交叉验证，那我们干脆先把交叉验证里面的bad case拿出来看看，看看人眼审核，是否能发现什么蛛丝马迹，
#是我们忽略了哪些信息，使得这些乘客被判定错了。再把bad case上得到的想法和前头系数分析的合在一起，然后逐个试试。

# =============================================================================
# bad case
# =============================================================================

#下面我们做数据分割，并且在原始数据集上瞄一眼bad case：
########################################################这个350行一直有问题做不出来。

# =============================================================================
# =============================================================================
# # # # 分割数据，按照 训练数据:cv数据 = 7:3的比例
# # X_train, X_test, y_train, y_test = cross_validation.train_test_split(df.iloc[:,1:], df.iloc[:,0], test_size=0.3, random_state=0)
# #  
# # train_df = X_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# # # # 生成模型
# # clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# # clf.fit(X_train.as_matrix(), y_train)
# # 
# # # # 对cross validation数据进行预测
# # 
# # cv_df = X_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# # predictions = clf.predict(cv_df.as_matrix())
# # origin_data_train = pd.read_csv("~/Desktop/kaggle/train.csv")
# # bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(X_test[predictions != y_test.as_matrix()]['PassengerId'].values)]
# # bad_cases
# =============================================================================
# =============================================================================


#现在有了”train_df” 和 “vc_df” 两个数据部分，前者用于训练model，后者用于评定和选择模型。
#拿到bad case以后，分析那些数据对结果的影响较大，可以作出一些组合属性，或者有些变量可以分成两个变量做，有些没用的可以取代哦etc。
# =============================================================================
# 我们随便列一些可能可以做的优化操作：
# 
# Age属性不使用现在的拟合方式，而是根据名称中的『Mr』『Mrs』『Miss』等的平均值进行填充。
# Age不做成一个连续值属性，而是使用一个步长进行离散化，变成离散的类目feature。
# Cabin再细化一些，对于有记录的Cabin属性，我们将其分为前面的字母部分(我猜是位置和船层之类的信息) 
# 和 后面的数字部分(应该是房间号，有意思的事情是，如果你仔细看看原始数据，你会发现，这个值大的情况下，似乎获救的可能性高一些)。
# Pclass和Sex俩太重要了，我们试着用它们去组出一个组合属性来试试，这也是另外一种程度的细化。
# 单加一个Child字段，Age<=12的，设为1，其余为0(你去看看数据，确实小盆友优先程度很高啊)
# 如果名字里面有『Mrs』，而Parch>1的，我们猜测她可能是一个母亲，应该获救的概率也会提高，
# 因此可以多加一个Mother字段，此种情况下设为1，其余情况下设为0
# 登船港口可以考虑先去掉试试(Q和C本来就没权重，S有点诡异)
# 把堂兄弟/兄妹 和 Parch 还有自己 个数加在一起组一个Family_size字段(考虑到大家族可能对最后的结果有影响)
# Name是一个我们一直没有触碰的属性，我们可以做一些简单的处理，比如说男性中带某些字眼的
# (‘Capt’, ‘Don’, ‘Major’, ‘Sir’)可以统一到一个Title，女性也一样。
# =============================================================================

# =============================================================================
# 试验的过程比较漫长，也需要有耐心，而且我们经常会面临很尴尬的状况，就是我们灵光一闪，
# 想到一个feature，然后坚信它一定有效，结果试验下来，效果还不如试验之前的结果。
# 恩，需要坚持和耐心，以及不断的挖掘。
# 
# 最好的结果是在：
# 『Survived~C(Pclass)+C(Title)+C(Sex)+C(Age_bucket)+C(Cabin_num_bucket)Mother+Fare+Family_Size』
# 
# =============================================================================


# =============================================================================
# =============================================================================
# # learning curves
# =============================================================================
# =============================================================================

# =============================================================================
# 有一个可能发生的问题就是，随着我们不断的做feature engineering，产生的特征越来越多，用这些特征去
# 训练我们的模型，会使训练结果越来越好，从而逐步丧失泛化能力，从而在待预测的数据上表现不佳。这就是
# 过拟合问题。
# 而在机器学习的问题上，对于过拟合和欠拟合两种情形。我们优化的方式是不同的。
# 对过拟合而言，通常以下策略对结果优化是有用的：
# 做一下feature selection，挑出较好的feature的subset来做training
# 提供更多的数据，从而弥补原始数据的bias问题，学习到的model也会更准确
# 而对于欠拟合而言，我们通常需要更多的feature，更复杂的模型来提高准确度。
# 
# 我们可以用sklearn里面的learning curve来帮我们分析模型的状态。
# 
# 我们一起画一下最先得到的baseline model的learning curve。
# 
# =============================================================================

import numpy as ny
import matplotlib.pyplot as plt

from sklearn.learning_curve import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"treaning set score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"cv score")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(clf, u"learning curve", X, y)


# =============================================================================
# 目前的曲线看来，我们的model并不处于overfitting的状态
# (overfitting的表现一般是训练集上得分高，而交叉验证集上要低很多，中间的gap比较大)。
# 因此我们可以再做些feature engineering的工作，添加一些新产出的特征或者组合特征到模型中。
# =============================================================================

# =============================================================================
# =============================================================================
# # 模型融合(model ensemble)
# =============================================================================
# =============================================================================

# =============================================================================
# 
# 我们要祭出机器学习/数据挖掘上通常最后会用到的大杀器了。恩，模型融合。
# 最简单的模型融合大概就是这么个意思，比如分类问题，当我们手头上有一堆在同一份数据集上
# 训练得到的分类器(比如logistic regression，SVM，KNN，random forest，神经网络)，
# 那我们让他们都分别去做判定，然后对结果做投票统计，取票数最多的结果为最后结果。
# 
# 模型融合可以比较好地缓解，训练过程中产生的过拟合问题，从而对于结果的准确度提升有一定的帮助。
# 
# 话说回来，回到我们现在的问题。你看，我们现在只讲了logistic regression，
# 如果我们还想用这个融合思想去提高我们的结果，我们该怎么做呢？
# 
# 既然这个时候模型没得选，那咱们就在数据上动动手脚咯。大家想想，如果模型出现过拟合现在，
# 一定是在我们的训练上出现拟合过度造成的对吧。
# 
# 那我们干脆就不要用全部的训练集，每次取训练集的一个subset，做训练，这样，
# 我们虽然用的是同一个机器学习算法，但是得到的模型却是不一样的；同时，因为我们没有任何一份子数据集是全的，
# 因此即使出现过拟合，也是在子训练集上出现过拟合，而不是全体数据上，这样做一个融合，可能对最后的结果有一定的帮助。
# 对，这就是常用的Bagging。
# =============================================================================


from sklearn.ensemble import BaggingRegressor

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到BaggingRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("~/Desktop/kaggle/logistic_regression_bagging_predictions.csv", index=False)













