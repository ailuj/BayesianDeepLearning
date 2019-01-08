import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import tensorflow as tf
import edward as ed


data = pd.read_csv("data/loan.csv", low_memory=False)
print(data.shape)

print("preparing data")
#True if default, False if not default

data['Default_Binary'] = False
data.Default_Binary = data.loan_status.str.contains(r'Late|Default|Charged Off')

#print(data.Default_Binary.head())
data['Purpose_Cat'] = 0

category_dict = {'debt_consolidation': 1, 'credit_card': 2, 'home_improvement': 3,
                 'other': 4, 'major_purchase': 5, 'small_business': 6,
                 'car': 7, 'medical': 8, 'moving': 9,
                 'vacation': 10, 'house': 11, 'wedding': 12,
                 'renewable_energy': 13, 'educational': 14}

for index, value in data.purpose.iteritems():
    data.loc[index, "Purpose_Cat"] = category_dict[value]

#print(data.purpose)

"""print("preparing test set")
df_train = pd.get_dummies(data["purpose"])
df_train['Default_Binary'] = data['Default_Binary']
df_train['int_rate'] = data['int_rate']
df_train['funded_amnt'] = data['funded_amnt']

print(list(data))"""
df_train = pd.get_dummies(data.purpose).astype(int)

"""df_train.columns = ['debt_consolidation', 'credit_card', 'home_improvement',
                    'other', 'major_purchase', 'small_business', 'car', 'medical',
                    'moving', 'vacation', 'house', 'wedding', 'renewable_energy', 'educational']"""

# Also add the target column we created at first
#df_binary = pd.get_dummies(data.Default_Binary).astype(int)
#print(df_binary.head())
#df_train.head()

"""x_reshape = np.array(data.int_rate.values).reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x_reshape)
data['int_rate_scaled'] = pd.DataFrame(x_scaled)
#print(data.int_rate_scaled[0:5])

x_reshape = np.array(data.funded_amnt.values).reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x_reshape)
data['funded_amnt_scaled'] = pd.DataFrame(x_scaled)
#print(data.funded_amnt_scaled[0:5])"""

df_train['int_rate'] = data['int_rate']
df_train['funded_amnt'] = data['funded_amnt']
default = pd.get_dummies(data['Default_Binary']).astype(int)
df_train['Default_True'] = default[True]
df_train['Default_False'] = default[False]

#df_train['Default_True'] = df_binary[True]
#df_train['Default_False'] = df_binary[False]
#print(df_train.dtypes)

COLUMNS = ['debt_consolidation','credit_card','home_improvement',
           'other','major_purchase','small_business','car','medical',
           'moving','vacation','house','wedding','renewable_energy','educational',
           'funded_amnt_scaled','int_rate_scaled','Default_Binary']

FEATURES = ['debt_consolidation','credit_card','home_improvement',
           'other','major_purchase','small_business','car','medical',
           'moving','vacation','house','wedding','renewable_energy','educational',
           'funded_amnt_scaled','int_rate_scaled']

#CONTINUOUS_COLUMNS = ['funded_amnt_scaled','int_rate_scaled']
#CATEGORICAL_COLUMNS = ['Purpose_Cat']

LABEL = 'Default_Binary'

data_x = df_train.loc[:, :'funded_amnt'].as_matrix().astype(np.float32)
data_y = df_train.loc[:, 'Default_True':].as_matrix().astype(np.float32)

#print(data_y.head())
#config = tf.contrib.learn.RunConfig(keep_checkpoint_max=1) ######## DO NOT DELETE

#might need to lower N for smaller data sets
N = 7000
train_x, test_x = data_x[:N], data_x[N:]
train_y, test_y = data_y[:N], data_y[N:]

feature_num = train_x.shape[1]

in_size = train_x.shape[1]
print("IN SIZE")
print(in_size)
out_size = train_y.shape[1]
print("OUT SIZE")
print(out_size)

EPOCH_NUM = 10
BATCH_SIZE = 1000

# for bayesian neural network
train_y2 = np.argmax(train_y, axis=1)

test_y2 = np.argmax(test_y, axis=1)
x_ = tf.placeholder(tf.float32, shape=(None, in_size))
y_ = tf.placeholder(tf.int32, shape=BATCH_SIZE)

w = ed.models.Normal(loc=tf.zeros([in_size, out_size]), scale=tf.ones([in_size, out_size]))
b = ed.models.Normal(loc=tf.zeros([out_size]), scale=tf.ones([out_size]))
y_pre = ed.models.Categorical(tf.matmul(x_, w) + b)

qw = ed.models.Normal(loc=tf.Variable(tf.random_normal([in_size, out_size])), scale=tf.Variable(tf.random_normal([in_size, out_size])))
qb = ed.models.Normal(loc=tf.Variable(tf.random_normal([out_size])), scale=tf.Variable(tf.random_normal([out_size])))

y = ed.models.Categorical(tf.matmul(x_, qw) + qb)

inference = ed.KLqp({w: qw, b: qb}, data={y_pre: y_})
inference.initialize()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

with sess:
    samples_num = 100
    for epoch in tqdm(range(EPOCH_NUM), file=sys.stdout):
        perm = np.random.permutation(N)
        for i in range(0, N, BATCH_SIZE):
            batch_x = train_x[perm[i:i+BATCH_SIZE]]
            batch_y = train_y2[perm[i:i+BATCH_SIZE]]
            inference.update(feed_dict={x_: batch_x, y_: batch_y})
        y_samples = y.sample(samples_num).eval(feed_dict={x_: train_x})
        acc = (np.round(y_samples.sum(axis=0) / samples_num) == train_y2).mean()
        y_samples = y.sample(samples_num).eval(feed_dict={x_: test_x})
        test_acc = (np.round(y_samples.sum(axis=0) / samples_num) == test_y2).mean()
        if (epoch+1) % 1 == 0:
            tqdm.write('epoch:\t{}\taccuracy:\t{}\tvalidation accuracy:\t{}'.format(epoch+1, acc, test_acc))