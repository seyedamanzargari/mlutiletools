import cv2
import numpy as np
import pandas as pd
import os
import requests
import io
os.system('pip install --upgrade xgboost')
import xgboost as xgb
import pathlib
from sklearn.metrics import mean_squared_error,accuracy_score,auc,roc_curve
try:
    import pyswarms as ps
except Exception:
    print(f"{Exception} \nPySwarms not installed...\n Install use 'pip install pyswarms'")
    try:
        os.system('pip install pyswarms')
        print("PySwarms installed")
        import pyswarms as ps
    except:
        exit()
#import tensorflow as tf
from time import time
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score

def load_images_from_folder(dirctory, image_size, label=None, cvt_color=None, array=True):
    images = []
    file_name = []
    image_size = tuple(image_size)
    for filename in os.listdir(dirctory):
        file_name.append(filename)
        img = cv2.imread(os.path.join(dirctory,filename))
        img = cv2.resize(img,image_size)
        if cvt_color is not None:
            cvt_color = list(cvt_color)
            for color in cvt_color:
                img = cv2.cvtColor(img, color)
        if img is not None:
            images.append(img)

    if array:
        images = np.array(images)

    if label is not None:
        labels = []
        for i in range(len(images)):
            labels.append(label)
        return images, labels    

    return images


def array_from_url(url, dtype='int32'):
    name = pathlib.Path(requests.utils.urlparse(url).path).name
    response = requests.get(url)
    if (response.status_code == 200):
        response.raise_for_status()
        data = np.load(io.BytesIO(response.content))
        data = data.astype(dtype)
        return data
    else:
        return response.status_code    


def keras_application_feature_extractor(model, dataset):
    modeldfeature = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    feature_extractor= modeldfeature.predict(dataset)
    feature = feature_extractor.reshape(feature_extractor.shape[0], -1)
    del feature_extractor
    return feature


def list_labeling(list, label):
    listlen = len(list)
    return [label for j in range(listlen)], listlen
    

def counts_from_confusion_matrix(confusion):
    counts_list = []
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]

        fn_mask = np.zeros(confusion.shape)
        fn_mask[i, :] = 1
        fn_mask[i, i] = 0
        fn = np.sum(np.multiply(confusion, fn_mask))

        fp_mask = np.zeros(confusion.shape)
        fp_mask[:, i] = 1
        fp_mask[i, i] = 0
        fp = np.sum(np.multiply(confusion, fp_mask))

        tn_mask = 1 - (fn_mask + fp_mask)
        tn_mask[i, i] = 0
        tn = np.sum(np.multiply(confusion, tn_mask))

        counts_list.append({'Class':i,'TP':tp,'FN':fn,'FP':fp,'TN':tn})

    return counts_list


def KFold_cross_val_test(model, X, Y, K=5, Report=False, shuffle=True, print_report=False, random_state=False):
    if random_state:
        cv = KFold(n_splits=K, shuffle=shuffle, random_state=random_state)
    else:
        cv = KFold(n_splits=K, shuffle=shuffle)    
    avgacc = 0
    f = 0
    report = []
    ticf=time()
    for train_ix, test_ix in cv.split(X):
        result={}
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, Y_test = Y[train_ix], Y[test_ix]
        
        tic = time()
        model.fit(train_X, train_y)
        toc = time()
        
        Y_pred = model.predict(test_X)
        acc = accuracy_score(Y_test, Y_pred)
        
        result["Accuracy score"] = acc
        avgacc += result["Accuracy score"]
        result["confusion matrix"] = confusion_matrix(Y_test, Y_pred)
        result["f1 score"] = f1_score(Y_test, Y_pred, average="macro")
        result["precision score"] = precision_score(Y_test, Y_pred, average="macro")
        result["Specifity"] = result["confusion matrix"][1,1] / (result["confusion matrix"][1,0] + result["confusion matrix"][1,1])
        result["Sensitivity"] = result["confusion matrix"][0,0] / (result["confusion matrix"][0,0] + result["confusion matrix"][0,1])
        
        f = f+1
        if print_report:
            print(f"\n ======== fold {f} ========")
            print(f"\nAccuracy score : ", result["Accuracy score"])
            print(f"\nconfusion matrix : \n", result["confusion matrix"])
            print(f"\nf1 score : ", result["f1 score"])
            print(f"\nprecision score : ", result["precision score"])
            print(f"\nSpecifity : ", result["Specifity"])
            print(f"\nSensitivity : ", result["Sensitivity"])
            print(f"\nTraining Time: {toc-tic:.3f} s")
            print(f"\n---------------------------\n")
        report.append(result)
    tocf=time()    
    total_acc = avgacc/5
    print('\nAccuracy : ', total_acc, f"Time: {tocf-ticf:.3f} s")
    if Report:
        return total_acc, report
    else:
        total_acc    


class pspso:
    best_paricle_cost_ann =None
    best_model_ann=None
    best_history_ann=None
    best_particle_position_ann=None
       
    verbose=0
    early_stopping=20
    
    defaultparams= None
    parameters=None
    paramdetails= None
    rounding=None

    def __init__(self, estimator='xgboost', params=None, task="binary classification",score= 'acc',):
        self.estimator = estimator
        self.task=task
        self.score=score
        self.cost=None
        self.pos=None
        self.model=None
        self.duration=None
        self.combined=None
        self.label=None
        self.rmse=None
        self.optimizer=None
        pspso.parameters,pspso.defaultparams,self.x_min,self.x_max,pspso.rounding,self.bounds, self.dimensions,pspso.paramdetails=pspso.read_parameters(params,self.estimator,self.task)

    @staticmethod
    def get_default_search_space(estimator,task):

        if estimator == 'xgboost':
            if task == 'binary classification':
                params = {"learning_rate":  [0.1,0.3,2],
                  #"max_depth": [1,10,0],
                  "n_estimators": [2,70,0],
                  #"subsample": [0.7,1,2],
                  }
            elif task == 'multi classification':
                params = {"learning_rate":  [0.1,0.3,2],
                  #"max_depth": [1,10,0],
                  "n_estimators": [2,70,0],
                  #"subsample": [0.7,1,2],
                  }      
            else:
                params = {"objective": ["reg:linear","reg:tweedie","reg:gamma"],
                  "learning_rate":  [0.1,0.3,2],
                  #"max_depth": [1,10,0],
                  "n_estimators": [2,70,0],
                  #"subsample": [0.7,1,2]
                  }
        return params
    
    @staticmethod
    def get_default_params(estimator, task):
        defaultparams= {}
        if estimator == 'xgboost':
            defaultparams.update({'learning_rate':0.01,'n_estimators':40,"verbosity":0,"use_label_encoder":False})
            if task =='binary classification': # default activation
                defaultparams.update({'objective':'binary:logistic','booster':'gblinear'})
            elif task =='multi classification': # default activation
                defaultparams.update({'objective':'multi:softmax','booster':'gblinear'})    
            elif task =='regression':
                defaultparams.update({'objective':'reg:tweedie','eval_metric':["rmse"]})    
        return defaultparams
        
    
    @staticmethod
    def read_parameters(params=None,estimator=None, task=None):
        if params == None:
            params=pspso.get_default_search_space(estimator,task)
        x_min,x_max,rounding,parameters=[],[],[],[]
        for key in params:
            if all(isinstance(item, str) for item in params[key]):
                of=params[key]
                x_min.append(0)
                x_max.append(len(of)-1)
                parameters.append(key)
                rounding.append(0)
            else:
                thelist=params[key]
                x_min.append(thelist[0])
                x_max.append(thelist[1])
                parameters.append(key)
                rounding.append(thelist[2])
        bounds = (np.asarray(x_min), np.asarray(x_max))   
        dimensions=len(x_min)
        defaultparams=pspso.get_default_params(estimator, task)                              
        return parameters,defaultparams, x_min,x_max,rounding,bounds, dimensions,params
    
    @staticmethod
    def decode_parameters(particle):
        decodeddict={}
        for d in range(0,len(particle)):
            key=pspso.parameters[d]
            particlevalueatd=particle[d]
            if all(isinstance(item, str) for item in pspso.paramdetails[key]):
                index=int(round(particlevalueatd))
                decodeddict[key] = pspso.paramdetails[key][index]
            else:
                decodeddict[key] =round(particlevalueatd,pspso.rounding[pspso.parameters.index(key)])
                if pspso.rounding[pspso.parameters.index(key)] == 0:
                    decodeddict[key]=int(decodeddict[key])
        return decodeddict
        
    @staticmethod
    def forward_prop_xgboost(particle,task,score,X_train,Y_train,X_val,Y_val,X_full,Y_full, XGBooster):       
        model=None 
        eval_set = [(X_val, Y_val)]
        try:
            decodedparams = pspso.decode_parameters(particle)
            modelparameters = {**pspso.defaultparams,**decodedparams}
            modelparameters['booster'] = XGBooster
            f = open("log.txt", "a")
            f.write(f'\nParameters: {modelparameters}')
            if task =='regression':
                model = xgb.XGBRegressor(**modelparameters)
            else :
                model = xgb.XGBClassifier(**modelparameters)
            model.fit(X_train,Y_train,early_stopping_rounds=pspso.early_stopping,eval_set=eval_set,verbose=pspso.verbose )
            pred = pspso.predict(model,'xgboost',task, score,X_val,Y_val,X_full, Y_full)
            f.write(f' - Accuracy: {1-pred} /')
            f.close()
            return pred,model
        except Exception as e:
            print('An exception occured in XGBoost training.')
            print(e)
            return None,None

    @staticmethod
    def f(q,estimator,task,score,X_train,Y_train,X_val,Y_val,X_full, Y_full, XGBooster):
        n_particles = q.shape[0]
        if estimator=='xgboost':
            e = [pspso.forward_prop_xgboost(q[i],task,score,X_train,Y_train,X_val,Y_val,X_full, Y_full, XGBooster) for i in range(n_particles)]
            j=[e[i][0] for i in range(n_particles)]
        return np.array(j) 

    @staticmethod
    def rebuildmodel(estimator,pos,task,score,X_train,Y_train,X_val,Y_val,X_full, Y_full, XGBooster):
      if estimator=='xgboost':
          met,model=pspso.forward_prop_xgboost(pos,task,score,X_train,Y_train,X_val,Y_val,X_full, Y_full, XGBooster)
      return met,model
  
    def fitpspso(self, X_train=None, Y_train=None, X_val=None,Y_val=None,X_full=None,Y_full=None,psotype='global',number_of_particles=1, number_of_iterations=10, options = {'c1':  1.49618, 'c2':  1.49618, 'w': 0.7298}, xgbooster='gbtree'):
        print("Running PSO Search .....")
        self.selectiontype= "PSO"
        self.number_of_particles=number_of_particles
        self.number_of_iterations=number_of_iterations
        self.psotype=psotype
        self.options=options
        self.number_of_attempts=self.number_of_iterations *self.number_of_particles
        self.totalnbofcombinations= self.calculatecombinations()
        pspso.best_paricle_cost_ann =None
        pspso.best_model_ann=None
        pspso.best_history_ann=None
        pspso.best_particle_position_ann=None
        
        kwargs = {"estimator":self.estimator, "task":self.task, "score":self.score, "X_train" : X_train, "Y_train" : Y_train, 
                  "X_val" : X_val,"Y_val":Y_val, 'X_full':X_full, 'Y_full':Y_full, 'XGBooster':xgbooster}
        if psotype =='global':
            self.optimizer = ps.single.GlobalBestPSO(n_particles=self.number_of_particles, dimensions=self.dimensions, options=self.options,bounds=self.bounds)
        elif psotype =='local':
            self.optimizer = ps.single.LocalBestPSO(n_particles=self.number_of_particles, dimensions=self.dimensions, options=self.options,bounds=self.bounds)
        start=time()
        self.cost, self.pos = self.optimizer.optimize(pspso.f, iters=self.number_of_iterations,**kwargs)
        end=time()
        self.duration=end-start          
        self.met,self.model=pspso.rebuildmodel(self.estimator,self.pos,self.task,self.score,X_train,Y_train,X_val,Y_val,X_full, Y_full, xgbooster)
        
        return self.pos,self.cost,self.duration,self.model,self.optimizer
    
    def print_results(self):
        print("Estimator: " + self.estimator)
        print("Task: "+ self.task)
        print("Selection type: "+ str(self.selectiontype))
        print("Number of attempts:" + str(self.number_of_attempts))
        print("Total number of combinations: " + str(self.totalnbofcombinations))
        print("Parameters:")
        print(pspso.decode_parameters(self.pos))
        print("Global best position: " + str(self.pos))
        print("Global best cost: " +str(round(self.cost,4)))
        print("Time taken to find the set of parameters: "+ str(self.duration))
        if self.selectiontype == "PSO":
            print("Number of particles: " +str(self.number_of_particles))
            print("Number of iterations: "+ str(self.number_of_iterations))

    def calculatecombinations(self):
        index=0
        zarb = 1
        thedict={}
        for i,j in zip(self.x_min,self.x_max):
            a=np.arange(i, j+0.000001, 10**(-1*self.rounding[index]))
            a=np.round(a,self.rounding[index])
            thedict[pspso.parameters[index]]=a
            index=index+1
        for kk in thedict.keys():
          zarb *= len(thedict[kk])
        return zarb
    
    @staticmethod
    def predict(model,estimator,task, score,X_val,Y_val,X_full,Y_full):
        if score=='rmse':
            preds_val=model.predict(X_val)
            met = np.sqrt(mean_squared_error(Y_val, preds_val))
            return met

        if task == 'binary classification':
            if score == 'acc':
                accc,_= KFold_cross_val_test(model=model, X=X_full, Y=Y_full, K=5, Report=True, print_report=False, shuffle=True, random_state=14)
                return 1-accc
            elif score == 'auc':
                preds_val = model.predict_proba(X_val)
                fpr, tpr, thresholds = roc_curve(Y_val, preds_val[:,1]) 
                met = auc(fpr, tpr)
                return 1-met
        if task == "multi classification":
            if score == 'acc':
                preds_val = model.predict(X_val)
                met = accuracy_score(Y_val,preds_val)
                print(f'\nAccuracy: {met}\n')
                return 1-met
            elif score == 'auc':
                preds_val = model.predict_proba(X_val)
                fpr, tpr, thresholds = roc_curve(Y_val, preds_val[:,1]) 
                met = auc(fpr, tpr)
                return 1-met

    def save_optimizer_details(self):
        opt={}
        opt['pos_history']=self.optimizer.pos_history
        opt['cost_history']=self.optimizer.cost_history
        opt['bounds']=self.optimizer.bounds
        opt['init_pos']=self.optimizer.init_pos
        opt['swarm_size']=self.optimizer.swarm_size
        opt['options']=self.optimizer.options
        opt['name']=self.optimizer.name
        opt['n_particles']=self.optimizer.n_particles
        return opt