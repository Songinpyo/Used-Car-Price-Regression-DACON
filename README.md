# Used-Car-Price-Regression-DACON
Predict used car price with car information. This is competition held in DACON

Competition link : You can get Dataset from this link

https://dacon.io/competitions/official/235901/overview/description

![image](https://user-images.githubusercontent.com/104220612/172037814-4ace4b82-870f-404a-9c50-fce86235d1bf.png)

### Key Idea
Preprocessing key idea was transforming 'object' data to 'int' data.

Predict 'target' data with **'Pytorch DNN'** and **'Pycaret Auto ML'**

Tuning hyper parameter with **'Optuna'**

### Data set
Train data has 'title', 'odometer', 'location', 'isimported', 'engine', 'paint', 'year', 'target' columns

Test data has 'title', 'odometer', 'location', 'isimported', 'engine', 'paint', 'year' columns.


### Code example
```python
models = [
    ('LGBMRegressor',LGBMRegressor(n_estimators=800,max_depth=13, learning_rate=0.01,subsample=0.7,random_state=0)),
    ('XGBRegressor',XGBRegressor(n_estimators=1600,max_depth=13,learning_rate=0.02,random_state=0,subsample=0.7,verbosity=0)),
    ('CatBoostRegressor',CatBoostRegressor(n_estimators=800,max_depth=10,learning_rate=0.02,subsample=0.8,random_state=0,verbose=0)),
    ('GradientBoostingRegressor',GradientBoostingRegressor(n_estimators=1200,max_depth=9,learning_rate=0.01,subsample=0.8,random_state=0)),
    ('ExtraTreesRegressor',ExtraTreesRegressor(n_estimators=500,max_depth=9,random_state=0,criterion='mse')),
    ('RandomForestRegressor',RandomForestRegressor(n_estimators=4000,max_depth=9,random_state=0,criterion='mse'))]
    
for name, model in models:
    model.fit(x_train, y_train)
    print(f'{name}: ', NMAE(y_val,model.predict(x_val)))
```

### Future work
It could be better to change 'odometer' int data into categorical data.

Example) from 0 to 10000 = 0, from 10000 to 20000 = 1 ...
