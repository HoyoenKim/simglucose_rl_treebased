<h2>Reference</h2>
- https://www.kaggle.com/competitions/brist1d/discussion/555236
- https://github.com/scuya2050/brist1d_blood_glucose_prediction_competition
- https://github.com/jxx123/simglucose

<h2>Settings</h2>
<h3>1. Enviroment setup</h3>

    $ pip install -r requirements.txt

<h3>2. Preprocess kaggle data & train lightgbm model</h3>
- follow data setup same as here:
- https://github.com/scuya2050/brist1d_blood_glucose_prediction_competition

<h3>3. Train RL model</h3>

    $ python simglucose_rl_try.py 

<h4> 3.1. Check the tensorboard </h4>

    $ tensorboard --logdir ./logs
 
- http://localhost:6006/#timeseries