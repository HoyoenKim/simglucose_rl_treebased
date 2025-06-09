<h2>Reference</h2>

- https://www.kaggle.com/competitions/brist1d/discussion/555236
- https://github.com/scuya2050/brist1d_blood_glucose_prediction_competition
- https://github.com/jxx123/simglucose

<h2>Settings</h2>
<h3>1. Enviroment setup</h3>

    $ pip install -r requirements.txt

<h3>2. Preprocess Kaggle Dataset & Train LightGBM model</h3>

- Set up Kaggle Dataset and train LightGBM model details follow here:
- https://github.com/scuya2050/brist1d_blood_glucose_prediction_competition

<h4> 2.1. Preprocess Kaggle Dataset </h4>
    
    $ python prepare_data.py 

<h4> 2.2. Train LightGBM Model </h4>

    $ python train.py 

<h4> 2.3. Prepare LightGBM Model </h4>

- Place LightGBM model at ./lgbm_model.pkl

<h3>3. Train RL model</h3>

    $ python simglucose_rl_try.py 

<h4> 3.1. Check the tensorboard </h4>

    $ tensorboard --logdir ./logs
 
- http://localhost:6006/#timeseries

<h3>4. Evaluate RL model</h3>

    $ python simglucose_rl_ev.py

- eval_result_bg.png
- eval_result_ins.png
- terminal output
> Eval 20 eps: -275.31 ± 0.00 <br>
> Time-in-Range (%): 67.93 <br>
> LBGI (mean): 28.39 <br>
> HBGI (mean): 4.61 <br>

<h3>5. See the details of our study</h3>

- Result of our works at /Result
- Details of our works at /doc