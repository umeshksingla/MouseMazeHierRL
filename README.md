# MouseMaze
 
Original code and raw data from [Rosenberg-2021-Repository](https://github.com/markusmeister/Rosenberg-2021-Repository)

Google notebook link [here](https://docs.google.com/document/d/1FG4x-Lj7eFH-U-M5xrcb6oEvnsBYPWCYeSBzgtgq2p8/edit?usp=sharing).


## Steps to implement a model 
You can follow the below steps for `model` = `"TDlambdaXSteps_"`:

* Create a directory inside the root of the repo, say `<model>stan`.
* Stan File: Add your stan model file with name `<model>.stan` to this directory.
* Python model class file: Add a model class with name `<model>model.py`.

## To train model parameters in stan:
* Add the python code to initialize your model class, call its various 
functions such as extract traj data, get model configuration like number of states, 
etc as relevant and call the stan script. Also, save your stan results. 
Refer to `<model>stan/<model>run_model.py` for an example.
* Add the bash script to run your job on ssrde server. Initialize
any params you need and call the python job basically. Pass the various
configuration options like directory paths, log output etc here.
Refer to `submit_stan_fit` in `<model>stan` folder.
Use this command to run the model: 
```bash
$ submit_stan_fit <model>run_model.py
```
    
## To simulate:
* To simulate an agent based on the fitted parameters you got in train step,
simply import the model class and run your simulate function.
Refer to `<model>simulate.py` for an example.
* Otherwise, you can run a simple TDLambda agent with arbitrarily set parameters 
  with code exemplified in the `__main__`  of `TDLambda_model.py`
