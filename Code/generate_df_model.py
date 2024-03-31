import numpy as np

####### MODEL #####################################

size = 10000 # number of instances
law = 'normal' # law for the distribution of the features
num_model = 0 # set a number associated to the model to repair it in files

# define your model

def model(x):

    return np.sin(2 * np.pi * x[0]) * np.sin(np.pi * (x[1] + x[2])) + x[3] + x[4]

##################################################
import pandas as pd


def new_csv(model=model, size=size, law=law, num_model=num_model):
    # couting number of parameters

    nb_of_parameters = 1
    boo = True
    while boo:
        try:
            model([0 for _ in range(nb_of_parameters)])
            boo = False
        except:
            nb_of_parameters += 1

    # create a random matrix of features

    if law == 'normal':
        random_matrix = np.random.normal(0, 1, size=(size, nb_of_parameters))

    elif law == 'student':
        random_matrix = np.random.standard_t(df=2, size=(size, nb_of_parameters))

    # create the dataframe

    df = pd.DataFrame(random_matrix, columns=['x{}'.format(i) for i in range(nb_of_parameters)]) 
    df['f'] = df.apply(model, axis=1)
    
    # save the dataframe in a csv file
    
    z = np.random.randint(0, 100000)
    name = "dataframes/df_model{}_".format(num_model) + law + "_{}.csv".format(z)
    df.to_csv(name, index=False)

    return name


if __name__ == '__main__':

    new_csv(model=model, size=size, law=law, num_model=num_model)
