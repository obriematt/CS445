# Matthew O'Brien
# CS 445 Homework 1
# Perceptron All Pairs


# This import was only included once to allow for the creating of the data files that were included
import Data_Parsing as parser
import numpy as np
import random
learning_rate = 0.2

#Perceptron class that will be trained for the data
class Perceptron:

    #Initializing data. Values correspond to letters. IE A v B perceptron is value1 = 1 and value2 = 2
    def __init__(self, i, j):
        self.value1 = i
        self.value2 = j
        self.weights = self.randomweights()
        self.bias = random.uniform(-1,1)
        self.accuracy = 0.0

    # Function to get random weights from -1 to 1.
    def randomweights(self):
        weights = []
        for i in range(16):
            weights.append(random.uniform(-1,1))
        return weights

    #Trains individual perceptron. Designed for repetition on each set.
    def train_me(self, training_set):

        #Sets a delta that is greater than initial accuracy
        delta_accuracy = 1

        #Repeats until little change has happened in accuracy. Used to combat overtraining
        while delta_accuracy > .00002:
            counter = 0
            previous_accuracy = self.accuracy
            #Goes through the training set
            for i in range(len(training_set)):


                #Find out what letter we are looking for 1(A) - 26(Z)
                #This trend is used later for testing as well.
                expected_letter = training_set[i,0]
                #Value1 is a positive output
                if expected_letter == self.value1:
                    test_number = 1
                #Value2 is a negative output
                if expected_letter == self.value2:
                    test_number = -1

                #Get the row of inputs for that expected letter.
                input_value = training_set[i,1:]

                #Get the output of that row with the sgn function, -1 or 1
                output = np.dot(self.weights,input_value) + self.bias
                output = sgn_function(output)

                #If the output is not the same as the expected letter. Change the weights
                if output != test_number:
                    self.fix_me(test_number,input_value)

                #Else the output is right, increase the counter for accuracy measurement
                if output == test_number:
                    counter = counter + 1

            #Calculate accuracy and delta for repetition or not.
            self.accuracy = counter / (len(training_set)+1)
            delta_accuracy = abs(previous_accuracy-self.accuracy)

    #Used for testing to see what letter each perceptron thinks it's looking at.
    def letter_check(self, test_set):

        #Get the array of attributes
        input_value = test_set[1:]

        #Calculate the output of these with weights, values and bias.
        output = np.dot(self.weights, input_value) + self.bias
        output = sgn_function(output)

        #Match output to one of two letter choices.
        #Again, positive output is value1 and negative for value2
        if output == 1:
            return self.value1
        if output == -1:
            return self.value2

    #Readjust the weights
    def fix_me(self,expected_letter,input_value):

        #Adjustment function.
        for i in range(16):
            self.weights[i] = self.weights[i] + (learning_rate*expected_letter*input_value[i])
        self.bias = self.bias + learning_rate*expected_letter*1

# Function to run the test data
def test_perceptron_data(test_data, perceptron_list):


    #Empty list of letter responses from perceptrons
    which_letter = []

    #Trains through the perceptron list returning letter "Votes"
    for i in perceptron_list:
        which_letter.append(i.letter_check(test_data))
    #Finds and returns the most common integer(corresponding letter)
    counts = np.bincount(which_letter)
    return np.argmax(counts)


#Creates the array of 325 perceptrons with values to determine which is which
def create_perceptron_array():
    perceptron_list = []
    for i in range(1,27):
        for j in range(i,27):
            #Don't have to have an A v A perceptron and so on.
            if i != j:
                perceptron_list.append(Perceptron(i,j))
    return perceptron_list



# Creates an array of the data file for use in perceptrons
def make_training_data(textfile):

    #Formatting for the array from the text files
    temp_a_data = np.genfromtxt(textfile, delimiter=',', dtype = 'O')
    #Loop to traverse the file
    for i in range(len(temp_a_data)):
        #Converts the first letter of each line into an integer. Required for the numpy array
        temp_a_data[i,0] = ord(temp_a_data[i,0].lower()) - 96.
    #Makes the data numpy floats
    array_a_data = temp_a_data.astype(np.float32)
    #Divide all the attributes by 15, as it is the largest value.
    array_a_data[:,1:] = array_a_data[:, 1:] / float(15.0)
    #Gives back the new numpy array of floats.
    return array_a_data

# Shuffles two training arrays for testing
def shuffle_training_data(a, b):

    #Makes the two arrays of selected training data. All A's and all B's data for example.
    array1 = make_training_data(a)
    array2 = make_training_data(b)

    #Concatenates them into one array.
    array3 = np.concatenate((array1,array2), axis=0)

    #Shuffles the array. Interwoven data is better for training.
    np.random.shuffle(array3)

    #Returns new shuffled array.
    return array3


# Training a perceptron based on it's training set and corresponding letter
def training_perceptrons(perceptron_list,training_set, letter_to_check1, letter_to_check2):

    #Converts the two letters into numbers to match the arrays.
    to_check = ord(letter_to_check1.lower()) - 96.
    to_check2 = ord(letter_to_check2.lower()) - 96.

    #Mostly used for testing. To make sure the perceptrons and data were matching up.
    x = len(perceptron_list)
    list_of_perceptrons = []

    #Used for testing. Making sure the perceptrons were correct.
    for i in range(x):

        if (perceptron_list[i].value1 == to_check or perceptron_list[i].value2 == to_check) and (perceptron_list[i].value1 == to_check2 or perceptron_list[i].value2 == to_check2):
            list_of_perceptrons.append(i)

    #Goes through the perceptron list with the data sets and trains.
    for i in perceptron_list:

        #Used to make sure the AvB perceptron only sees the A and B data.
        if (i.value1 == to_check or i.value2 == to_check) and (i.value1 == to_check2 or i.value2 == to_check2):
            i.train_me(training_set)


    return perceptron_list



#SGN function for output
def sgn_function(x):
    if x < 0:
        return -1
    if x >= 0:
        return 1


#This function is ugly. And should have been optimized.
#Ran out of time, used the ugly format because it worked.
def train_all_perceptrons(perceptrons):


    # Creating testing data for perceptrons
    # This is horrific code.
    ab = shuffle_training_data("A_testing_Data.txt","B_testing_Data.txt")
    ac = shuffle_training_data("A_testing_Data.txt", "C_testing_Data.txt")
    ad = shuffle_training_data("A_testing_Data.txt", "D_testing_Data.txt")
    ae = shuffle_training_data("A_testing_Data.txt", "E_testing_Data.txt")
    af = shuffle_training_data("A_testing_Data.txt", "F_testing_Data.txt")
    ag = shuffle_training_data("A_testing_Data.txt", "G_testing_Data.txt")
    ah = shuffle_training_data("A_testing_Data.txt", "H_testing_Data.txt")
    ai = shuffle_training_data("A_testing_Data.txt", "I_testing_Data.txt")
    aj = shuffle_training_data("A_testing_Data.txt", "J_testing_Data.txt")
    ak = shuffle_training_data("A_testing_Data.txt", "K_testing_Data.txt")
    al = shuffle_training_data("A_testing_Data.txt", "L_testing_Data.txt")
    am = shuffle_training_data("A_testing_Data.txt", "M_testing_Data.txt")
    an = shuffle_training_data("A_testing_Data.txt", "N_testing_Data.txt")
    ao = shuffle_training_data("A_testing_Data.txt", "O_testing_Data.txt")
    ap = shuffle_training_data("A_testing_Data.txt", "P_testing_Data.txt")
    aq = shuffle_training_data("A_testing_Data.txt", "Q_testing_Data.txt")
    ar = shuffle_training_data("A_testing_Data.txt", "R_testing_Data.txt")
    as_ = shuffle_training_data("A_testing_Data.txt", "S_testing_Data.txt")
    at = shuffle_training_data("A_testing_Data.txt", "T_testing_Data.txt")
    au = shuffle_training_data("A_testing_Data.txt", "U_testing_Data.txt")
    av = shuffle_training_data("A_testing_Data.txt", "V_testing_Data.txt")
    aw = shuffle_training_data("A_testing_Data.txt", "W_testing_Data.txt")
    ax = shuffle_training_data("A_testing_Data.txt", "X_testing_Data.txt")
    ay = shuffle_training_data("A_testing_Data.txt", "Y_testing_Data.txt")
    az = shuffle_training_data("A_testing_Data.txt", "Z_testing_Data.txt")
    bc = shuffle_training_data("B_testing_Data.txt", "C_testing_Data.txt")
    bd = shuffle_training_data("B_testing_Data.txt", "D_testing_Data.txt")
    be = shuffle_training_data("B_testing_Data.txt", "E_testing_Data.txt")
    bf = shuffle_training_data("B_testing_Data.txt", "F_testing_Data.txt")
    bg = shuffle_training_data("B_testing_Data.txt", "G_testing_Data.txt")
    bh = shuffle_training_data("B_testing_Data.txt", "H_testing_Data.txt")
    bi = shuffle_training_data("B_testing_Data.txt", "I_testing_Data.txt")
    bj = shuffle_training_data("B_testing_Data.txt", "J_testing_Data.txt")
    bk = shuffle_training_data("B_testing_Data.txt", "K_testing_Data.txt")
    bl = shuffle_training_data("B_testing_Data.txt", "L_testing_Data.txt")
    bm = shuffle_training_data("B_testing_Data.txt", "M_testing_Data.txt")
    bn = shuffle_training_data("B_testing_Data.txt", "N_testing_Data.txt")
    bo = shuffle_training_data("B_testing_Data.txt", "O_testing_Data.txt")
    bp = shuffle_training_data("B_testing_Data.txt", "P_testing_Data.txt")
    bq = shuffle_training_data("B_testing_Data.txt", "Q_testing_Data.txt")
    br = shuffle_training_data("B_testing_Data.txt", "R_testing_Data.txt")
    bs = shuffle_training_data("B_testing_Data.txt", "S_testing_Data.txt")
    bt = shuffle_training_data("B_testing_Data.txt", "T_testing_Data.txt")
    bu = shuffle_training_data("B_testing_Data.txt", "U_testing_Data.txt")
    bv = shuffle_training_data("B_testing_Data.txt", "V_testing_Data.txt")
    bw = shuffle_training_data("B_testing_Data.txt", "W_testing_Data.txt")
    bx = shuffle_training_data("B_testing_Data.txt", "X_testing_Data.txt")
    by = shuffle_training_data("B_testing_Data.txt", "Y_testing_Data.txt")
    bz = shuffle_training_data("B_testing_Data.txt", "Z_testing_Data.txt")
    cd = shuffle_training_data("C_testing_Data.txt", "D_testing_Data.txt")
    ce = shuffle_training_data("C_testing_Data.txt", "E_testing_Data.txt")
    cf = shuffle_training_data("C_testing_Data.txt", "F_testing_Data.txt")
    cg = shuffle_training_data("C_testing_Data.txt", "G_testing_Data.txt")
    ch = shuffle_training_data("C_testing_Data.txt", "H_testing_Data.txt")
    ci = shuffle_training_data("C_testing_Data.txt", "I_testing_Data.txt")
    cj = shuffle_training_data("C_testing_Data.txt", "J_testing_Data.txt")
    ck = shuffle_training_data("C_testing_Data.txt", "K_testing_Data.txt")
    cl = shuffle_training_data("C_testing_Data.txt", "L_testing_Data.txt")
    cm = shuffle_training_data("C_testing_Data.txt", "M_testing_Data.txt")
    cn = shuffle_training_data("C_testing_Data.txt", "N_testing_Data.txt")
    co = shuffle_training_data("C_testing_Data.txt", "O_testing_Data.txt")
    cp = shuffle_training_data("C_testing_Data.txt", "P_testing_Data.txt")
    cq = shuffle_training_data("C_testing_Data.txt", "Q_testing_Data.txt")
    cr = shuffle_training_data("C_testing_Data.txt", "R_testing_Data.txt")
    cs = shuffle_training_data("C_testing_Data.txt", "S_testing_Data.txt")
    ct = shuffle_training_data("C_testing_Data.txt", "T_testing_Data.txt")
    cu = shuffle_training_data("C_testing_Data.txt", "U_testing_Data.txt")
    cv = shuffle_training_data("C_testing_Data.txt", "V_testing_Data.txt")
    cw = shuffle_training_data("C_testing_Data.txt", "W_testing_Data.txt")
    cx = shuffle_training_data("C_testing_Data.txt", "X_testing_Data.txt")
    cy = shuffle_training_data("C_testing_Data.txt", "Y_testing_Data.txt")
    cz = shuffle_training_data("C_testing_Data.txt", "Z_testing_Data.txt")
    de = shuffle_training_data("D_testing_Data.txt", "E_testing_Data.txt")
    df = shuffle_training_data("D_testing_Data.txt", "F_testing_Data.txt")
    dg = shuffle_training_data("D_testing_Data.txt", "G_testing_Data.txt")
    dh = shuffle_training_data("D_testing_Data.txt", "H_testing_Data.txt")
    di = shuffle_training_data("D_testing_Data.txt", "I_testing_Data.txt")
    dj = shuffle_training_data("D_testing_Data.txt", "J_testing_Data.txt")
    dk = shuffle_training_data("D_testing_Data.txt", "K_testing_Data.txt")
    dl = shuffle_training_data("D_testing_Data.txt", "L_testing_Data.txt")
    dm = shuffle_training_data("D_testing_Data.txt", "M_testing_Data.txt")
    dn = shuffle_training_data("D_testing_Data.txt", "N_testing_Data.txt")
    do = shuffle_training_data("D_testing_Data.txt", "O_testing_Data.txt")
    dp = shuffle_training_data("D_testing_Data.txt", "P_testing_Data.txt")
    dq = shuffle_training_data("D_testing_Data.txt", "Q_testing_Data.txt")
    dr = shuffle_training_data("D_testing_Data.txt", "R_testing_Data.txt")
    ds = shuffle_training_data("D_testing_Data.txt", "S_testing_Data.txt")
    dt = shuffle_training_data("D_testing_Data.txt", "T_testing_Data.txt")
    du = shuffle_training_data("D_testing_Data.txt", "U_testing_Data.txt")
    dv = shuffle_training_data("D_testing_Data.txt", "V_testing_Data.txt")
    dw = shuffle_training_data("D_testing_Data.txt", "W_testing_Data.txt")
    dx = shuffle_training_data("D_testing_Data.txt", "X_testing_Data.txt")
    dy = shuffle_training_data("D_testing_Data.txt", "Y_testing_Data.txt")
    dz = shuffle_training_data("D_testing_Data.txt", "Z_testing_Data.txt")
    ef = shuffle_training_data("E_testing_Data.txt", "F_testing_Data.txt")
    eg = shuffle_training_data("E_testing_Data.txt", "G_testing_Data.txt")
    eh = shuffle_training_data("E_testing_Data.txt", "H_testing_Data.txt")
    ei = shuffle_training_data("E_testing_Data.txt", "I_testing_Data.txt")
    ej = shuffle_training_data("E_testing_Data.txt", "J_testing_Data.txt")
    ek = shuffle_training_data("E_testing_Data.txt", "K_testing_Data.txt")
    el = shuffle_training_data("E_testing_Data.txt", "L_testing_Data.txt")
    em = shuffle_training_data("E_testing_Data.txt", "M_testing_Data.txt")
    en = shuffle_training_data("E_testing_Data.txt", "N_testing_Data.txt")
    eo = shuffle_training_data("E_testing_Data.txt", "O_testing_Data.txt")
    ep = shuffle_training_data("E_testing_Data.txt", "P_testing_Data.txt")
    eq = shuffle_training_data("E_testing_Data.txt", "Q_testing_Data.txt")
    er = shuffle_training_data("E_testing_Data.txt", "R_testing_Data.txt")
    es = shuffle_training_data("E_testing_Data.txt", "S_testing_Data.txt")
    et = shuffle_training_data("E_testing_Data.txt", "T_testing_Data.txt")
    eu = shuffle_training_data("E_testing_Data.txt", "U_testing_Data.txt")
    ev = shuffle_training_data("E_testing_Data.txt", "V_testing_Data.txt")
    ew = shuffle_training_data("E_testing_Data.txt", "W_testing_Data.txt")
    ex = shuffle_training_data("E_testing_Data.txt", "X_testing_Data.txt")
    ey = shuffle_training_data("E_testing_Data.txt", "Y_testing_Data.txt")
    ez = shuffle_training_data("E_testing_Data.txt", "Z_testing_Data.txt")
    fg = shuffle_training_data("F_testing_Data.txt", "G_testing_Data.txt")
    fh = shuffle_training_data("F_testing_Data.txt", "H_testing_Data.txt")
    fi = shuffle_training_data("F_testing_Data.txt", "I_testing_Data.txt")
    fj = shuffle_training_data("F_testing_Data.txt", "J_testing_Data.txt")
    fk = shuffle_training_data("F_testing_Data.txt", "K_testing_Data.txt")
    fl = shuffle_training_data("F_testing_Data.txt", "L_testing_Data.txt")
    fm = shuffle_training_data("F_testing_Data.txt", "M_testing_Data.txt")
    fn = shuffle_training_data("F_testing_Data.txt", "N_testing_Data.txt")
    fo = shuffle_training_data("F_testing_Data.txt", "O_testing_Data.txt")
    fp = shuffle_training_data("F_testing_Data.txt", "P_testing_Data.txt")
    fq = shuffle_training_data("F_testing_Data.txt", "Q_testing_Data.txt")
    fr = shuffle_training_data("F_testing_Data.txt", "R_testing_Data.txt")
    fs = shuffle_training_data("F_testing_Data.txt", "S_testing_Data.txt")
    ft = shuffle_training_data("F_testing_Data.txt", "T_testing_Data.txt")
    fu = shuffle_training_data("F_testing_Data.txt", "U_testing_Data.txt")
    fv = shuffle_training_data("F_testing_Data.txt", "V_testing_Data.txt")
    fw = shuffle_training_data("F_testing_Data.txt", "W_testing_Data.txt")
    fx = shuffle_training_data("F_testing_Data.txt", "X_testing_Data.txt")
    fy = shuffle_training_data("F_testing_Data.txt", "Y_testing_Data.txt")
    fz = shuffle_training_data("F_testing_Data.txt", "Z_testing_Data.txt")
    gh = shuffle_training_data("G_testing_Data.txt", "H_testing_Data.txt")
    gi = shuffle_training_data("G_testing_Data.txt", "I_testing_Data.txt")
    gj = shuffle_training_data("G_testing_Data.txt", "J_testing_Data.txt")
    gk = shuffle_training_data("G_testing_Data.txt", "K_testing_Data.txt")
    gl = shuffle_training_data("G_testing_Data.txt", "L_testing_Data.txt")
    gm = shuffle_training_data("G_testing_Data.txt", "M_testing_Data.txt")
    gn = shuffle_training_data("G_testing_Data.txt", "N_testing_Data.txt")
    go = shuffle_training_data("G_testing_Data.txt", "O_testing_Data.txt")
    gp = shuffle_training_data("G_testing_Data.txt", "P_testing_Data.txt")
    gq = shuffle_training_data("G_testing_Data.txt", "Q_testing_Data.txt")
    gr = shuffle_training_data("G_testing_Data.txt", "R_testing_Data.txt")
    gs = shuffle_training_data("G_testing_Data.txt", "S_testing_Data.txt")
    gt = shuffle_training_data("G_testing_Data.txt", "T_testing_Data.txt")
    gu = shuffle_training_data("G_testing_Data.txt", "U_testing_Data.txt")
    gv = shuffle_training_data("G_testing_Data.txt", "V_testing_Data.txt")
    gw = shuffle_training_data("G_testing_Data.txt", "W_testing_Data.txt")
    gx = shuffle_training_data("G_testing_Data.txt", "X_testing_Data.txt")
    gy = shuffle_training_data("G_testing_Data.txt", "Y_testing_Data.txt")
    gz = shuffle_training_data("G_testing_Data.txt", "Z_testing_Data.txt")
    hi = shuffle_training_data("H_testing_Data.txt", "I_testing_Data.txt")
    hj = shuffle_training_data("H_testing_Data.txt", "J_testing_Data.txt")
    hk = shuffle_training_data("H_testing_Data.txt", "K_testing_Data.txt")
    hl = shuffle_training_data("H_testing_Data.txt", "L_testing_Data.txt")
    hm = shuffle_training_data("H_testing_Data.txt", "M_testing_Data.txt")
    hn = shuffle_training_data("H_testing_Data.txt", "N_testing_Data.txt")
    ho = shuffle_training_data("H_testing_Data.txt", "O_testing_Data.txt")
    hp = shuffle_training_data("H_testing_Data.txt", "P_testing_Data.txt")
    hq = shuffle_training_data("H_testing_Data.txt", "Q_testing_Data.txt")
    hr = shuffle_training_data("H_testing_Data.txt", "R_testing_Data.txt")
    hs = shuffle_training_data("H_testing_Data.txt", "S_testing_Data.txt")
    ht = shuffle_training_data("H_testing_Data.txt", "T_testing_Data.txt")
    hu = shuffle_training_data("H_testing_Data.txt", "U_testing_Data.txt")
    hv = shuffle_training_data("H_testing_Data.txt", "V_testing_Data.txt")
    hw = shuffle_training_data("H_testing_Data.txt", "W_testing_Data.txt")
    hx = shuffle_training_data("H_testing_Data.txt", "X_testing_Data.txt")
    hy = shuffle_training_data("H_testing_Data.txt", "Y_testing_Data.txt")
    hz = shuffle_training_data("H_testing_Data.txt", "Z_testing_Data.txt")
    ij = shuffle_training_data("I_testing_Data.txt", "J_testing_Data.txt")
    ik = shuffle_training_data("I_testing_Data.txt", "K_testing_Data.txt")
    il = shuffle_training_data("I_testing_Data.txt", "L_testing_Data.txt")
    im = shuffle_training_data("I_testing_Data.txt", "M_testing_Data.txt")
    ins_ = shuffle_training_data("I_testing_Data.txt", "N_testing_Data.txt")
    io = shuffle_training_data("I_testing_Data.txt", "O_testing_Data.txt")
    ip = shuffle_training_data("I_testing_Data.txt", "P_testing_Data.txt")
    iq = shuffle_training_data("I_testing_Data.txt", "Q_testing_Data.txt")
    ir = shuffle_training_data("I_testing_Data.txt", "R_testing_Data.txt")
    is_ = shuffle_training_data("I_testing_Data.txt", "S_testing_Data.txt")
    it = shuffle_training_data("I_testing_Data.txt", "T_testing_Data.txt")
    iu = shuffle_training_data("I_testing_Data.txt", "U_testing_Data.txt")
    iv = shuffle_training_data("I_testing_Data.txt", "V_testing_Data.txt")
    iw = shuffle_training_data("I_testing_Data.txt", "W_testing_Data.txt")
    ix = shuffle_training_data("I_testing_Data.txt", "X_testing_Data.txt")
    iy = shuffle_training_data("I_testing_Data.txt", "Y_testing_Data.txt")
    iz = shuffle_training_data("I_testing_Data.txt", "Z_testing_Data.txt")
    jk = shuffle_training_data("J_testing_Data.txt", "K_testing_Data.txt")
    jl = shuffle_training_data("J_testing_Data.txt", "L_testing_Data.txt")
    jm = shuffle_training_data("J_testing_Data.txt", "M_testing_Data.txt")
    jn = shuffle_training_data("J_testing_Data.txt", "N_testing_Data.txt")
    jo = shuffle_training_data("J_testing_Data.txt", "O_testing_Data.txt")
    jp = shuffle_training_data("J_testing_Data.txt", "P_testing_Data.txt")
    jq = shuffle_training_data("J_testing_Data.txt", "Q_testing_Data.txt")
    jr = shuffle_training_data("J_testing_Data.txt", "R_testing_Data.txt")
    js = shuffle_training_data("J_testing_Data.txt", "S_testing_Data.txt")
    jt = shuffle_training_data("J_testing_Data.txt", "T_testing_Data.txt")
    ju = shuffle_training_data("J_testing_Data.txt", "U_testing_Data.txt")
    jv = shuffle_training_data("J_testing_Data.txt", "V_testing_Data.txt")
    jw = shuffle_training_data("J_testing_Data.txt", "W_testing_Data.txt")
    jx = shuffle_training_data("J_testing_Data.txt", "X_testing_Data.txt")
    jy = shuffle_training_data("J_testing_Data.txt", "Y_testing_Data.txt")
    jz = shuffle_training_data("J_testing_Data.txt", "Z_testing_Data.txt")
    kl = shuffle_training_data("K_testing_Data.txt", "L_testing_Data.txt")
    km = shuffle_training_data("K_testing_Data.txt", "M_testing_Data.txt")
    kn = shuffle_training_data("K_testing_Data.txt", "N_testing_Data.txt")
    ko = shuffle_training_data("K_testing_Data.txt", "O_testing_Data.txt")
    kp = shuffle_training_data("K_testing_Data.txt", "P_testing_Data.txt")
    kq = shuffle_training_data("K_testing_Data.txt", "Q_testing_Data.txt")
    kr = shuffle_training_data("K_testing_Data.txt", "R_testing_Data.txt")
    ks = shuffle_training_data("K_testing_Data.txt", "S_testing_Data.txt")
    kt = shuffle_training_data("K_testing_Data.txt", "T_testing_Data.txt")
    ku = shuffle_training_data("K_testing_Data.txt", "U_testing_Data.txt")
    kv = shuffle_training_data("K_testing_Data.txt", "V_testing_Data.txt")
    kw = shuffle_training_data("K_testing_Data.txt", "W_testing_Data.txt")
    kx = shuffle_training_data("K_testing_Data.txt", "X_testing_Data.txt")
    ky = shuffle_training_data("K_testing_Data.txt", "Y_testing_Data.txt")
    kz = shuffle_training_data("K_testing_Data.txt", "Z_testing_Data.txt")
    lm = shuffle_training_data("L_testing_Data.txt", "M_testing_Data.txt")
    ln = shuffle_training_data("L_testing_Data.txt", "N_testing_Data.txt")
    lo = shuffle_training_data("L_testing_Data.txt", "O_testing_Data.txt")
    lp = shuffle_training_data("L_testing_Data.txt", "P_testing_Data.txt")
    lq = shuffle_training_data("L_testing_Data.txt", "Q_testing_Data.txt")
    lr = shuffle_training_data("L_testing_Data.txt", "R_testing_Data.txt")
    ls = shuffle_training_data("L_testing_Data.txt", "S_testing_Data.txt")
    lt = shuffle_training_data("L_testing_Data.txt", "T_testing_Data.txt")
    lu = shuffle_training_data("L_testing_Data.txt", "U_testing_Data.txt")
    lv = shuffle_training_data("L_testing_Data.txt", "V_testing_Data.txt")
    lw = shuffle_training_data("L_testing_Data.txt", "W_testing_Data.txt")
    lx = shuffle_training_data("L_testing_Data.txt", "X_testing_Data.txt")
    ly = shuffle_training_data("L_testing_Data.txt", "Y_testing_Data.txt")
    lz = shuffle_training_data("L_testing_Data.txt", "Z_testing_Data.txt")
    mn = shuffle_training_data("M_testing_Data.txt", "N_testing_Data.txt")
    mo = shuffle_training_data("M_testing_Data.txt", "O_testing_Data.txt")
    mp = shuffle_training_data("M_testing_Data.txt", "P_testing_Data.txt")
    mq = shuffle_training_data("M_testing_Data.txt", "Q_testing_Data.txt")
    mr = shuffle_training_data("M_testing_Data.txt", "R_testing_Data.txt")
    ms = shuffle_training_data("M_testing_Data.txt", "S_testing_Data.txt")
    mt = shuffle_training_data("M_testing_Data.txt", "T_testing_Data.txt")
    mu = shuffle_training_data("M_testing_Data.txt", "U_testing_Data.txt")
    mv = shuffle_training_data("M_testing_Data.txt", "V_testing_Data.txt")
    mw = shuffle_training_data("M_testing_Data.txt", "W_testing_Data.txt")
    mx = shuffle_training_data("M_testing_Data.txt", "X_testing_Data.txt")
    my = shuffle_training_data("M_testing_Data.txt", "Y_testing_Data.txt")
    mz = shuffle_training_data("M_testing_Data.txt", "Z_testing_Data.txt")
    no = shuffle_training_data("N_testing_Data.txt", "O_testing_Data.txt")
    np = shuffle_training_data("N_testing_Data.txt", "P_testing_Data.txt")
    nq = shuffle_training_data("N_testing_Data.txt", "Q_testing_Data.txt")
    nr = shuffle_training_data("N_testing_Data.txt", "R_testing_Data.txt")
    ns = shuffle_training_data("N_testing_Data.txt", "S_testing_Data.txt")
    nt = shuffle_training_data("N_testing_Data.txt", "T_testing_Data.txt")
    nu = shuffle_training_data("N_testing_Data.txt", "U_testing_Data.txt")
    nv = shuffle_training_data("N_testing_Data.txt", "V_testing_Data.txt")
    nw = shuffle_training_data("N_testing_Data.txt", "W_testing_Data.txt")
    nx = shuffle_training_data("N_testing_Data.txt", "X_testing_Data.txt")
    ny = shuffle_training_data("N_testing_Data.txt", "Y_testing_Data.txt")
    nz = shuffle_training_data("N_testing_Data.txt", "Z_testing_Data.txt")
    op = shuffle_training_data("O_testing_Data.txt", "P_testing_Data.txt")
    oq = shuffle_training_data("O_testing_Data.txt", "Q_testing_Data.txt")
    or_ = shuffle_training_data("O_testing_Data.txt", "R_testing_Data.txt")
    os = shuffle_training_data("O_testing_Data.txt", "S_testing_Data.txt")
    ot = shuffle_training_data("O_testing_Data.txt", "T_testing_Data.txt")
    ou = shuffle_training_data("O_testing_Data.txt", "U_testing_Data.txt")
    ov = shuffle_training_data("O_testing_Data.txt", "V_testing_Data.txt")
    ow = shuffle_training_data("O_testing_Data.txt", "W_testing_Data.txt")
    ox = shuffle_training_data("O_testing_Data.txt", "X_testing_Data.txt")
    oy = shuffle_training_data("O_testing_Data.txt", "Y_testing_Data.txt")
    oz = shuffle_training_data("O_testing_Data.txt", "Z_testing_Data.txt")
    pq = shuffle_training_data("P_testing_Data.txt", "Q_testing_Data.txt")
    pr = shuffle_training_data("P_testing_Data.txt", "R_testing_Data.txt")
    ps = shuffle_training_data("P_testing_Data.txt", "S_testing_Data.txt")
    pt = shuffle_training_data("P_testing_Data.txt", "T_testing_Data.txt")
    pu = shuffle_training_data("P_testing_Data.txt", "U_testing_Data.txt")
    pv = shuffle_training_data("P_testing_Data.txt", "V_testing_Data.txt")
    pw = shuffle_training_data("P_testing_Data.txt", "W_testing_Data.txt")
    px = shuffle_training_data("P_testing_Data.txt", "X_testing_Data.txt")
    py = shuffle_training_data("P_testing_Data.txt", "Y_testing_Data.txt")
    pz = shuffle_training_data("P_testing_Data.txt", "Z_testing_Data.txt")
    qr = shuffle_training_data("Q_testing_Data.txt", "R_testing_Data.txt")
    qs = shuffle_training_data("Q_testing_Data.txt", "S_testing_Data.txt")
    qt = shuffle_training_data("Q_testing_Data.txt", "T_testing_Data.txt")
    qu = shuffle_training_data("Q_testing_Data.txt", "U_testing_Data.txt")
    qv = shuffle_training_data("Q_testing_Data.txt", "V_testing_Data.txt")
    qw = shuffle_training_data("Q_testing_Data.txt", "W_testing_Data.txt")
    qx = shuffle_training_data("Q_testing_Data.txt", "X_testing_Data.txt")
    qy = shuffle_training_data("Q_testing_Data.txt", "Y_testing_Data.txt")
    qz = shuffle_training_data("Q_testing_Data.txt", "Z_testing_Data.txt")
    rs = shuffle_training_data("R_testing_Data.txt", "S_testing_Data.txt")
    rt = shuffle_training_data("R_testing_Data.txt", "T_testing_Data.txt")
    ru = shuffle_training_data("R_testing_Data.txt", "U_testing_Data.txt")
    rv = shuffle_training_data("R_testing_Data.txt", "V_testing_Data.txt")
    rw = shuffle_training_data("R_testing_Data.txt", "W_testing_Data.txt")
    rx = shuffle_training_data("R_testing_Data.txt", "X_testing_Data.txt")
    ry = shuffle_training_data("R_testing_Data.txt", "Y_testing_Data.txt")
    rz = shuffle_training_data("R_testing_Data.txt", "Z_testing_Data.txt")
    st = shuffle_training_data("S_testing_Data.txt", "T_testing_Data.txt")
    su = shuffle_training_data("S_testing_Data.txt", "U_testing_Data.txt")
    sv = shuffle_training_data("S_testing_Data.txt", "V_testing_Data.txt")
    sw = shuffle_training_data("S_testing_Data.txt", "W_testing_Data.txt")
    sx = shuffle_training_data("S_testing_Data.txt", "X_testing_Data.txt")
    sy = shuffle_training_data("S_testing_Data.txt", "Y_testing_Data.txt")
    sz = shuffle_training_data("S_testing_Data.txt", "Z_testing_Data.txt")
    tu = shuffle_training_data("T_testing_Data.txt", "U_testing_Data.txt")
    tv = shuffle_training_data("T_testing_Data.txt", "V_testing_Data.txt")
    tw = shuffle_training_data("T_testing_Data.txt", "W_testing_Data.txt")
    tx = shuffle_training_data("T_testing_Data.txt", "X_testing_Data.txt")
    ty = shuffle_training_data("T_testing_Data.txt", "Y_testing_Data.txt")
    tz = shuffle_training_data("T_testing_Data.txt", "Z_testing_Data.txt")
    uv = shuffle_training_data("U_testing_Data.txt", "V_testing_Data.txt")
    uw = shuffle_training_data("U_testing_Data.txt", "W_testing_Data.txt")
    ux = shuffle_training_data("U_testing_Data.txt", "X_testing_Data.txt")
    uy = shuffle_training_data("U_testing_Data.txt", "Y_testing_Data.txt")
    uz = shuffle_training_data("U_testing_Data.txt", "Z_testing_Data.txt")
    vw = shuffle_training_data("V_testing_Data.txt", "W_testing_Data.txt")
    vx = shuffle_training_data("V_testing_Data.txt", "X_testing_Data.txt")
    vy = shuffle_training_data("V_testing_Data.txt", "Y_testing_Data.txt")
    vz = shuffle_training_data("V_testing_Data.txt", "Z_testing_Data.txt")
    wx = shuffle_training_data("W_testing_Data.txt", "X_testing_Data.txt")
    wy = shuffle_training_data("W_testing_Data.txt", "Y_testing_Data.txt")
    wz = shuffle_training_data("W_testing_Data.txt", "Z_testing_Data.txt")
    xy = shuffle_training_data("X_testing_Data.txt", "Y_testing_Data.txt")
    xz = shuffle_training_data("X_testing_Data.txt", "Z_testing_Data.txt")
    yz = shuffle_training_data("Y_testing_Data.txt", "Z_testing_Data.txt")


    #More horrific coding. Should have been fixed.
    #Went with what I know works instead.
    perceptrons = training_perceptrons(perceptrons,ab,"A","B")
    # print("testing accuracy for 0")
    # print(perceptrons[0].accuracy)
    perceptrons = training_perceptrons(perceptrons,ac, "A","C")
    # print(perceptrons[1].accuracy)
    perceptrons = training_perceptrons(perceptrons,ad, "A","D")
    # print(perceptrons[2].accuracy)
    perceptrons = training_perceptrons(perceptrons,ae, "A","E")
    # print(perceptrons[3].accuracy)
    perceptrons = training_perceptrons(perceptrons,af, "A","F")
    # print(perceptrons[4].accuracy)
    perceptrons = training_perceptrons(perceptrons,ag, "A","G")
    # print(perceptrons[5].accuracy)
    perceptrons = training_perceptrons(perceptrons,ah, "A","H")
    # print(perceptrons[6].accuracy)
    perceptrons = training_perceptrons(perceptrons,ai, "A","I")
    # print(perceptrons[7].accuracy)
    perceptrons = training_perceptrons(perceptrons,aj, "A","J")
    # print(perceptrons[8].accuracy)
    perceptrons = training_perceptrons(perceptrons,ak, "A","K")
    # print(perceptrons[9].accuracy)
    perceptrons = training_perceptrons(perceptrons,al, "A","L")
    # print(perceptrons[10].accuracy)
    perceptrons = training_perceptrons(perceptrons,am, "A","M")
    # print(perceptrons[11].accuracy)
    perceptrons = training_perceptrons(perceptrons,an, "A","N")
    # print(perceptrons[12].accuracy)
    perceptrons = training_perceptrons(perceptrons,ao, "A","O")
    # print(perceptrons[13].accuracy)
    perceptrons = training_perceptrons(perceptrons,ap, "A","P")
    # print(perceptrons[14].accuracy)
    perceptrons = training_perceptrons(perceptrons,aq, "A","Q")
    # print(perceptrons[15].accuracy)
    perceptrons = training_perceptrons(perceptrons,ar, "A","R")
    # print(perceptrons[16].accuracy)
    perceptrons = training_perceptrons(perceptrons,as_, "A","S")
    # print(perceptrons[17].accuracy)
    perceptrons = training_perceptrons(perceptrons,at, "A","T")
    # print(perceptrons[18].accuracy)
    perceptrons = training_perceptrons(perceptrons,au, "A","U")
    # print("Testing accuracy for 19")
    # print(perceptrons[19].accuracy)
    perceptrons = training_perceptrons(perceptrons,av, "A","V")
    # print("testing accuracy for 20")
    # print(perceptrons[20].accuracy)
    perceptrons = training_perceptrons(perceptrons,aw, "A","W")
    perceptrons = training_perceptrons(perceptrons,ax, "A","X")
    perceptrons = training_perceptrons(perceptrons,ay, "A","Y")
    perceptrons = training_perceptrons(perceptrons,az, "A","Z")
    perceptrons = training_perceptrons(perceptrons,bc, "B","C")
    perceptrons = training_perceptrons(perceptrons,bd, "B","D")
    perceptrons = training_perceptrons(perceptrons,be, "B","E")
    perceptrons = training_perceptrons(perceptrons,bf, "B","F")
    perceptrons = training_perceptrons(perceptrons,bg, "B","G")
    perceptrons = training_perceptrons(perceptrons,bh, "B","H")
    perceptrons = training_perceptrons(perceptrons,bi, "B","I")
    perceptrons = training_perceptrons(perceptrons,bj, "B","J")
    perceptrons = training_perceptrons(perceptrons,bk, "B","K")
    perceptrons = training_perceptrons(perceptrons,bl, "B","L")
    perceptrons = training_perceptrons(perceptrons,bm, "B","M")
    perceptrons = training_perceptrons(perceptrons,bn, "B","N")
    perceptrons = training_perceptrons(perceptrons,bo, "B","O")
    perceptrons = training_perceptrons(perceptrons,bp, "B","P")
    perceptrons = training_perceptrons(perceptrons,bq, "B","Q")
    perceptrons = training_perceptrons(perceptrons,br, "B","R")
    perceptrons = training_perceptrons(perceptrons,bs, "B","S")
    # print("testing accuracy for 41")
    # print(perceptrons[41].accuracy)
    perceptrons = training_perceptrons(perceptrons,bt, "B","T")
    perceptrons = training_perceptrons(perceptrons,bu, "B","U")
    perceptrons = training_perceptrons(perceptrons,bv, "B","V")
    perceptrons = training_perceptrons(perceptrons,bw, "B","W")
    perceptrons = training_perceptrons(perceptrons,bx, "B","X")
    perceptrons = training_perceptrons(perceptrons,by, "B","Y")
    perceptrons = training_perceptrons(perceptrons,bz, "B","Z")
    perceptrons = training_perceptrons(perceptrons,cd, "C","D")
    perceptrons = training_perceptrons(perceptrons,ce, "C","E")
    perceptrons = training_perceptrons(perceptrons,cf, "C","F")
    perceptrons = training_perceptrons(perceptrons,cg, "C","G")
    perceptrons = training_perceptrons(perceptrons,ch, "C","H")
    perceptrons = training_perceptrons(perceptrons,ci, "C","I")
    perceptrons = training_perceptrons(perceptrons,cj, "C","J")
    perceptrons = training_perceptrons(perceptrons,ck, "C","K")
    perceptrons = training_perceptrons(perceptrons,cl, "C","L")
    perceptrons = training_perceptrons(perceptrons,cm, "C","M")
    perceptrons = training_perceptrons(perceptrons,cn, "C","N")
    perceptrons = training_perceptrons(perceptrons,co, "C","O")
    perceptrons = training_perceptrons(perceptrons,cp, "C","P")
    # print("testing accuracy for 61")
    # print(perceptrons[61].accuracy)
    perceptrons = training_perceptrons(perceptrons,cq, "C","Q")
    perceptrons = training_perceptrons(perceptrons,cr, "C","R")
    perceptrons = training_perceptrons(perceptrons,cs, "C","S")
    perceptrons = training_perceptrons(perceptrons,ct, "C","T")
    perceptrons = training_perceptrons(perceptrons,cu, "C","U")
    perceptrons = training_perceptrons(perceptrons,cv, "C","V")
    perceptrons = training_perceptrons(perceptrons,cw, "C","W")
    perceptrons = training_perceptrons(perceptrons,cx, "C","X")
    perceptrons = training_perceptrons(perceptrons,cy, "C","Y")
    perceptrons = training_perceptrons(perceptrons,cz, "C","Z")
    perceptrons = training_perceptrons(perceptrons,de, "D","E")
    perceptrons = training_perceptrons(perceptrons,df, "D","F")
    perceptrons = training_perceptrons(perceptrons,dg, "D","G")
    perceptrons = training_perceptrons(perceptrons,dh, "D","H")
    perceptrons = training_perceptrons(perceptrons,di, "D","I")
    perceptrons = training_perceptrons(perceptrons,dj, "D","J")
    perceptrons = training_perceptrons(perceptrons,dk, "D","K")
    perceptrons = training_perceptrons(perceptrons,dl, "D","L")
    perceptrons = training_perceptrons(perceptrons,dm, "D","M")
    perceptrons = training_perceptrons(perceptrons,dn, "D","N")
    # print("testing accuracy for 81")
    # print(perceptrons[81].accuracy)
    perceptrons = training_perceptrons(perceptrons,do, "D","O")
    perceptrons = training_perceptrons(perceptrons,dp, "D","P")
    perceptrons = training_perceptrons(perceptrons,dq, "D","Q")
    perceptrons = training_perceptrons(perceptrons,dr, "D","R")
    perceptrons = training_perceptrons(perceptrons,ds, "D","S")
    perceptrons = training_perceptrons(perceptrons,dt, "D","T")
    perceptrons = training_perceptrons(perceptrons,du, "D","U")
    perceptrons = training_perceptrons(perceptrons,dv, "D","V")
    perceptrons = training_perceptrons(perceptrons,dw, "D","W")
    perceptrons = training_perceptrons(perceptrons,dx, "D","X")
    perceptrons = training_perceptrons(perceptrons,dy, "D","Y")
    perceptrons = training_perceptrons(perceptrons,dz, "D","Z")
    perceptrons = training_perceptrons(perceptrons,ef, "E","F")
    perceptrons = training_perceptrons(perceptrons,eg, "E","G")
    perceptrons = training_perceptrons(perceptrons,eh, "E","H")
    perceptrons = training_perceptrons(perceptrons,ei, "E","I")
    perceptrons = training_perceptrons(perceptrons,ej, "E","J")
    perceptrons = training_perceptrons(perceptrons,ek, "E","K")
    perceptrons = training_perceptrons(perceptrons,el, "E","L")
    perceptrons = training_perceptrons(perceptrons,em, "E","M")
    # print("testing accuracy for 101")
    # print(perceptrons[101].accuracy)
    perceptrons = training_perceptrons(perceptrons,en, "E","N")
    perceptrons = training_perceptrons(perceptrons,eo, "E","O")
    perceptrons = training_perceptrons(perceptrons,ep, "E","P")
    perceptrons = training_perceptrons(perceptrons,eq, "E","Q")
    perceptrons = training_perceptrons(perceptrons,er, "E","R")
    perceptrons = training_perceptrons(perceptrons,es, "E","S")
    perceptrons = training_perceptrons(perceptrons,et, "E","T")
    perceptrons = training_perceptrons(perceptrons,eu, "E","U")
    perceptrons = training_perceptrons(perceptrons,ev, "E","V")
    perceptrons = training_perceptrons(perceptrons,ew, "E","W")
    perceptrons = training_perceptrons(perceptrons,ex, "E","X")
    perceptrons = training_perceptrons(perceptrons,ey, "E","Y")
    perceptrons = training_perceptrons(perceptrons,ez, "E","Z")
    perceptrons = training_perceptrons(perceptrons,fg, "F","G")
    perceptrons = training_perceptrons(perceptrons,fh, "F","H")
    perceptrons = training_perceptrons(perceptrons,fi, "F","I")
    perceptrons = training_perceptrons(perceptrons,fj, "F","J")
    perceptrons = training_perceptrons(perceptrons,fk, "F","K")
    perceptrons = training_perceptrons(perceptrons,fl, "F","L")
    perceptrons = training_perceptrons(perceptrons,fm, "F","M")
    # print("testing accuracy for 121")
    # print(perceptrons[121].accuracy)
    perceptrons = training_perceptrons(perceptrons,fn, "F","N")
    perceptrons = training_perceptrons(perceptrons,fo, "F","O")
    perceptrons = training_perceptrons(perceptrons,fp, "F","P")
    perceptrons = training_perceptrons(perceptrons,fq, "F","Q")
    perceptrons = training_perceptrons(perceptrons,fr, "F","R")
    perceptrons = training_perceptrons(perceptrons,fs, "F","S")
    perceptrons = training_perceptrons(perceptrons,ft, "F","T")
    perceptrons = training_perceptrons(perceptrons,fu, "F","U")
    perceptrons = training_perceptrons(perceptrons,fv, "F","V")
    perceptrons = training_perceptrons(perceptrons,fw, "F","W")
    perceptrons = training_perceptrons(perceptrons,fx, "F","X")
    perceptrons = training_perceptrons(perceptrons,fy, "F","Y")
    perceptrons = training_perceptrons(perceptrons,fz, "F","Z")
    perceptrons = training_perceptrons(perceptrons,gh, "G","H")
    perceptrons = training_perceptrons(perceptrons,gi, "G","I")
    perceptrons = training_perceptrons(perceptrons,gj, "G","J")
    perceptrons = training_perceptrons(perceptrons,gk, "G","K")
    perceptrons = training_perceptrons(perceptrons,gl, "G","L")
    perceptrons = training_perceptrons(perceptrons,gm, "G","M")
    perceptrons = training_perceptrons(perceptrons,gn, "G","N")
    # print("testing accuracy for 141")
    # print(perceptrons[141].accuracy)
    perceptrons = training_perceptrons(perceptrons,go, "G","O")
    perceptrons = training_perceptrons(perceptrons,gp, "G","P")
    perceptrons = training_perceptrons(perceptrons,gq, "G","Q")
    perceptrons = training_perceptrons(perceptrons,gr, "G","R")
    perceptrons = training_perceptrons(perceptrons,gs, "G","S")
    perceptrons = training_perceptrons(perceptrons,gt, "G","T")
    perceptrons = training_perceptrons(perceptrons,gu, "G","U")
    perceptrons = training_perceptrons(perceptrons,gv, "G","V")
    perceptrons = training_perceptrons(perceptrons,gw, "G","W")
    perceptrons = training_perceptrons(perceptrons,gx, "G","X")
    perceptrons = training_perceptrons(perceptrons,gy, "G","Y")
    perceptrons = training_perceptrons(perceptrons,gz, "G","Z")
    perceptrons = training_perceptrons(perceptrons,hi, "H","I")
    perceptrons = training_perceptrons(perceptrons,hj, "H","J")
    perceptrons = training_perceptrons(perceptrons,hk, "H","K")
    perceptrons = training_perceptrons(perceptrons,hl, "H","L")
    perceptrons = training_perceptrons(perceptrons,hm, "H","M")
    perceptrons = training_perceptrons(perceptrons,hn, "H","N")
    perceptrons = training_perceptrons(perceptrons,ho, "H","O")
    perceptrons = training_perceptrons(perceptrons,hp, "H","P")
    # print("testing accuracy for 161")
    # print(perceptrons[161].accuracy)
    perceptrons = training_perceptrons(perceptrons,hq, "H","Q")
    perceptrons = training_perceptrons(perceptrons,hr, "H","R")
    perceptrons = training_perceptrons(perceptrons,hs, "H","S")
    perceptrons = training_perceptrons(perceptrons,ht, "H","T")
    perceptrons = training_perceptrons(perceptrons,hu, "H","U")
    perceptrons = training_perceptrons(perceptrons,hv, "H","V")
    perceptrons = training_perceptrons(perceptrons,hw, "H","W")
    perceptrons = training_perceptrons(perceptrons,hx, "H","X")
    perceptrons = training_perceptrons(perceptrons,hy, "H","Y")
    perceptrons = training_perceptrons(perceptrons,hz, "H","Z")
    perceptrons = training_perceptrons(perceptrons,ij, "I","J")
    perceptrons = training_perceptrons(perceptrons,ik, "I","K")
    perceptrons = training_perceptrons(perceptrons,il, "I","L")
    perceptrons = training_perceptrons(perceptrons,im, "I","M")
    perceptrons = training_perceptrons(perceptrons,ins_, "I","N")
    perceptrons = training_perceptrons(perceptrons,io, "I","O")
    perceptrons = training_perceptrons(perceptrons,ip, "I","P")
    perceptrons = training_perceptrons(perceptrons,iq, "I","Q")
    perceptrons = training_perceptrons(perceptrons,ir, "I","R")
    perceptrons = training_perceptrons(perceptrons,is_, "I","S")
    # print("testing accuracy for 181")
    # print(perceptrons[181].accuracy)
    perceptrons = training_perceptrons(perceptrons,it, "I","T")
    perceptrons = training_perceptrons(perceptrons,iu, "I","U")
    perceptrons = training_perceptrons(perceptrons,iv, "I","V")
    perceptrons = training_perceptrons(perceptrons,iw, "I","W")
    perceptrons = training_perceptrons(perceptrons,ix, "I","X")
    perceptrons = training_perceptrons(perceptrons,iy, "I","Y")
    perceptrons = training_perceptrons(perceptrons,iz, "I","Z")
    perceptrons = training_perceptrons(perceptrons,jk, "J","K")
    perceptrons = training_perceptrons(perceptrons,jl, "J","L")
    perceptrons = training_perceptrons(perceptrons,jm, "J","M")
    perceptrons = training_perceptrons(perceptrons,jn, "J","N")
    perceptrons = training_perceptrons(perceptrons,jo, "J","O")
    perceptrons = training_perceptrons(perceptrons,jp, "J","P")
    perceptrons = training_perceptrons(perceptrons,jq, "J","Q")
    perceptrons = training_perceptrons(perceptrons,jr, "J","R")
    perceptrons = training_perceptrons(perceptrons,js, "J","S")
    perceptrons = training_perceptrons(perceptrons,jt, "J","T")
    perceptrons = training_perceptrons(perceptrons,ju, "J","U")
    perceptrons = training_perceptrons(perceptrons,jv, "J","V")
    perceptrons = training_perceptrons(perceptrons,jw, "J","W")
    # print("testing accuracy for 201")
    # print(perceptrons[201].accuracy)
    perceptrons = training_perceptrons(perceptrons,jx, "J","X")
    perceptrons = training_perceptrons(perceptrons,jy, "J","Y")
    perceptrons = training_perceptrons(perceptrons,jz, "J","Z")
    perceptrons = training_perceptrons(perceptrons,kl, "K","L")
    perceptrons = training_perceptrons(perceptrons,km, "K","M")
    perceptrons = training_perceptrons(perceptrons,kn, "K","N")
    perceptrons = training_perceptrons(perceptrons,ko, "K","O")
    perceptrons = training_perceptrons(perceptrons,kp, "K","P")
    perceptrons = training_perceptrons(perceptrons,kq, "K","Q")
    perceptrons = training_perceptrons(perceptrons,kr, "K","R")
    perceptrons = training_perceptrons(perceptrons,ks, "K","S")
    perceptrons = training_perceptrons(perceptrons,kt, "K","T")
    perceptrons = training_perceptrons(perceptrons,ku, "K","U")
    perceptrons = training_perceptrons(perceptrons,kv, "K","V")
    perceptrons = training_perceptrons(perceptrons,kw, "K","W")
    perceptrons = training_perceptrons(perceptrons,kx, "K","X")
    perceptrons = training_perceptrons(perceptrons,ky, "K","Y")
    perceptrons = training_perceptrons(perceptrons,kz, "K","Z")
    perceptrons = training_perceptrons(perceptrons,lm, "L","M")
    perceptrons = training_perceptrons(perceptrons,ln, "L","N")
    # print("testing accuracy for 221")
    # print(perceptrons[221].accuracy)
    perceptrons = training_perceptrons(perceptrons,lo, "L","O")
    perceptrons = training_perceptrons(perceptrons,lp, "L","P")
    perceptrons = training_perceptrons(perceptrons,lq, "L","Q")
    perceptrons = training_perceptrons(perceptrons,lr, "L","R")
    perceptrons = training_perceptrons(perceptrons,ls, "L","S")
    perceptrons = training_perceptrons(perceptrons,lt, "L","T")
    perceptrons = training_perceptrons(perceptrons,lu, "L","U")
    perceptrons = training_perceptrons(perceptrons,lv, "L","V")
    perceptrons = training_perceptrons(perceptrons,lw, "L","W")
    perceptrons = training_perceptrons(perceptrons,lx, "L","X")
    perceptrons = training_perceptrons(perceptrons,ly, "L","Y")
    perceptrons = training_perceptrons(perceptrons,lz, "L","Z")
    perceptrons = training_perceptrons(perceptrons,mn, "M","N")
    perceptrons = training_perceptrons(perceptrons,mo, "M","O")
    perceptrons = training_perceptrons(perceptrons,mp, "M","P")
    perceptrons = training_perceptrons(perceptrons,mq, "M","Q")
    perceptrons = training_perceptrons(perceptrons,mr, "M","R")
    perceptrons = training_perceptrons(perceptrons,ms, "M","S")
    perceptrons = training_perceptrons(perceptrons,mt, "M","T")
    perceptrons = training_perceptrons(perceptrons,mu, "M","U")
    # print("testing accuracy for 241")
    # print(perceptrons[241].accuracy)
    perceptrons = training_perceptrons(perceptrons,mv, "M","V")
    perceptrons = training_perceptrons(perceptrons,mw, "M","W")
    perceptrons = training_perceptrons(perceptrons,mx, "M","X")
    perceptrons = training_perceptrons(perceptrons,my, "M","Y")
    perceptrons = training_perceptrons(perceptrons,mz, "M","Z")
    perceptrons = training_perceptrons(perceptrons,no, "N","O")
    perceptrons = training_perceptrons(perceptrons,np, "N","P")
    perceptrons = training_perceptrons(perceptrons,nq, "N","Q")
    perceptrons = training_perceptrons(perceptrons,nr, "N","R")
    perceptrons = training_perceptrons(perceptrons,ns, "N","S")
    perceptrons = training_perceptrons(perceptrons,nt, "N","T")
    perceptrons = training_perceptrons(perceptrons,nu, "N","U")
    perceptrons = training_perceptrons(perceptrons,nv, "N","V")
    perceptrons = training_perceptrons(perceptrons,nw, "N","W")
    perceptrons = training_perceptrons(perceptrons,nx, "N","X")
    perceptrons = training_perceptrons(perceptrons,ny, "N","Y")
    perceptrons = training_perceptrons(perceptrons,nz, "N","Z")
    perceptrons = training_perceptrons(perceptrons,op, "O","P")
    perceptrons = training_perceptrons(perceptrons,oq, "O","Q")
    perceptrons = training_perceptrons(perceptrons,or_, "O","R")
    # print("testing accuracy for 261")
    # print(perceptrons[261].accuracy)
    perceptrons = training_perceptrons(perceptrons,os, "O","S")
    perceptrons = training_perceptrons(perceptrons,ot, "O","T")
    perceptrons = training_perceptrons(perceptrons,ou, "O","U")
    perceptrons = training_perceptrons(perceptrons,ov, "O","V")
    perceptrons = training_perceptrons(perceptrons,ow, "O","W")
    perceptrons = training_perceptrons(perceptrons,ox, "O","X")
    perceptrons = training_perceptrons(perceptrons,oy, "O","Y")
    perceptrons = training_perceptrons(perceptrons,oz, "O","Z")
    perceptrons = training_perceptrons(perceptrons,pq, "P","Q")
    perceptrons = training_perceptrons(perceptrons,pr, "P","R")
    perceptrons = training_perceptrons(perceptrons,ps, "P","S")
    perceptrons = training_perceptrons(perceptrons,pt, "P","T")
    perceptrons = training_perceptrons(perceptrons,pu, "P","U")
    perceptrons = training_perceptrons(perceptrons,pv, "P","V")
    perceptrons = training_perceptrons(perceptrons,pw, "P","W")
    perceptrons = training_perceptrons(perceptrons,px, "P","X")
    perceptrons = training_perceptrons(perceptrons,py, "P","Y")
    perceptrons = training_perceptrons(perceptrons,pz, "P","Z")
    perceptrons = training_perceptrons(perceptrons,qr, "Q","R")
    perceptrons = training_perceptrons(perceptrons,qs, "Q","S")
    # print("testing accuracy for 281")
    # print(perceptrons[281].accuracy)
    perceptrons = training_perceptrons(perceptrons,qt, "Q","T")
    perceptrons = training_perceptrons(perceptrons,qu, "Q","U")
    perceptrons = training_perceptrons(perceptrons,qv, "Q","V")
    perceptrons = training_perceptrons(perceptrons,qw, "Q","W")
    perceptrons = training_perceptrons(perceptrons,qx, "Q","X")
    perceptrons = training_perceptrons(perceptrons,qy, "Q","Y")
    perceptrons = training_perceptrons(perceptrons,qz, "Q","Z")
    perceptrons = training_perceptrons(perceptrons,rs, "R","S")
    perceptrons = training_perceptrons(perceptrons,rt, "R","T")
    perceptrons = training_perceptrons(perceptrons,ru, "R","U")
    perceptrons = training_perceptrons(perceptrons,rv, "R","V")
    perceptrons = training_perceptrons(perceptrons,rw, "R","W")
    perceptrons = training_perceptrons(perceptrons,rx, "R","X")
    perceptrons = training_perceptrons(perceptrons,ry, "R","Y")
    perceptrons = training_perceptrons(perceptrons,rz, "R","Z")
    perceptrons = training_perceptrons(perceptrons,st, "S","T")
    perceptrons = training_perceptrons(perceptrons,su, "S","U")
    perceptrons = training_perceptrons(perceptrons,sv, "S","V")
    perceptrons = training_perceptrons(perceptrons,sw, "S","W")
    perceptrons = training_perceptrons(perceptrons,sx, "S","X")
    # print("testing accuracy for 301")
    # print(perceptrons[301].accuracy)
    perceptrons = training_perceptrons(perceptrons,sy, "S","Y")
    perceptrons = training_perceptrons(perceptrons,sz, "S","Z")
    perceptrons = training_perceptrons(perceptrons,tu, "T","U")
    perceptrons = training_perceptrons(perceptrons,tv, "T","V")
    perceptrons = training_perceptrons(perceptrons,tw, "T","W")
    perceptrons = training_perceptrons(perceptrons,tx, "T","X")
    perceptrons = training_perceptrons(perceptrons,ty, "T","Y")
    perceptrons = training_perceptrons(perceptrons,tz, "T","Z")
    perceptrons = training_perceptrons(perceptrons,uv, "U","V")
    perceptrons = training_perceptrons(perceptrons,uw, "U","W")
    perceptrons = training_perceptrons(perceptrons,ux, "U","X")
    perceptrons = training_perceptrons(perceptrons,uy, "U","Y")
    perceptrons = training_perceptrons(perceptrons,uz, "U","Z")
    perceptrons = training_perceptrons(perceptrons,vw, "V","W")
    perceptrons = training_perceptrons(perceptrons,vx, "V","X")
    perceptrons = training_perceptrons(perceptrons,vy, "V","Y")
    perceptrons = training_perceptrons(perceptrons,vz, "V","Z")
    perceptrons = training_perceptrons(perceptrons,wx, "W","X")
    perceptrons = training_perceptrons(perceptrons,wy, "W","Y")
    perceptrons = training_perceptrons(perceptrons,wz, "W","Z")
    # print("testing accuracy for 321")
    # print(perceptrons[321].accuracy)
    perceptrons = training_perceptrons(perceptrons,xy, "X","Y")
    perceptrons = training_perceptrons(perceptrons,xz, "X","Z")
    perceptrons = training_perceptrons(perceptrons,yz, "Y","Z")
    # print("testing accuracy for 324")
    # print(perceptrons[324].accuracy)
    #for i in range(324):
        #print(perceptrons[i].accuracy)

    # Making sure all of the perceptrons are working.
    for i in range(324):
        if perceptrons[i].accuracy == 0:
            print(i)


    return perceptrons

#Main function definition
def main():

    # Makes the testing data. Very ugly function. Should optimize later.
    print("Making the training data...")
    test_data = make_training_data("letter_data_halves2.txt")

    #Making the perceptron list
    print("Making the perceptrons...")
    perceptrons = create_perceptron_array()

    # Training perceptrons. Also very ugly function. Should optimize later.
    print("Training the perceptrons...")
    train_all_perceptrons(perceptrons)

    #Making the matrix of confusion
    confused_matrix = np.zeros(shape=(27,27),dtype=int)

    #Used for overall accuracy
    count = 0

    #Running the test data against the perceptron list.
    print("Testing has started!")
    for i in range(len(test_data)):
        #Split down into a single row to test with perceptron list
        to_test = test_data[i,0:]

        #Get the actual letter index
        actual_letter = test_data[i,0]
        y = actual_letter.astype(int)

        #Find what letter the perceptron list thinks it's looking at
        letter = test_perceptron_data(to_test,perceptrons)

        #If there is a match, increase count for accuracy
        if test_data[i,0] == letter:
            count += 1
        #Confusion matrix for what the perceptron sees index
        x = letter.astype(int)
        confused_matrix[x,y] += 1
    #Report the accuracy
    print("The overall accuracy of the perceptrons is:",count/len(test_data))

    #Report the matrix. 26 v 26. Index 1:1 is A:A, and 1:2 is A:B and so on.
    #Shows a count for how many times the program had this combination happen
    #of which letter is input and what letter is output in integer format
    print("The confusion matrix of the test data:")
    print(confused_matrix[1:,1:])

main()