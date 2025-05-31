import math

# for implemnting the code i used the example given in the leacture and in the slide of (Classification – Artificial Neural Networks)

# Training data: input (1, 0, 1), target output 1
data = [
    ([1, 0, 1], [1])  # sample, target
]

# Learning rate
lrn_rate = 0.9

# Network structure
input_size = 3
hidden_size = 2
output_size = 1

# the weights  
weights_input_hidden = [
    [0.2, -0.3],  # Weights from input 1 to hidden 4 , 5
    [0.4, 0.1],   # Weights from input 2 to hidden 4 , 5
    [-0.5, 0.2]   # Weights from input 3 to hidden 4 , 5
]

weights_output_input = [
    [-0.3],       # Weight from hidden 1 to output
    [-0.2]        # Weight from hidden 2 to output
]

# bias 
bias_hidden = [-0.4, 0.2]
bias_output = 0.1


# prediction befor traning and update the weights and bias
print("predictions befor update wegihts and bias :")
for sample, target in data:
    
    O_hidden = []
    # Ij = wij * Oi + theta_j  
    for j in range(hidden_size):
        I = 0
        for i in range(input_size):
            I += sample[i] * weights_input_hidden[i][j]
        I += bias_hidden[j]
        # using sigmod to find O =  1 / 1 + e ^ (-Ij)
        O =  1 / (1 + math.exp(-I))
        O_hidden.append(O)


    # this is the input to the output layer 
    I_output = 0
    for j in range(hidden_size):
        I_output += O_hidden[j] * weights_output_input[j][0]
    I_output += bias_output
    O_output = 1 / (1 + math.exp(-I_output))
    print(f"Input: {sample}, Predicted output: {O_output}")

# Training loop
traning_count = 0
# we can put a condition to terminate traning 
 
while traning_count < 500:
    for sample, target in data:

        # ouutput from hidden
        O_hidden = []
        for j in range(hidden_size):
            I = 0
            # Ij = wij * Oi + theta_j  
            for i in range(input_size):
                I += weights_input_hidden[i][j] * sample[i] 
            I += bias_hidden[j]
            O = 1 / (1 + math.exp(-I))
            O_hidden.append(O)

        # input to the output layer 
        I_output = 0
        for j in range(hidden_size):
            I_output += O_hidden[j] * weights_output_input[j][0]
        I_output += bias_output
        O_output = 1 / (1 + math.exp(-I_output))

        #### #### backpropigation #### ####

        # Error of the output
        output_err = O_output * (1 - O_output) * (target[0] - O_output)

        # Error of the hidden layer
        hidden_layer_err = []
        for j in range(hidden_size):
            err = O_hidden[j] * (1 - O_hidden[j]) * (output_err * weights_output_input[j][0])
            hidden_layer_err.append(err)

        #### #### weigth and bias updates #### ####

        # Update weights of hidden to output
        for j in range(hidden_size):
            weights_output_input[j][0] += lrn_rate * O_hidden[j] * output_err

        # Update weights input to hidden layer
        for i in range(input_size):
            for j in range(hidden_size):
                weights_input_hidden[i][j] += lrn_rate * sample[i] * hidden_layer_err[j]

        # Update biases
        for j in range(hidden_size):
            bias_hidden[j] += lrn_rate * hidden_layer_err[j]
        bias_output += lrn_rate * output_err
        traning_count += 1
        
        
# prediction after updating the wegihts and the bias
print("prediction after update wegihts and bias :")
for sample, target in data:
    
    O_hidden = []
    for j in range(hidden_size):
        I = 0
        for i in range(input_size):
            I += sample[i] * weights_input_hidden[i][j]
        I += bias_hidden[j]
        O = 1 / (1 + math.exp(-I))
        O_hidden.append(O)

    # this is the input to the output layer 
    I_output = 0
    for j in range(hidden_size):
        I_output += O_hidden[j] * weights_output_input[j][0]
    I_output += bias_output
    O_output = 1 / (1 + math.exp(-I_output))
    print(f"Input: {sample}, Predicted output: {O_output}")
    
print("\n")


# i used this for my data mining project
# models = {
#     "Decision Tree": DecisionTreeClassifier(max_depth = 5, criterion = "gini"),
#     "KNeighbors Classifier": KNeighborsClassifier(n_neighbors = 5),
#     "svm": SVC(),
#     "Navie Bais": GaussianNB(),
#     "Neural Network": MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, activation='logistic', solver='adam', random_state=42),
#     "Random Forest Classifier": RandomForestClassifier(),
#     "xTree": ExtraTreesClassifier(),
#     "Logistic": LogisticRegression(),
#     "xgBoost": XGBClassifier(),
# }



# accuracy_scores = []
# predicted = []

# for i in models:
#     models[i].fit(x_train, y_train)
#     y_pred = models[i].predict(x_test)
#     accuracy_scores.append(int(accuracy_score(y_pred, y_test) * 100))
#     predicted.append(y_pred)

# # for j, k in zip(accuracy_scores, models):
# #     print (' \n ', k, ' accuracy : ', j, ' %  ')