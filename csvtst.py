__author__ = 'skynet'

import csv
import math
import copy
from scipy import stats
#This is the important lists, which contain the training data, results, the test data and its results and some intermediate lists .
train_lab = []
train_data = []
train_data_domain = []
full_attr_domain = []
attr_midpoints = []
test_data = []
test_result=[]
test_lab= []

#This is the decision tree node structure, which has the following important elements.
#1. Attribute boolean, this list is used to keep track of what attributes have been used for splitting.
#2. Attribute sample space, this is used to keep track of the sample space of different nodes while splitting.
#3. True and False keeps track of the actual true and false values for that node.
#4. Splitting attribute, is the attribute based on which the node is being split
class decision_tree_node:
    def __init__(self):
        #self.state =  [['o' for x in range(7)] for x in range(7)]
        self.child_list = []
        self.attribute_boolean = [0 for i in range(len(train_data_domain))]
        self.attribute_sample_space = [0 for i in range(len(train_data))]
        self.true = 0
        self.false = 0
        self.splitting_attribute = 0
        #initial sample space empty for all attributes.update for nodes. initialise root to all 1's.



#This procedure is used to extract the data from the csv files and put them into usable lists.
def csvtest():

    read_train_feat = csv.reader(open('/home/skynet/Python/Projects/clickstream-data/trainfeat.csv','rb'))
    read_train_labs = csv.reader(open('/home/skynet/Python/Projects/clickstream-data/trainlabs.csv','rb'))
    read_test_feat = csv.reader(open('/home/skynet/Python/Projects/clickstream-data/testfeat.csv','rb'))
    read_test_labs = csv.reader(open('/home/skynet/Python/Projects/clickstream-data/testlabs.csv','rb'))

    for row in read_train_labs:
        train_lab.append(int(row[0]))
    #print train_lab

    for row in read_train_feat:
        a = row[0].split()
        for i in range(len(a)):
            a[i] = int(a[i])
        train_data.append(a)

    for row in read_test_feat:
        a = row[0].split()
        for i in range(len(a)):
            a[i] = int(a[i])
        test_data.append(a)

    for row in read_test_labs:
        test_lab.append(int(row[0]))




#This method is used to find the what are the different ranges of values, the attribute takes and this can be used to take care of the splitting.



def attribute_domain():
    no_of_attr = len(train_data[0])
    train_set_len = len(train_data)
    #print "attr no:", no_of_attr
    #print "train set length:", train_set_len
    for attribute in range(no_of_attr):
        temp_attr_domain = []
        for i in range(train_set_len):
            if train_data[i][attribute] not in temp_attr_domain:
                temp_attr_domain.append(train_data[i][attribute])

        train_data_domain.append(temp_attr_domain)

    for i in range(len(train_data_domain)):

        attr_midpoints.append(float((max(train_data_domain[i])+min(train_data_domain[i]))/2))


    for attribute in range(len(train_data_domain)):
        temp_list = []
        for j in range(len(train_data_domain[attribute])):
            count = 0
            for k in range(len(train_data)):
                if(train_data[k][attribute]==train_data_domain[attribute][j]):
                    count+=1
            temp_list.append((train_data_domain[attribute][j],count))

        full_attr_domain.append(temp_list)

#This method is used to calculate the entropy for each attribute, depending on the sample space of the current node and
#hence, is used to pick the attribute with the maximum information.

def find_next_attribute(attribute_flag, attribute_sample_space):

    information = []

    for i in range(len(train_data[0])):
        information.append(float('inf'))

    sample_space_length = 0

    for i in range(len(attribute_sample_space)):
        if attribute_sample_space[i]==1:
            sample_space_length+=1

    for attribute_no in range(len(attribute_flag)):
        if attribute_flag[attribute_no]==0: #if flag set to 1, attribute has been used

            x_less=0
            x_greater=0
            pr_x_less = 0.0
            pr_x_greater = 0.0
            y_less_true=0
            y_less_false=0
            y_greater_true=0
            y_greater_false=0

            for i in range(len(train_data)):
                if attribute_sample_space[i]==1 and train_data[i][attribute_no]<=attr_midpoints[attribute_no]:
                    x_less+=1
                    if train_lab[i] == 1:
                        y_less_true+=1
                    else:
                        y_less_false+=1
                if attribute_sample_space[i]==1 and train_data[i][attribute_no]>attr_midpoints[attribute_no]:
                    x_greater+=1
                    if train_lab[i]==1:
                        y_greater_true+=1
                    else:
                        y_greater_false+=1

            pr_x_less = float(x_less)/float(sample_space_length)#make it proper floating point, else will give error.denominator wrong.
            pr_x_greater = float(x_greater)/float(sample_space_length)


            if pr_x_less==0:
                pr_y_less_true=0.0
                pr_y_less_false=0.0
            else:
                pr_y_less_true = float(y_less_true)/float(y_less_true+y_less_false)
                pr_y_less_false = float(y_less_false)/float(y_less_true+y_less_false)
            if pr_x_greater==0:
                pr_y_greater_true=0.0
                pr_y_greater_false=0.0
            else:

                pr_y_greater_true = float(y_greater_true)/float(y_greater_true+y_greater_false)
                pr_y_greater_false = float(y_greater_false)/float(y_greater_true+y_greater_false)

            if pr_y_less_true==0.0:
                log_pr_y_less_true=float('-inf')

            else:
                log_pr_y_less_true = float(math.log(pr_y_less_true,2))

            if pr_y_less_false == 0.0:
                log_pr_y_less_false = float('-inf')

            else:
                log_pr_y_less_false = float(math.log(pr_y_less_false,2))
            if pr_y_greater_true==0.0:
                log_pr_y_greater_true = float('-inf')

            else:
                log_pr_y_greater_true = float(math.log(pr_y_greater_true,2))
            if pr_y_greater_false==0.0:
                log_pr_y_greater_false = float('-inf')

            else:
                log_pr_y_greater_false = float(math.log(pr_y_greater_false,2))
            if log_pr_y_less_true == float('-inf') or log_pr_y_less_false == float('-inf'):
                entropy_x_less = float('-inf')
            else:
                entropy_x_less = -(pr_x_less*((pr_y_less_true)*log_pr_y_less_true+(pr_y_less_false)*log_pr_y_less_false))

            if log_pr_y_greater_false ==float('-inf') or log_pr_y_greater_true==float('-inf'):
                entropy_x_greater=float('-inf')
            else:
                entropy_x_greater = -(pr_x_greater*((pr_y_greater_true)*log_pr_y_greater_true+(pr_y_greater_false)*log_pr_y_greater_false))
            entropy_x = entropy_x_less+entropy_x_greater

            information[attribute_no]=entropy_x



    return information.index(min(information))

#This is the main method which creates the decision tree in a recursive manner.
#Based on the attribute, the sample space is split into atmost two parts and assigned to the children. Pruning methods
#are done by calculating the chi square statistic. In addition, other pruning is also done, based on the number of true and
#false values in each attribute. For ex: If there is a 100% majority of either True or False values, the node is terminated as a leaf node
#and further tree building does not happen in that node.


def create_decision_tree(node):

    for i in range(len(node.attribute_boolean)):
        if node.attribute_boolean[i]==0:

            break
    else:

        return 1

    for i in range(len(node.attribute_sample_space)):
        if node.attribute_sample_space[i]==1:
            break
    else:

        return 1

    attribute_to_expand = find_next_attribute(node.attribute_boolean, node.attribute_sample_space)


    node.splitting_attribute = attribute_to_expand
    node.attribute_boolean[attribute_to_expand]=1


    node_positive=0
    node_negative=0
    sample_space_length=0
    for i in range(len(node.attribute_sample_space)):
        if node.attribute_sample_space[i]==1:
            sample_space_length+=1
            if train_lab[i]==1:
                node_positive+=1
            else:
                node_negative+=1



#Initialisation values for the different values used to calculate the chi square statistic.
    T_1=0
    T_2=0

    p_1_true=0
    p_1_false=0
    p_2_true=0
    p_2_false=0

    for j in range(len(train_data)):
        if(train_data[j][attribute_to_expand]<=attr_midpoints[attribute_to_expand]):
            T_1+=1
            if train_lab[j]==1:
                p_1_true+=1
            else:
                p_1_false+=1

    for j in range(len(train_data)):
        if(train_data[j][attribute_to_expand]>attr_midpoints[attribute_to_expand]):
            T_2+=1
            if train_lab[j]==1:
                p_2_true+=1
            else:
                p_2_false+=1

    total_children = 0
    if T_1 >0 and T_2 >0:
        total_children=2
    elif T_1==0 and T_2==0:
        total_children=0
    else:
        total_children=1

    p_1_true_exp= float(node_positive)*float(T_1)/float(sample_space_length)
    p_1_false_exp= float(node_negative)*float(T_1)/float(sample_space_length)
    p_2_true_exp= float(node_positive)*float(T_2)/float(sample_space_length)
    p_2_false_exp= float(node_negative)*float(T_2)/float(sample_space_length)

    chi_p1_f=0
    chi_p1_t=0
    chi_p2_f=0
    chi_p2_t=0

    if p_1_true_exp!=0:
        chi_p1_t = float((math.pow((p_1_true_exp-p_1_true),2))/float(p_1_true_exp))
    if p_1_false_exp!=0:
        chi_p1_f = float((math.pow((p_1_false_exp-p_1_false),2))/float(p_1_false_exp))
    if p_2_true_exp!=0:
        chi_p2_t = float((math.pow((p_2_true_exp-p_2_true),2))/float(p_2_true_exp))
    if p_2_false_exp!=0:
        chi_p2_f = float((math.pow((p_2_false_exp-p_2_false),2))/float(p_2_false_exp))

    #S value is calculated according to the formula and the chi square statistic is calculated using the scipy python module.
    s = chi_p1_f+chi_p1_t+chi_p2_f+chi_p2_t


    p_value = float(1)-float(stats.chi2._cdf(s,1))

    #Here, the threshold is set and only, if the p-value exceeds the threshold, the tree is built for that node, or else the tree is not built.

    if(p_value < 0.01):


        if(T_1)>0:
            child = decision_tree_node()
            child.attribute_boolean = copy.deepcopy(node.attribute_boolean)
            child.true = p_1_true
            child.false = p_1_false
            for i in range(len(child.attribute_sample_space)):
                if train_data[i][attribute_to_expand]<=attr_midpoints[attribute_to_expand]:
                    child.attribute_sample_space[i]=1
            node.child_list.append(child)

        if(T_2)>0:
            child = decision_tree_node()
            child.attribute_boolean = copy.deepcopy(node.attribute_boolean)
            child.true = p_2_true
            child.false = p_2_false
            for i in range(len(child.attribute_sample_space)):
                if train_data[i][attribute_to_expand]>attr_midpoints[attribute_to_expand]:
                    child.attribute_sample_space[i]=1
            node.child_list.append(child)

        for j in range(total_children):
            if node.child_list[j].true !=0 and node.child_list[j].false !=0:
                success = create_decision_tree(node.child_list[j])
    else:

        success = create_decision_tree(node)

#This method is used to recursively traverse the tree and classify the test data.This is a recursive method.
#The tree is traversed till a leaf node is reached and the test data is classified on the leaf node.

def classify_test_instance(node,data):
    if len(node.child_list)==0:

        if node.true > node.false:

            test_result.append(1)
        else:
            test_result.append(0)
        return
    else:
        if len(node.child_list)==2 and data[node.splitting_attribute]<=attr_midpoints[node.splitting_attribute]:
            classify_test_instance(node.child_list[0],data)
        elif len(node.child_list)==2 and data[node.splitting_attribute]>attr_midpoints[node.splitting_attribute]:
            classify_test_instance(node.child_list[1],data)

#This is the main function, where all the methods to extract data from the excel,creating the decision tree and the test data classficiation methods are called.

def main():
    print "in main"
    csvtest()
    attribute_domain()
    root = decision_tree_node()

    for i in range(len(root.attribute_sample_space)):
        root.attribute_sample_space[i]=1

    tree_Created = create_decision_tree(root)
    for i in range(len(test_data)):
        classify_test_instance(root,test_data[i])

    print len(test_result),":test results:",test_result
    total_correct_predictions = 0
    for i in range(len(test_result)):
        if test_lab[i]==test_result[i]:
            total_correct_predictions+=1
    print "total_correct_predictions:",total_correct_predictions
    print "accuracy:", float(total_correct_predictions)/float(len(test_result))

if __name__ == "__main__":
    main()