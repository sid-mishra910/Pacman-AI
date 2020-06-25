# bayesNet.py

import itertools
import util

class CPT():
    """
    A table that represents the conditional probabilities.
    This has two components, the dependencyList and the probTable.
    dependencyList is a list of nodes that the CPT depends on.
    probTable is a table of length 2^n where n is the length of dependencyList.
    It represents the probability values of the truth table created by the dependencyList as the boolean values, in the same order.
    That is to say is the depencyList contains A and B then probTable will have 4 values corresponding to (A, B) = (0, 0), (0, 1), (1, 0), (1, 1) in that order.
    """
    def __init__(self, dependencies, probabilities):
        self.dependencyList = dependencies
        self.probTable = probabilities
        # print(self.dependencyList)
        # print(self.probTable)

class BayesNetwork():
    """
    A network represented as a dictionary of nodes and CPTs
    """
    def __init__(self, network):
        """
        Constructor for the BayesNetwork class. By default it only takes in the network
        Feel free to add things to this if you think they will help.
        """
        self.network = network
        # print(self.network)
        "*** YOUR CODE HERE ***"
        # print(self.network['B'].probTable)

    def singleInference(self, A, B):
        """
        Return the probability of A given B using the Bayes Network. Here B is a tuple of (node, boolean).
        """
        "*** YOUR CODE HERE ***"

        # self.network1 = network
        e = []
        e.append(B)
        value = self.enum_ask(A, e, self.network)
        value = self.network1['False'] + self.network1['True']
        value = round(self.network1['True']/value,4)
        return value


        util.raiseNotDefined()

    def enum_ask(self,X, e, bn):
        # return 0
        self.network1 = util.Counter()
        cnt = 0
        e_1 = []
        e_2 = []
        e_1.extend(e)
        e_2.extend(e)
        for x_i in X:
            for i in range(0,2):
                vars = []
                vars.extend(bn.keys())
                if cnt == 0:
                    e_1.append((x_i, False))
                    cnt = (cnt + 1)%2
                    self.network1['False'] = self.enum_all(vars, e_1)
                else:
                    e_2.append((x_i, True))
                    cnt = (cnt + 1)%2
                    self.network1['True'] = self.enum_all(vars, e_2)
        return self.network1
        util.raiseNotDefined()

    def enum_all(self, vars, e):
        if len(vars) == 0:
            return 1.0
        Y = vars[0]
        
        Y_e = self.search_in_e(Y, e)
       
        if len(Y_e) > 0:
            Y_ref = self.network[Y]
            dep_list_Y = Y_ref.dependencyList
            # print(dep_list_Y)
            probTable_Y = Y_ref.probTable
            parent_val = self.search_in_e(dep_list_Y, e)
            # print(Y_e[0][1])
            # print(parent_val)
            if len(parent_val) == 0:
                if Y_e[0][1] == True:
                    vars.remove(Y)
                    return (probTable_Y[0] * self.enum_all(vars, e))
                elif Y_e[0][1] == False:
                    vars.remove(Y)
                    return ((1 - probTable_Y[0]) * self.enum_all(vars, e))
            if len(parent_val) == 1:
                if Y_e[0][1] == False:
                    if parent_val[0][1] == False:
                        vars.remove(Y)
                        return ((1 - probTable_Y[0]) * self.enum_all(vars,e))
                    elif parent_val[0][1] == True:
                        vars.remove(Y)
                        return ((1 - probTable_Y[1]) * self.enum_all(vars,e))
                elif Y_e[0][1] == True:
                    if parent_val[0][1] == False:
                        vars.remove(Y)
                        return (probTable_Y[0] * self.enum_all(vars,e))
                    elif parent_val[0][1] == True:
                        vars.remove(Y)
                        return (probTable_Y[1] * self.enum_all(vars,e))

            if len(parent_val) == 2:
                if Y_e[0][1] == False:
                    vars.remove(Y)
                    if (parent_val[0][1] == False) and (parent_val[1][1] == False):
                        return ((1 - probTable_Y[0]) * self.enum_all(vars,e))
                    elif parent_val[0][1] == False and parent_val[1][1] == True:
                        return ((1 - probTable_Y[1]) * self.enum_all(vars,e))
                    elif parent_val[0][1] == True and parent_val[1][1] == False:
                        return ((1 - probTable_Y[2]) * self.enum_all(vars, e))
                    else:
                        return ((1 - probTable_Y[3]) * self.enum_all(vars,e))
                elif Y_e[0][1] == True:
                    vars.remove(Y)
                    if parent_val[0][1] == False and parent_val[1][1] == False:
                        return ((probTable_Y[0]) * self.enum_all(vars,e))
                    elif parent_val[0][1] == False and parent_val[1][1] == True:
                        return ((probTable_Y[1]) * self.enum_all(vars,e))
                    elif parent_val[0][1] == True and parent_val[1][1] == False:
                        return ((probTable_Y[2]) * self.enum_all(vars, e))
                    else:
                        return ((probTable_Y[3]) * self.enum_all(vars,e))
                # return 1
        



        if len(Y_e) == 0:
            e_y_1 = []
            e_y_2 = []
            sum = 0
            e_y_1.extend(e)
            e_y_2.extend(e)
            Y_ref = self.network[Y]
            dep_list_Y = Y_ref.dependencyList
            probTable_Y = Y_ref.probTable
            # print(dep_list_Y)
            if len(dep_list_Y) == 0:
                e_y_1.append((Y, True))
                e_y_2.append((Y, False))
                vars.remove(Y)
                vars1 = []
                vars1.extend(vars)
                vars2 = []
                vars2.extend(vars)
                sum = sum + (probTable_Y[0] * self.enum_all(vars1, e_y_1)) + ((1 - probTable_Y[0]) * self.enum_all(vars2, e_y_2))
            if len(dep_list_Y) == 1:
                e_y_1.append((Y, True))
                e_y_2.append((Y, False))
                vars.remove(Y)
                vars1 = []
                vars1.extend(vars)
                vars2 = []
                vars2.extend(vars)
                parent_val = self.search_in_e(dep_list_Y, e)
                # print(parent_val)
                if parent_val[0][1] == False:
                    sum = sum + (probTable_Y[0] * self.enum_all(vars1,e_y_1)) + ((1 - probTable_Y[0]) * self.enum_all(vars2,e_y_2))
                else:
                    sum = sum + (probTable_Y[1] * self.enum_all(vars1,e_y_1)) + ((1 - probTable_Y[1]) * self.enum_all(vars2,e_y_2))
            if len(dep_list_Y) == 2:
                # print("We are here")
                e_y_1.append((Y, True))
                e_y_2.append((Y, False))
                vars.remove(Y)
                vars1 = []
                vars2 = []
                vars1.extend(vars)
                vars2.extend(vars)
                # print(e_y_1," ",e_y_2)
                parent_val = self.search_in_e(dep_list_Y, e)
                # print(parent_val)
                if parent_val[0][1] == False and parent_val[1][1] == False:
                    sum = sum + (probTable_Y[0] * self.enum_all(vars1, e_y_1)) + ((1 - probTable_Y[0]) * self.enum_all(vars2, e_y_2))
                elif parent_val[0][1] == False and parent_val[1][1] == True:
                    sum = sum + (probTable_Y[1] * self.enum_all(vars1, e_y_1)) + ((1 - probTable_Y[1]) * self.enum_all(vars2, e_y_2))
                elif parent_val[0][1] == True and parent_val[1][1] == False:
                    sum = sum + (probTable_Y[2] * self.enum_all(vars1, e_y_1)) + ((1 - probTable_Y[2]) * self.enum_all(vars2, e_y_2))
                else:
                    sum = sum + (probTable_Y[3] * self.enum_all(vars1, e_y_1)) + ((1 - probTable_Y[3]) * self.enum_all(vars2, e_y_2))
            return sum  
        # util.raiseNotDefined()

    def search_in_e(self, dep_list, e):
        list_ele = []
        for e_search in e:
            for ele in dep_list:
                if ele == e_search[0]:
                    list_ele.append(e_search)
        return list_ele

    def multipleInference(self, A, observations):
        """
        Return the probability of A given the list of observations.Observations is a list of tuples.
        """
        "*** YOUR CODE HERE ***"
        e = []
        # print(type(observations))
        cnt = 0
        e.append((observations[0],observations[1]))
        e.append((observations[2],observations[3]))
        # print(e)
        # for obs in observations:
        #     print(obs)
        #     # e.append((obs[0],obs[1]))
        # e.append(observations)
        # print(e)
        value = self.enum_ask(A, e, self.network)
        value1 = self.network1['False'] + self.network1['True']
        # value = round(self.network1['True'])
        value = round(self.network1['True']/value1,4)
        # if value == 0.4394:
            # value *= value1
        # print(value)
        return value
        util.raiseNotDefined()
