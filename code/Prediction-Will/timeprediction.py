import csv
import time
import numpy as np
from collections import Counter

class TimePredictionModel:
    # Types of abstractions that can be used to build the model
    SEQUENCE_ABSTRACTION = 0
    SET_ABSTRACTION = 1
    MULTISET_ABSTRACTION = 2

    def __init__(self, cases=[], abstraction=SEQUENCE_ABSTRACTION, horizon=0, calendar=None):
        self.abstraction = abstraction
        # Unlimited horizon is applied if this parameter is set to 0
        self.horizon = horizon
        self.calendar = calendar
        self.states = dict()
        self.initialstate = self.addState([])
        self.build(cases)

    def addState(self, trace):
        state = self.codeState(trace)
        if state not in self.states:
            self.states[state] = []
        return state

    def codeState(self, trace):
        if self.abstraction == self.SEQUENCE_ABSTRACTION:
            # Sequence of execution matters
            state = tuple(trace)
        elif self.abstraction == self.SET_ABSTRACTION:
            # Sequence and repetitions do not matter
            state = frozenset(trace)
        elif self.abstraction == self.MULTISET_ABSTRACTION:
            # Sequence does not matter, but repetions do
            state = tuple(sorted(Counter(trace).items()))
        else:
            raise ValueError("Invalid abstraction type.")
        return state

    def build(self, cases):
        for case in cases:
            self.processCase(case)

    def processCase(self, case):
        activities, eventtimes = zip(*case)
        #print("Case activities:", activities)
        for i in range(len(case)):
            # t is the time the state is visited; e is the elapsed time since the start
            # of the case; r is the remaining flow time; s is the sojourn time, i.e., the
            # time until the next event
            t = time.mktime(eventtimes[i])
            e = self.elapsedTime(eventtimes[0], eventtimes[i])
            r = self.elapsedTime(eventtimes[i], eventtimes[-1])
            if i < len(case) - 1:
                s = self.elapsedTime(eventtimes[i], eventtimes[i+1])
            else:
                s = -1
            initial = 0
            if self.horizon > 0 and i >= self.horizon:
                # Use a limited horizon, i.e., consider the k lasts activities executed
                # (k is the value set to self.horizon)
                initial = i - self.horizon + 1
            for j in range(initial, i+1):
                state = self.addState(activities[j:i+1])
                #print("State:", state)
                self.states[state].append((t, e, r, s))
                #print("Annotations:", self.states[state])

    def elapsedTime(self, starttime, endtime):
        # TODO: Use calendar
        start = time.mktime(starttime)
        end = time.mktime(endtime)
        return end - start

    def timePredictionFunction(self, measurements):
        mean = np.mean(measurements)
        std = np.std(measurements)
        min = np.min(measurements)
        max = np.max(measurements)
        return mean, std, min, max

    def predictRemainingTime(self, partialtrace):
        #print("Predicting remaining time for partial trace", partialtrace)
        initial = 0
        if self.horizon > 0 and len(partialtrace) > self.horizon:
            initial = len(partialtrace) - self.horizon
            #print("Initial:", initial)
        while initial < len(partialtrace):
            state = self.codeState(partialtrace[initial:])
            if state in self.states:
                #print("State:", state)
                t, e, r, s = zip(*self.states[state])
                predicted = self.timePredictionFunction(r)
                #print("Predicted:", predicted)
                return predicted
            initial += 1
        # This will only happen if the partial trace contains an activity that did not
        # appear in the training set
        return self.fallThrough()

    def fallThrough(self):
        # Could not find any match in te model for the given trace, so use all
        # measures stored for states composed of a single activity
        allRemaining = []
        for state, annotations in self.states.items():
            if self.abstraction == self.MULTISET_ABSTRACTION:
                state = list(state.elements())
            if len(state) == 1:
                t, e, r, s = zip(*annotations)
                allRemaining.extend(r)
        if len(allRemaining) > 0:
            predicted = self.timePredictionFunction(allRemaining)
            return predicted
        else:
            # This will only happen if the model is empty (no case has been processed)
            print("Partial trace does not fit any state in the model. Cannot predict.")
            return None, None, None, None


def loadCases(logname, columns, timeformat):
    csvfile = open("../data/%s" % logname, 'r')
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    # Skip headers
    next(csvreader)
    cases = []
    previouscase = None
    category = None
    case = []
    # Columns are (CaseId, ActivityId, CompleteTimestamp)
    for row in csvreader:
        # Assume events are ordered by CaseID and then CompleteTimestamp in the event log
        if row[columns[0]] != previouscase:
            if len(case) > 0:
                # This is the first event for a new case
                cases.append((previouscase, category, case))
                case = []
            previouscase = row[columns[0]]
            if len(columns) > 3:
                category = row[columns[3]]
        eventtime = time.strptime(row[columns[2]], timeformat)
        case.append((row[columns[1]], eventtime))
    # Add the last case
    cases.append((previouscase, category, case))
    return cases

def splitIntoCategories(cases):
    casesets = dict()
    for caseid, category, case in cases:
        if category not in casesets:
            casesets[category] = [(caseid, case)]
        else:
            casesets[category].append((caseid, case))
    return casesets

def runTimePredictions(cases, eventlog):
    # Divide the data set into folds for model generation and prediction
    foldsize = int(round(len(cases)/3))
    trainingset = cases[:2*foldsize]
    testset = cases[2*foldsize:]

    # Build the model
    caseids, trainingset = zip(*trainingset)
    model = TimePredictionModel(trainingset, abstraction=TimePredictionModel.SEQUENCE_ABSTRACTION, horizon=8)

    # Make predictions
    with open('../results/predictions_%s' % eventlog, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["CaseID", "Prefix length", "RT ground truth", "Predicted RT", "Std Deviation", "MAE"])
        prefixlength = 2
        casestotest = True
        while casestotest:
            print("Predicting remaining time using prefix length", prefixlength)
            casestotest = False
            for caseid, case in testset:
                if len(case) > prefixlength:
                    activities, eventtimes = zip(*case)
                    #print("Predicting remaining time for case", caseid)
                    predicted, std, min, max = model.predictRemainingTime(activities[:prefixlength])
                    groundtruth = model.elapsedTime(eventtimes[prefixlength-1], eventtimes[-1])
                    if predicted is not None:
                        mae = abs(predicted - groundtruth)
                    else:
                        mae = None
                    csvwriter.writerow((caseid, prefixlength, groundtruth, predicted, std, mae))
                    casestotest = True
            prefixlength += 1

if __name__ == '__main__':
    eventlog = "1.00_preproc_WFM.csv"
    #eventlog = "helpdesk.csv"
    columns = (0, 1, 2, 3)
    timeformat = "%Y-%m-%d %H:%M:%S"
    #eventlog = "running-example.csv"
    #columns = (3, 0, 7)
    #timeformat = "%Y-%m-%d %H:%M:%S%z"
    cases = loadCases(eventlog, columns, timeformat)
    casessets = splitIntoCategories(cases)
    for category, cases in casessets.items():
        print("Predicting remaining time for case category", category)
        runTimePredictions(cases, ("cat%s_" % category) + eventlog)
